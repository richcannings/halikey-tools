#!/usr/bin/env python3
"""
halikey_oscillator.py
---------------------
Generates a 575 Hz sidetone from a HaliKey v1.4 iambic keyer.
Supports both Iambic A and Iambic B modes.
Optionally outputs clean CW to a second audio device (e.g., VB-Cable).

Usage:
    # Basic usage (sidetone only)
    python3 halikey_oscillator.py --port /dev/cu.usbserial-DK0E4EEM --wpm 18 --mode B
    
    # With clean CW output to VB-Cable
    python3 halikey_oscillator.py --wpm 18 --mode A --output VB-Cable
    
    # List available audio devices
    python3 halikey_oscillator.py --list-devices

Defaults:
    --port /dev/cu.usbserial-DK0E4EEM
    --wpm 18
    --mode B
    --output None (no clean output)

Modes:
    A: Iambic A (no completion element)
    B: Iambic B (adds completion element on squeeze)

Outputs:
    - Sidetone: Default audio device with attack/release envelope
    - Clean CW: Optional second device with instant on/off (no envelope)

Dependencies:
    pip install sounddevice numpy pyserial
"""

import argparse
import time
import math
import numpy as np
import sounddevice as sd
import serial
import serial.tools.list_ports

# ------------------- Configuration -------------------

DEFAULT_PORT = "/dev/cu.usbserial-DK0E4EEM"
DEFAULT_WPM = 18
DEFAULT_MODE = "B"  # Iambic mode: A or B
DEFAULT_OUTPUT_DEVICE = "VB-Cable"  # Default clean output device
TONE_FREQ = 575.0
SAMPLE_RATE = 48000
ATTACK = 0.002  # For sidetone envelope
RELEASE = 0.003  # For sidetone envelope
GAIN = 0.3
BLOCK_SIZE = 64  # Smaller block size for lower latency

# ------------------- Timing -------------------

def dit_length(wpm):
    """Calculate dit duration in seconds from WPM"""
    return 1.2 / wpm

# ------------------- Streaming Audio Generator -------------------

class CWGenerator:
    """Low-latency streaming CW tone generator"""
    
    def __init__(self, freq, sample_rate, attack, release, gain):
        self.attack_samples = int(attack * sample_rate)
        self.release_samples = int(release * sample_rate)
        self.gain = gain
        self.phase = 0.0
        self.amplitude = 0.0
        self.target_amplitude = 0.0
        self.phase_increment = 2.0 * math.pi * freq / sample_rate
        
    def set_keyed(self, keyed):
        """Set whether the key is down"""
        self.target_amplitude = self.gain if keyed else 0.0
    
    def generate(self, frames):
        """Generate audio samples with smooth attack/release"""
        output = np.zeros(frames, dtype=np.float32)
        
        for i in range(frames):
            # Smooth amplitude transitions
            if self.amplitude < self.target_amplitude:
                self.amplitude += self.gain / max(1, self.attack_samples)
                self.amplitude = min(self.amplitude, self.target_amplitude)
            elif self.amplitude > self.target_amplitude:
                self.amplitude -= self.gain / max(1, self.release_samples)
                self.amplitude = max(self.amplitude, self.target_amplitude)
            
            # Generate sine wave
            output[i] = self.amplitude * math.sin(self.phase)
            self.phase += self.phase_increment
            
            # Keep phase in range
            if self.phase >= 2.0 * math.pi:
                self.phase -= 2.0 * math.pi
        
        return output

class CleanCWGenerator:
    """Clean CW tone generator without envelope for external output"""
    
    def __init__(self, freq, sample_rate, gain):
        self.gain = gain
        self.phase = 0.0
        self.keyed = False
        self.phase_increment = 2.0 * math.pi * freq / sample_rate
        
    def set_keyed(self, keyed):
        """Set whether the key is down"""
        self.keyed = keyed
    
    def generate(self, frames):
        """Generate clean square-wave keyed tone"""
        output = np.zeros(frames, dtype=np.float32)
        
        if self.keyed:
            for i in range(frames):
                output[i] = self.gain * math.sin(self.phase)
                self.phase += self.phase_increment
                if self.phase >= 2.0 * math.pi:
                    self.phase -= 2.0 * math.pi
        
        return output

# ------------------- HaliKey Serial Reader -------------------

def run_halikey(port, wpm, mode, output_device=None):
    try:
        ser = serial.Serial(port, 115200, timeout=0.001)
        print(f"[INFO] Connected to {ser.port}")
    except Exception as e:
        print(f"[ERROR] Could not open serial {port}: {e}")
        print("\n[INFO] Available serial ports:")
        ports = serial.tools.list_ports.comports()
        for p in ports:
            print(f"  - {p.device}: {p.description}")
        return

    dit = dit_length(wpm)
    dah = 3 * dit
    inter_element = dit
    
    # Validate and normalize mode
    mode = mode.upper()
    if mode not in ['A', 'B']:
        print(f"[ERROR] Invalid mode '{mode}'. Must be 'A' or 'B'. Defaulting to 'B'.")
        mode = 'B'
    
    print(f"[INFO] WPM={wpm}  Dit={dit:.3f}s  Dah={dah:.3f}s")
    print(f"[INFO] Mode: Iambic {mode}")

    # Create audio generators
    sidetone_gen = CWGenerator(TONE_FREQ, SAMPLE_RATE, ATTACK, RELEASE, GAIN)
    
    # Sidetone audio callback
    def sidetone_callback(outdata, frames, time_info, status):
        if status:
            print(f"[WARN] Sidetone audio status: {status}")
        outdata[:, 0] = sidetone_gen.generate(frames)
    
    # Start sidetone stream (default audio output)
    sidetone_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        callback=sidetone_callback,
        latency='low'
    )
    sidetone_stream.start()
    print("[INFO] Sidetone output started (default audio device)")
    
    # Setup clean output if device specified
    clean_stream = None
    clean_gen = None
    if output_device:
        try:
            # Find the output device
            devices = sd.query_devices()
            device_id = None
            
            for i, dev in enumerate(devices):
                if output_device.lower() in dev['name'].lower() and dev['max_output_channels'] > 0:
                    device_id = i
                    break
            
            if device_id is not None:
                clean_gen = CleanCWGenerator(TONE_FREQ, SAMPLE_RATE, GAIN)
                
                def clean_callback(outdata, frames, time_info, status):
                    if status:
                        print(f"[WARN] Clean output audio status: {status}")
                    outdata[:, 0] = clean_gen.generate(frames)
                
                clean_stream = sd.OutputStream(
                    device=device_id,
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    channels=1,
                    callback=clean_callback,
                    latency='low'
                )
                clean_stream.start()
                print(f"[INFO] Clean CW output started: {devices[device_id]['name']}")
            else:
                print(f"[WARN] Output device '{output_device}' not found. Available devices:")
                for i, dev in enumerate(devices):
                    if dev['max_output_channels'] > 0:
                        print(f"  [{i}] {dev['name']}")
        except Exception as e:
            print(f"[WARN] Could not setup clean output: {e}")
    
    print(f"[INFO] Iambic {mode} keyer active... Ctrl+C to quit.\n")

    # Keyer state machine
    IDLE, DIT, DAH, DIT_WAIT, DAH_WAIT = 0, 1, 2, 3, 4
    
    state = IDLE
    dit_latch = False
    dah_latch = False
    iambic_latch = False
    element_start_time = 0
    
    def read_paddles():
        """Read paddle states: CTS=dit, DCD=dah"""
        return ser.cts, ser.cd
    
    try:
        while True:
            dit_paddle, dah_paddle = read_paddles()
            current_time = time.perf_counter()  # High-resolution timer
            
            # State machine
            if state == IDLE:
                # Not sending anything, check for paddle input
                if dit_paddle and dah_paddle:
                    sidetone_gen.set_keyed(True)
                    if clean_gen:
                        clean_gen.set_keyed(True)
                    state = DIT
                    element_start_time = current_time
                    print("[DEBUG] Squeeze: Dit")
                elif dit_paddle:
                    sidetone_gen.set_keyed(True)
                    if clean_gen:
                        clean_gen.set_keyed(True)
                    state = DIT
                    element_start_time = current_time
                    print("[DEBUG] Dit")
                elif dah_paddle:
                    sidetone_gen.set_keyed(True)
                    if clean_gen:
                        clean_gen.set_keyed(True)
                    state = DAH
                    element_start_time = current_time
                    print("[DEBUG] Dah")
                else:
                    time.sleep(0.0001)
                    
            elif state == DIT:
                # Sending dit - latch opposite paddle only
                if dah_paddle:
                    dah_latch = True
                    
                if current_time - element_start_time >= dit:
                    sidetone_gen.set_keyed(False)
                    if clean_gen:
                        clean_gen.set_keyed(False)
                    state = DIT_WAIT
                    element_start_time = current_time
                else:
                    time.sleep(0.0001)
                    
            elif state == DAH:
                # Sending dah - latch opposite paddle only
                if dit_paddle:
                    dit_latch = True
                    
                if current_time - element_start_time >= dah:
                    sidetone_gen.set_keyed(False)
                    if clean_gen:
                        clean_gen.set_keyed(False)
                    state = DAH_WAIT
                    element_start_time = current_time
                else:
                    time.sleep(0.0001)
                    
            elif state == DIT_WAIT:
                if current_time - element_start_time >= inter_element:
                    dit_paddle, dah_paddle = read_paddles()
                    
                    if dah_latch:
                        dah_latch = False
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DAH
                        element_start_time = current_time
                        if mode == 'B' and dit_paddle and dah_paddle and not iambic_latch:
                            iambic_latch = True
                            print("[DEBUG] Dah (latched, iambic B armed)")
                        else:
                            iambic_latch = False
                            print("[DEBUG] Dah (latched)")
                    elif iambic_latch and not dit_paddle and not dah_paddle:
                        iambic_latch = False
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DAH
                        element_start_time = current_time
                        print("[DEBUG] Dah (iambic B completion)")
                    elif iambic_latch:
                        iambic_latch = False
                        if dit_paddle:
                            sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                            state = DIT
                            element_start_time = current_time
                            print("[DEBUG] Dit (continued, iambic B cleared)")
                        elif dah_paddle:
                            sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                            state = DAH
                            element_start_time = current_time
                            print("[DEBUG] Dah (iambic B cleared)")
                    elif dit_paddle:
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DIT
                        element_start_time = current_time
                        print("[DEBUG] Dit (continued)")
                    elif dah_paddle:
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DAH
                        element_start_time = current_time
                        print("[DEBUG] Dah")
                    else:
                        state = IDLE
                        dit_latch = False
                        dah_latch = False
                        iambic_latch = False
                else:
                    time.sleep(0.0001)
                    
            elif state == DAH_WAIT:
                if current_time - element_start_time >= inter_element:
                    dit_paddle, dah_paddle = read_paddles()
                    
                    if dit_latch:
                        dit_latch = False
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DIT
                        element_start_time = current_time
                        if mode == 'B' and dit_paddle and dah_paddle and not iambic_latch:
                            iambic_latch = True
                            print("[DEBUG] Dit (latched, iambic B armed)")
                        else:
                            iambic_latch = False
                            print("[DEBUG] Dit (latched)")
                    elif iambic_latch and not dit_paddle and not dah_paddle:
                        iambic_latch = False
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DIT
                        element_start_time = current_time
                        print("[DEBUG] Dit (iambic B completion)")
                    elif iambic_latch:
                        iambic_latch = False
                        if dah_paddle:
                            sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                            state = DAH
                            element_start_time = current_time
                            print("[DEBUG] Dah (continued, iambic B cleared)")
                        elif dit_paddle:
                            sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                            state = DIT
                            element_start_time = current_time
                            print("[DEBUG] Dit (iambic B cleared)")
                    elif dah_paddle:
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DAH
                        element_start_time = current_time
                        print("[DEBUG] Dah (continued)")
                    elif dit_paddle:
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DIT
                        element_start_time = current_time
                        print("[DEBUG] Dit")
                    else:
                        state = IDLE
                        dit_latch = False
                        dah_latch = False
                        iambic_latch = False
                else:
                    time.sleep(0.0001)
            
    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")
    finally:
        sidetone_stream.stop()
        sidetone_stream.close()
        if clean_stream:
            clean_stream.stop()
            clean_stream.close()
        ser.close()

# ------------------- Entry Point -------------------

def main():
    parser = argparse.ArgumentParser(description="HaliKey v1.4 CW Sidetone Generator - Iambic Keyer")
    parser.add_argument("--port", type=str, default=DEFAULT_PORT, help="Serial port for HaliKey")
    parser.add_argument("--wpm", type=int, default=DEFAULT_WPM, help="CW speed in WPM")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=['A', 'B', 'a', 'b'], 
                        help="Keyer mode: A (iambic A) or B (iambic B, default)")
    parser.add_argument("--output", type=str, default=None, 
                        help=f"Clean CW output device name (default: None, use '{DEFAULT_OUTPUT_DEVICE}' for VB-Cable)")
    parser.add_argument("--list-devices", action="store_true", 
                        help="List available audio output devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print("Available audio output devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                default = " (DEFAULT)" if i == sd.default.device[1] else ""
                print(f"  [{i}] {dev['name']}{default}")
        return

    run_halikey(args.port, args.wpm, args.mode, args.output)

if __name__ == "__main__":
    main()

