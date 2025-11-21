#!/usr/bin/env python3
"""
halikey_oscillator.py
---------------------
A keyer that generates a sidetone and a clean oscillator from a HaliKey USB paddle interface.
Supports both Iambic A and Iambic B modes.
Optionally outputs clean CW to a second audio device (e.g., VB-Cable).

Usage:
    # Basic usage (sidetone only)
    python3 halikey_oscillator.py --port /dev/cu.usbserial-DK0E4012 --wpm 18 --mode B
    
    # With clean CW output to VB-Cable
    python3 halikey_oscillator.py --wpm 18 --mode A --output VB-Cable
    
    # Custom tone frequency
    python3 halikey_oscillator.py --wpm 20 --tone 700 --output VB-Cable
    
    # List available audio devices
    python3 halikey_oscillator.py --list-devices
    
    # Enable verbose mode with latency measurements
    python3 halikey_oscillator.py --wpm 18 --mode A --verbose

Defaults:
    --port /dev/cu.usbserial-DK0E4012
    --wpm 18
    --tone 575
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
import sys
import select
import termios
import tty
import numpy as np
import sounddevice as sd
import serial
import serial.tools.list_ports

# ------------------- Configuration -------------------

DEFAULT_PORT = "/dev/cu.usbserial-DK0E4012"
DEFAULT_WPM = 18
DEFAULT_MODE = "B"  # Iambic mode: A or B
DEFAULT_OUTPUT_DEVICE = "VB-Cable"  # Default clean output device
DEFAULT_TONE_FREQ = 575  # Hz
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

# ------------------- Keyboard Input Helper -------------------

def setup_terminal():
    """Setup terminal for non-blocking keyboard input"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)
    return old_settings

def restore_terminal(old_settings):
    """Restore terminal to original settings"""
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def check_quit_key():
    """Check if 'q' key has been pressed (non-blocking)"""
    if select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch.lower() == 'q':
            return True
    return False

def print_raw(msg):
    """Print with proper line endings for raw terminal mode"""
    sys.stdout.write(msg.replace('\n', '\r\n') + '\r\n')
    sys.stdout.flush()

# ------------------- HaliKey Serial Reader -------------------

def run_halikey(port, wpm, mode, tone_freq, output_device=None, verbose=False):
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
    print(f"[INFO] Tone: {tone_freq} Hz")
    if verbose:
        print(f"[INFO] Verbose mode: Latency measurements enabled")

    # Create audio generators
    sidetone_gen = CWGenerator(tone_freq, SAMPLE_RATE, ATTACK, RELEASE, GAIN)
    
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
                clean_gen = CleanCWGenerator(tone_freq, SAMPLE_RATE, GAIN)
                
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
    
    print(f"[INFO] Iambic {mode} keyer active... Press 'q' to quit.\n")

    # Keyer state machine
    IDLE, DIT, DAH, DIT_WAIT, DAH_WAIT = 0, 1, 2, 3, 4
    
    state = IDLE
    dit_latch = False
    dah_latch = False
    iambic_latch = False
    element_start_time = 0
    
    # Latency tracking for verbose mode
    latency_samples = []
    latency_count = 0
    LATENCY_REPORT_INTERVAL = 20  # Report every N paddle events
    
    def read_paddles():
        """Read paddle states: CTS=dit, DCD=dah"""
        return ser.cts, ser.cd
    
    # Setup terminal for keyboard input
    old_terminal_settings = setup_terminal()
    
    try:
        while True:
            # Check for quit key
            if check_quit_key():
                print_raw("")
                print_raw("[INFO] 'q' pressed, exiting...")
                
                # Print overall latency statistics if verbose mode was enabled
                if verbose and latency_samples:
                    print_raw("")
                    print_raw("=== Latency Statistics ===")
                    avg_latency = sum(latency_samples) / len(latency_samples)
                    min_latency = min(latency_samples)
                    max_latency = max(latency_samples)
                    print_raw(f"Total events: {len(latency_samples)}")
                    print_raw(f"Average latency: {avg_latency:.3f}ms")
                    print_raw(f"Min latency: {min_latency:.3f}ms")
                    print_raw(f"Max latency: {max_latency:.3f}ms")
                    print_raw("")
                
                break
            
            dit_paddle, dah_paddle = read_paddles()
            current_time = time.perf_counter()  # High-resolution timer
            
            # Report latency statistics periodically
            if verbose and latency_count > 0 and latency_count % LATENCY_REPORT_INTERVAL == 0:
                avg_latency = sum(latency_samples[-LATENCY_REPORT_INTERVAL:]) / LATENCY_REPORT_INTERVAL
                min_latency = min(latency_samples[-LATENCY_REPORT_INTERVAL:])
                max_latency = max(latency_samples[-LATENCY_REPORT_INTERVAL:])
                print_raw(f"[LATENCY] Last {LATENCY_REPORT_INTERVAL} events: avg={avg_latency:.3f}ms, min={min_latency:.3f}ms, max={max_latency:.3f}ms")
            
            # State machine
            if state == IDLE:
                # Not sending anything, check for paddle input
                if dit_paddle and dah_paddle:
                    paddle_detect_time = time.perf_counter()
                    sidetone_gen.set_keyed(True)
                    if clean_gen:
                        clean_gen.set_keyed(True)
                    audio_keyed_time = time.perf_counter()
                    
                    if verbose:
                        latency_ms = (audio_keyed_time - paddle_detect_time) * 1000
                        latency_samples.append(latency_ms)
                        latency_count += 1
                    
                    state = DIT
                    element_start_time = current_time
                    if verbose:
                        print_raw("[DEBUG] Squeeze: Dit")
                elif dit_paddle:
                    paddle_detect_time = time.perf_counter()
                    sidetone_gen.set_keyed(True)
                    if clean_gen:
                        clean_gen.set_keyed(True)
                    audio_keyed_time = time.perf_counter()
                    
                    if verbose:
                        latency_ms = (audio_keyed_time - paddle_detect_time) * 1000
                        latency_samples.append(latency_ms)
                        latency_count += 1
                    
                    state = DIT
                    element_start_time = current_time
                    if verbose:
                        print_raw("[DEBUG] Dit")
                elif dah_paddle:
                    paddle_detect_time = time.perf_counter()
                    sidetone_gen.set_keyed(True)
                    if clean_gen:
                        clean_gen.set_keyed(True)
                    audio_keyed_time = time.perf_counter()
                    
                    if verbose:
                        latency_ms = (audio_keyed_time - paddle_detect_time) * 1000
                        latency_samples.append(latency_ms)
                        latency_count += 1
                    
                    state = DAH
                    element_start_time = current_time
                    if verbose:
                        print_raw("[DEBUG] Dah")
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
                            if verbose:
                                print_raw("[DEBUG] Dah (latched, iambic B armed)")
                        else:
                            iambic_latch = False
                            if verbose:
                                print_raw("[DEBUG] Dah (latched)")
                    elif iambic_latch and not dit_paddle and not dah_paddle:
                        iambic_latch = False
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DAH
                        element_start_time = current_time
                        if verbose:
                            print_raw("[DEBUG] Dah (iambic B completion)")
                    elif iambic_latch:
                        iambic_latch = False
                        if dit_paddle:
                            sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                            state = DIT
                            element_start_time = current_time
                            if verbose:
                                print_raw("[DEBUG] Dit (continued, iambic B cleared)")
                        elif dah_paddle:
                            sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                            state = DAH
                            element_start_time = current_time
                            if verbose:
                                print_raw("[DEBUG] Dah (iambic B cleared)")
                    elif dit_paddle:
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DIT
                        element_start_time = current_time
                        if verbose:
                            print_raw("[DEBUG] Dit (continued)")
                    elif dah_paddle:
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DAH
                        element_start_time = current_time
                        if verbose:
                            print_raw("[DEBUG] Dah")
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
                            if verbose:
                                print_raw("[DEBUG] Dit (latched, iambic B armed)")
                        else:
                            iambic_latch = False
                            if verbose:
                                print_raw("[DEBUG] Dit (latched)")
                    elif iambic_latch and not dit_paddle and not dah_paddle:
                        iambic_latch = False
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DIT
                        element_start_time = current_time
                        if verbose:
                            print_raw("[DEBUG] Dit (iambic B completion)")
                    elif iambic_latch:
                        iambic_latch = False
                        if dah_paddle:
                            sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                            state = DAH
                            element_start_time = current_time
                            if verbose:
                                print_raw("[DEBUG] Dah (continued, iambic B cleared)")
                        elif dit_paddle:
                            sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                            state = DIT
                            element_start_time = current_time
                            if verbose:
                                print_raw("[DEBUG] Dit (iambic B cleared)")
                    elif dah_paddle:
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DAH
                        element_start_time = current_time
                        if verbose:
                            print_raw("[DEBUG] Dah (continued)")
                    elif dit_paddle:
                        sidetone_gen.set_keyed(True)
                        if clean_gen:
                            clean_gen.set_keyed(True)
                        state = DIT
                        element_start_time = current_time
                        if verbose:
                            print_raw("[DEBUG] Dit")
                    else:
                        state = IDLE
                        dit_latch = False
                        dah_latch = False
                        iambic_latch = False
                else:
                    time.sleep(0.0001)
            
    except KeyboardInterrupt:
        print_raw("[INFO] Ctrl+C pressed, exiting...")
    finally:
        # Print overall latency statistics if verbose mode was enabled
        if verbose and latency_samples:
            print_raw("")
            print_raw("=== Latency Statistics ===")
            avg_latency = sum(latency_samples) / len(latency_samples)
            min_latency = min(latency_samples)
            max_latency = max(latency_samples)
            print_raw(f"Total events: {len(latency_samples)}")
            print_raw(f"Average latency: {avg_latency:.3f}ms")
            print_raw(f"Min latency: {min_latency:.3f}ms")
            print_raw(f"Max latency: {max_latency:.3f}ms")
            print_raw("")
        
        # Restore terminal settings
        restore_terminal(old_terminal_settings)
        # Clean up audio streams
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
    parser.add_argument("--tone", type=int, default=DEFAULT_TONE_FREQ, help="Tone frequency in Hz")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=['A', 'B', 'a', 'b'], 
                        help="Keyer mode: A (iambic A) or B (iambic B, default)")
    parser.add_argument("--output", type=str, default=None, 
                        help=f"Clean CW output device name (default: None, use '{DEFAULT_OUTPUT_DEVICE}' for VB-Cable)")
    parser.add_argument("--list-devices", action="store_true", 
                        help="List available audio output devices and exit")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose mode with latency measurements")
    args = parser.parse_args()

    if args.list_devices:
        print("Available audio output devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                default = " (DEFAULT)" if i == sd.default.device[1] else ""
                print(f"  [{i}] {dev['name']}{default}")
        return

    run_halikey(args.port, args.wpm, args.mode, args.tone, args.output, args.verbose)

if __name__ == "__main__":
    main()

