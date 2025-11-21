#!/usr/bin/env python3
"""
halikey-vband.py
----------------
Converts HaliKey USB paddle interface to keyboard control keys for vband.
Left paddle (dit) -> Left Control key
Right paddle (dah) -> Right Control key

This is a direct paddle-to-key mapping with no keyer logic.
Pressing a paddle simulates pressing the key, releasing releases it.

Usage:
    # Basic usage with default port
    python3 halikey-vband.py
    
    # Specify serial port
    python3 halikey-vband.py --port /dev/cu.usbserial-DK0E4012
    
    # Enable verbose debug output
    python3 halikey-vband.py --verbose
    
    # List available serial ports
    python3 halikey-vband.py --list-ports

Defaults:
    --port /dev/cu.usbserial-DK0E4012

Dependencies:
    pip install pyserial pynput
"""

import argparse
import time
import sys
import select
import termios
import tty
import serial
import serial.tools.list_ports
from pynput.keyboard import Controller, Key

# ------------------- Configuration -------------------

DEFAULT_PORT = "/dev/cu.usbserial-DK0E4012"
POLL_INTERVAL = 0.01  # 10ms polling for low latency

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

# ------------------- HaliKey to Virtual Keyboard -------------------

def run_vband(port, verbose=False):
    """Main loop: read paddles and simulate keyboard"""
    try:
        ser = serial.Serial(port, 115200, timeout=POLL_INTERVAL)
        print(f"[INFO] Connected to {ser.port}")
    except Exception as e:
        print(f"[ERROR] Could not open serial {port}: {e}")
        print("\n[INFO] Available serial ports:")
        ports = serial.tools.list_ports.comports()
        for p in ports:
            print(f"  - {p.device}: {p.description}")
        return

    # Initialize virtual keyboard controller
    keyboard = Controller()
    
    print("[INFO] HaliKey Virtual Keyboard active")
    print("[INFO] Left paddle (dit) -> Left Control")
    print("[INFO] Right paddle (dah) -> Right Control")
    print("[INFO]")
    print("[INFO] IMPORTANT: macOS requires Accessibility permissions!")
    print("[INFO] Go to: System Settings > Privacy & Security > Accessibility")
    print("[INFO] Add Terminal or your Python app to the allowed list.")
    print("[INFO]")
    print("[INFO] Press 'q' to quit.\n")
    
    # Track key states to prevent repeated press/release
    left_ctrl_pressed = False
    right_ctrl_pressed = False

    # Setup terminal for keyboard input
    old_terminal_settings = setup_terminal()
    
    try:
        while True:
            # Check for quit key
            if check_quit_key():
                print_raw("")
                print_raw("[INFO] 'q' pressed, exiting...")
                break
            
            # Read current paddle states
            dah_paddle_pressed = ser.cd
            dit_paddle_pressed = ser.cts
            
            if dah_paddle_pressed and not right_ctrl_pressed:
                keyboard.press(Key.ctrl_r)
                right_ctrl_pressed = True
                if verbose:
                    print_raw("[DEBUG] Right Control pressed")
            elif not dah_paddle_pressed and right_ctrl_pressed:
                keyboard.release(Key.ctrl_r)
                right_ctrl_pressed = False
                if verbose:
                    print_raw("[DEBUG] Right Control released")

            if dit_paddle_pressed and not left_ctrl_pressed:
                keyboard.press(Key.ctrl_l)
                left_ctrl_pressed = True
                if verbose:
                    print_raw("[DEBUG] Left Control pressed")
            elif not dit_paddle_pressed and left_ctrl_pressed:
                keyboard.release(Key.ctrl_l)
                left_ctrl_pressed = False
                if verbose:
                    print_raw("[DEBUG] Left Control released")

            # Workaround for VBand not releasing keys when both paddles are released
            if not dit_paddle_pressed and not dah_paddle_pressed:
                keyboard.release(Key.ctrl_l)
                keyboard.release(Key.ctrl_r)
                left_ctrl_pressed = False
                right_ctrl_pressed = False
                if verbose:
                    print_raw("[DEBUG] All Controls released")
            
            # Small sleep to prevent CPU spinning
            time.sleep(POLL_INTERVAL)
            
    except KeyboardInterrupt:
        print_raw("\n[INFO] Ctrl+C pressed, exiting...")
    finally:
        # Make sure all keys are released
        if left_ctrl_pressed:
            keyboard.release(Key.ctrl_l)
            if verbose:
                print_raw("[DEBUG] Released Left Control on exit")
        if right_ctrl_pressed:
            keyboard.release(Key.ctrl_r)
            if verbose:
                print_raw("[DEBUG] Released Right Control on exit")
        
        # Restore terminal settings
        restore_terminal(old_terminal_settings)
        
        # Close serial port
        ser.close()

# ------------------- Entry Point -------------------

def main():
    parser = argparse.ArgumentParser(description="HaliKey Virtual Keyboard - Paddle to Control Keys")
    parser.add_argument("--port", type=str, default=DEFAULT_PORT, help="Serial port for HaliKey")
    parser.add_argument("--list-ports", action="store_true", 
                        help="List available serial ports and exit")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose mode with debug messages")
    args = parser.parse_args()

    if args.list_ports:
        print("Available serial ports:")
        ports = serial.tools.list_ports.comports()
        if ports:
            for p in ports:
                print(f"  - {p.device}: {p.description}")
        else:
            print("  No serial ports found")
        return

    run_vband(args.port, args.verbose)

if __name__ == "__main__":
    main()

