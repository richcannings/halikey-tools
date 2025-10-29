# HaliKey Oscillator

Low-latency iambic keyer software for the HaliKey v1.4 USB paddle interface. Practice CW with a sidetone or send CW over wfview using a virtual audio cable.

## Quick Start

### Installation

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Basic Usage

**Practice mode (sidetone only):**
```bash
python halikey-oscillator.py --wpm 18 --mode A
```

**Send CW over wfview (requires VB-Cable or similar):**
```bash
python halikey-oscillator.py --wpm 18 --mode A --output VB-Cable
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port PORT` | Serial port for HaliKey | `/dev/cu.usbserial-DK0E4EEM` |
| `--wpm SPEED` | CW speed in words per minute | `18` |
| `--mode A\|B` | Keyer mode (A or B) | `B` |
| `--output DEVICE` | Clean CW output device name | None |
| `--list-devices` | List available audio devices | - |

## Keyer Modes

- **Mode A (Iambic A)**: Clean alternation, no completion element
- **Mode B (Iambic B)**: Adds completion element when both paddles released during squeeze

## Audio Outputs

### Sidetone (Default Audio Device)
- 575 Hz tone with smooth attack/release envelope
- Goes to your default speakers/headphones
- Pleasant for practice

### Clean CW Output (Optional)
- Instant on/off without envelope
- Specify device with `--output`
- Perfect for feeding into wfview or logging software

## Setup for wfview

1. Install a virtual audio cable:
   - **macOS**: BlackHole or VB-Cable
   - **Windows**: VB-Cable
   - **Linux**: PulseAudio loopback

2. List available audio devices:
   ```bash
   python halikey-oscillator.py --list-devices
   ```

3. Start the keyer with clean output:
   ```bash
   python halikey-oscillator.py --wpm 20 --mode A --output VB-Cable
   ```

4. In wfview, set the audio input to the virtual cable device

5. Key away! Your CW will be transmitted through wfview

## Tips

- Use **Mode A** for most operating (cleaner, more predictable)
- Use **Mode B** if you prefer the "squeeze" completion element
- Lower WPM (12-15) for learning, higher (20-30) for QSOs
- The sidetone always plays through your default audio device for monitoring
- Press `Ctrl+C` to exit

## Requirements

- Python 3.7+
- HaliKey v1.4 (or compatible USB paddle interface)
- Virtual audio cable (only needed for wfview integration)

## Hardware

The HaliKey uses serial port control signals (CTS/DCD) for paddle detection, not data transmission. This provides ultra-low latency response.

## License

See LICENSE file for details.

