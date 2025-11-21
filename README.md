# HaliKey Tools

Low-latency tools for the [HaliKey USB paddle interface](https://electronics.halibut.com/product/halikey/). For use with Ham Radio Solutions VBand, DiDahDit, WFView, GGMorse, and more.

This repository contains two applications:

1. **halikey-oscillator.py** - Low-latency iambic keyer with sidetone
    * Send relatively clean CW over USB/LSB with [wfview](https://wfview.org/) using a virtual audio cable
    * Practice CW standalone with the sidetone or with [GGMorse](https://github.com/ggerganov/ggmorse) ggmorse-gui to decode

2. **halikey-vband.py** - Emulates the [Ham Radio Solutions VBand](https://hamradio.solutions/vband/) adapter 
    * Connect your paddle to the VBand website or other supported apps, like [DiDahDit](didahdit.com), for virtual CW practice over the internet

## Quick Start

### Installation

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### VBand Emulator Usage
The `halikey-vband.py` tool allows you to use your HaliKey paddle interface with the Ham Radio Solutions VBand virtual CW band website and other supporting the vband protocol.

**Usage:**
```bash
python halikey-vband.py --port /dev/cu.usbserial-[your port]
```

Visit [hamradio.solutions/vband](https://hamradio.solutions/vband/) to select a channel and practice CW with other operators over the internet.

### Oscillator Usage
Yes, this is a hack and inspired by a comment from W6EL in the [WFView Forums](https://forum.wfview.org/t/handling-cw/347/48).

**Practice mode (sidetone only):**
```bash
python halikey-oscillator.py --wpm 18 --port /dev/cu.usbserial-[your port]
```

**Send CW over wfview (requires VB-Cable or similar):**
```bash
python halikey-oscillator.py --wpm 18 --output VB-Cable --port /dev/cu.usbserial-[your port]
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port PORT` | Serial port for HaliKey | `/dev/cu.usbserial-DK0E4012` |
| `--wpm SPEED` | CW speed in words per minute | `18` |
| `--tone FREQ` | Tone frequency in Hz | `575` |
| `--mode A\|B` | Keyer mode (A or B) | `B` |
| `--output DEVICE` | Clean CW output device name | None |
| `--list-devices` | List available audio devices | - |
| `--verbose` | Enable latency measurements | Off |

## Additional Oscillator Information 

### Keyer Modes

- **Mode A (Iambic A)**: Clean alternation, no completion element
- **Mode B (Iambic B)**: Adds completion element when both paddles released during squeeze

### Audio Outputs

#### Sidetone (Default Audio Device)
- Default 575 Hz tone (configurable with `--tone`)
- Smooth attack/release envelope
- Goes to your default speakers/headphones
- Pleasant for practice

#### Clean CW Output (Optional)
- Same frequency as sidetone
- Instant on/off without envelope
- Specify device with `--output`
- Perfect for feeding into wfview or logging software

#### Setup for wfview

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

### Verbose Mode & Latency Measurements

Enable `--verbose` mode to see detailed debug output and latency measurements:

```bash
python halikey-oscillator.py --wpm 18 --mode A --verbose
```

**Verbose mode provides:**
- **Debug messages**: Shows each dit, dah, and state transition in real-time
- **Latency measurements**: Reports every 20 paddle events during operation
- **Exit statistics**: Displays overall latency summary when quitting

**Latency metrics include:**
- **Average latency**: Mean time from paddle detection to audio generation
- **Min latency**: Best-case latency observed
- **Max latency**: Worst-case latency observed
- **Total events**: Number of paddle events measured

**Note**: These measurements show the software processing time only (typically 0.01-0.1ms). Total perceived latency also includes:
- Serial port polling delays (~1-5ms)
- Audio buffer latency (configured for low latency)
- OS audio stack delays (~5-20ms)

Typical total system latency: 10-30ms

### Tips

- Use **Mode A** for most operating (cleaner, more predictable)
- Use **Mode B** if you prefer the "squeeze" completion element
- Lower WPM (12-15) for learning, higher (20-30) for QSOs
- The sidetone always plays through your default audio device for monitoring
- Use `--verbose` to monitor system performance and latency
- Press **'q'** to quit (or Ctrl+C)

## Requirements

- Python 3.7+
- HaliKey v1.4 (or compatible USB paddle interface)
- Virtual audio cable (only needed for wfview integration)

## Hardware

The HaliKey uses serial port control signals (CTS/DCD) for paddle detection, not data transmission. This provides ultra-low latency response.

## License

See LICENSE file for details.
