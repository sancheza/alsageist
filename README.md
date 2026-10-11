# AlsaGeist

Autonomous, voice-driven conversational animatronic Halloween display engine powered by offline Piper Neural Text-to-Speech (TTS), real-time Voice Activity Detection (VAD), and an offline zero-latency banter engine.

Designed for cross-platform operation on macOS development workstations and Raspberry Pi hardware with the Google AIY Voice Kit V1.

---

## 1. Overview

AlsaGeist runs an autonomous conversational character ("Grimjaw the Crypt Skeleton") that guards a candy bowl. It captures microphone audio in memory, detects completed speech boundaries via energy-based VAD, matches visitor utterances against an offline intent library, synthesizes character audio locally via Piper neural models in RAM, and drives an illuminated arcade status LED.

### Key Capabilities
* **Zero Disk I/O:** Audio capture, VAD framing, and speech synthesis operate directly in memory as raw PCM numpy arrays.
* **Offline Zero-Latency Dialogue:** Over 110 pre-programmed intents, pop-culture quotes, skeleton puns, and boundary rules respond in under 5 milliseconds.
* **Local Neural Speech:** Synthesizes gravelly character speech locally via Piper ONNX without cloud TTS APIs or subscription limits.
* **Cross-Platform:** Runs in terminal emulation mode on macOS or hardware GPIO/ALSA mode on Raspberry Pi.

---

## 2. System Requirements

### Development (macOS)
* macOS 12+ (Apple Silicon or Intel)
* Python 3.9+
* Built-in or USB microphone and speaker

### Production Hardware (Raspberry Pi)
* Raspberry Pi 3 Model B, 3B+, 4, or 5
* Google AIY Voice Kit V1 (2017 Voice HAT with stereo microphone daughterboard, 3-inch speaker, and arcade button)
* Raspberry Pi OS Lite (64-bit)
* 5V 2.5A+ power supply

---

## 3. Installation

### Objective: Set up a local Python virtual environment and install dependencies.

```bash
git clone https://github.com/sancheza/alsageist.git
cd alsageist
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

## 4. Downloading Voice Models

### Objective: Download the Piper ONNX voice weights into the `models/` directory.

Run the model downloader script:

```bash
python3 scripts/download_model.py --voice alan
```

Expected output:
```text
Downloading https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx -> models/piper/en_GB-alan-medium.onnx
Successfully saved models/piper/en_GB-alan-medium.onnx
```

---

## 5. Configuration & Environment Variables

AlsaGeist uses offline intent matching and local TTS by default. Speech-to-text requires an API key for Groq Whisper or Gemini Flash:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Groq API key for high-speed Whisper Large v3 STT | None |
| `GEMINI_API_KEY` | Google Gemini API key for Gemini Flash STT | None |

Configure your key in a `.env` file:

```ini
GROQ_API_KEY="your-api-key-here"
```

Or export it directly in your environment:

```bash
export GROQ_API_KEY="your-api-key-here"
```

---

## 6. Usage & Running the Display

### Objective: Start the interactive display daemon.

Run the main application:

```bash
./main.py
```

Expected output:
```text
======================================================================
 💀 ALSA-GEIST CRYPT SKELETON IS ACTIVE & LISTENING
 Engine: INTENT | Voice: Piper Alan (Local ONNX)
 Interaction Logging: Active -> logs/interactions.jsonl
 Speak into your microphone to start a conversation.
 Press Ctrl+C to stop.
======================================================================

🟢 [LED: IDLE (Breathing ~ ~ ~)]
```

Speak into your microphone. Grimjaw will detect your voice, transcribe the input, match the intent, synthesize speech, and play the response while pulsing the LED.

---

## 7. Command-Line Options

Inspect all runtime flags:

```bash
./main.py --help
```

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-v, --version` | Display application version and exit | Current version |
| `--verbose` | Enable verbose diagnostic logging | False |
| `--engine` | Dialogue engine: `intent` (offline) or `llm` (cloud) | `intent` |
| `--model-path` | Path to the Piper ONNX model file | `models/piper/en_GB-alan-medium.onnx` |
| `--input-device` | Input audio hardware device name or index | System default |
| `--output-device` | Output audio hardware device name or index | System default |
| `--volume` | Audio playback volume multiplier (`0.0` to `1.0`) | `0.8` (or `$VOLUME`) |
| `--mic-gain` | Microphone digital input gain multiplier | `1.5` (or `$MIC_GAIN`) |
| `--threshold` | RMS audio energy threshold for speech detection | `0.0035` (or `$VAD_THRESHOLD`) |
| `--min-speech-sec` | Minimum sustained speech duration in seconds | `0.35` |
| `--silence-timeout` | Trailing silence in seconds before speech boundary | `1.1` |
| `--tempo` | Piper phoneme length scale (higher = slower speech) | `1.15` |
| `--noise` | Piper generator noise level for gravel texture | `0.85` |

---

## 8. Conversational Intent Architecture

The offline dialogue engine in `services/intent_service.py` provides deterministic responses across over 110 categories:
* **Candy Protection & Bowl Rules:** Limits visitors to one piece, detects theft inquiries, and warns against greedy hands.
* **Classic Pop Culture & Cinema Quotes:** *I'm Gonna Git You Sucka*, *Blade Runner* (Tears in Rain), *Say Anything*, *Ali G*, *He-Man & Skeletor*.
* **Poetry & Music:** Edgar Allan Poe's *The Raven*, Robert Browning's *Childe Roland*, Michael Jackson's *Thriller*, *Monster Mash*, *The Time Warp*.
* **Anatomy & Skeleton Puns:** Dynamic banter regarding bones, teeth, absent organs, zero-calorie diets, and cold weather.
* **Visitor Prompts Reference:** See [`docs/custom_responses.md`](docs/custom_responses.md) for a catalog of supported spoken prompts.

---

## 9. Testing

### Objective: Run the automated test suite.

Execute pytest across all unit tests:

```bash
pytest tests/
```

Expected output:
```text
============================== 21 passed in 3.05s ==============================
```

---

## 10. Raspberry Pi & Google AIY Voice Kit Deployment

For a complete end-to-end guide covering hardware selection, modern Pi 4/5 alternatives, audio configurations, and systemd deployment, see the [Raspberry Pi Installation & Deployment Guide](docs/raspberry_pi_installation_guide.md).

### Step 1: Operating System Preparation
Flash **Raspberry Pi OS Lite (64-bit)** to a microSD card.

### Step 2: Enable Google AIY Voice HAT Driver
Edit `/boot/firmware/config.txt` (or `/boot/config.txt` on older releases):

```ini
# Disable default Broadcom 3.5mm/HDMI audio
dtparam=audio=off

# Enable Google AIY Voice HAT V1 driver overlay
dtoverlay=googlevoicehat-soundcard

# Enable I2S interface
dtparam=i2s=on
```

Reboot the Raspberry Pi:

```bash
sudo reboot
```

Verify audio registration:
```bash
aplay -l
arecord -l
```
Both commands must display `card 0: sndrpigooglevoi`.

### Step 3: Install Linux System Packages
```bash
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv portaudio19-dev libasound2-dev libatlas-base-dev alsa-utils
```

### Step 4: Configure Full-Duplex ALSA
Write `/etc/asound.conf` to enable concurrent microphone capture and speaker playback:

```ini
pcm.!default {
    type asym
    playback.pcm {
        type plug
        slave.pcm "hw:sndrpigooglevoi,0"
    }
    capture.pcm {
        type plug
        slave.pcm "hw:sndrpigooglevoi,0"
    }
}

ctl.!default {
    type hw
    card sndrpigooglevoi
}
```

### Step 5: Test Audio Output and Microphone Capture
The Google AIY Voice HAT uses a fixed-gain I2S codec without hardware mixer registers. Verify playback and capture directly:

```bash
speaker-test -c 2 -r 48000 -D default -t wav
arecord -D default -d 3 -f S16_LE -r 16000 -c 1 /tmp/test.wav && aplay -D default /tmp/test.wav
```

### Step 6: Autostart via systemd
Create `/etc/systemd/system/alsageist.service` (replacing `pi` with your username if different):

```ini
[Unit]
Description=AlsaGeist Conversational Display Service
After=network-online.target sound.target
Wants=network-online.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/alsageist
ExecStart=/home/pi/alsageist/venv/bin/python main.py
Restart=always
RestartSec=3
EnvironmentFile=/home/pi/alsageist/.env
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now alsageist.service
```

---

## 11. License

This project is licensed under the [MIT License](LICENSE).
