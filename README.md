# AlsaGeist

Autonomous, voice-driven conversational animatronic display engine powered by offline Piper Neural Text-to-Speech (TTS), real-time Voice Activity Detection (VAD), and modular personality profiles.

Designed for cross-platform operation on macOS and on Raspberry Pi hardware with the Google AIY Voice Kit V1.

---

## 1. Overview

AlsaGeist runs an autonomous conversational character driven by pluggable personality profiles (defaulting to Grimjaw the Crypt Skeleton). The name **AlsaGeist** combines **ALSA** (Advanced Linux Sound Architecture, the kernel-level audio subsystem driving low-latency microphone capture and playback on Linux) and **Geist** (German for *ghost* or *spirit*), characterizing an autonomous vocal spirit inhabiting animatronic display hardware. It captures microphone audio in memory, detects completed speech boundaries via energy-based VAD, matches visitor utterances against an offline intent library, synthesizes character audio locally via Piper neural models in RAM, and drives an illuminated arcade status LED.

### Key Capabilities
* **Zero Disk I/O:** Audio capture, VAD framing, and speech synthesis operate directly in memory as raw PCM numpy arrays.
* **Offline Zero-Latency Dialogue:** Over 110 pre-programmed intents, pop-culture quotes, skeleton puns, and boundary rules respond in under 5 milliseconds.
* **Local Neural Speech:** Synthesizes gravelly character speech locally via Piper ONNX without cloud TTS APIs or subscription limits.
* **Cross-Platform:** Runs in terminal emulation mode on macOS or hardware GPIO/ALSA mode on Raspberry Pi.

---

## 2. System Requirements

### macOS
* macOS 12+ (Apple Silicon or Intel)
* Python 3.9+
* Built-in or USB microphone and speaker

### Raspberry Pi
* Raspberry Pi 3 Model B, 3B+, 4, or 5
* Google AIY Voice Kit V1 (2017 Voice HAT with stereo microphone daughterboard, 3-inch speaker, and arcade button)
* Raspberry Pi OS Lite (64-bit)
* Python 3.9+ (Raspberry Pi OS Bookworm includes 3.11; the Docker image bundles its own, so the host needs none for Docker)
* 5V 2.5A+ power supply (an inadequate supply throttles the CPU and slows every response; see [Under-Voltage and CPU Throttling](docs/usage_guide.md#56-under-voltage-and-cpu-throttling-raspberry-pi))

---

## 3. Quick Start (Docker on Raspberry Pi)

### Objective: Deploy AlsaGeist on a fresh Raspberry Pi OS install using the automated Docker onboarding tool.

See the complete [Getting Started with Docker Guide](docs/getting_started_rpi_docker.md) for full instructions and hardware details.

**Step 1: Install the OS (on your workstation).** Using [Raspberry Pi Imager](https://www.raspberrypi.com/software/) on Linux, macOS, or Windows, write `2026-10-06-raspios-bookworm-arm64-lite.img` (Raspberry Pi OS Lite, 64-bit) or newer to a microSD card. In the OS customization settings, set a hostname, a username (the examples use `pi`), and a password. To boot without a keyboard or monitor, also (optionally) set Wi-Fi and enable SSH. Insert the card into the Raspberry Pi and power it on. See [Operating System Preparation](docs/overview.md#5-operating-system-preparation).

**Step 2: Run the setup (on the Raspberry Pi itself).** Everything below runs on the Raspberry Pi, not on your workstation. Connect with `ssh <username>@<hostname>.local` (or use a keyboard and monitor). First update the fresh OS, make sure `git` is installed, and reboot, then reconnect:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y git
sudo reboot
```

Then, from your home directory, clone the repository and run the setup:

```bash
cd ~
git clone https://github.com/sancheza/alsageist.git
cd alsageist
./scripts/setup_docker.sh
```

Follow the prompts to configure audio hardware, download Grimjaw's neural voice model, and set optional API keys. When using the Google AIY Voice HAT, reboot to activate the soundcard overlay:

```bash
sudo reboot
```

---

## 4. Manual Installation (Bare Metal / Python Venv)

### Objective: Set up a local Python virtual environment and install dependencies.

See the complete [Getting Started on Bare Metal Guide](docs/getting_started_rpi_bare_metal.md) for full driver setup and background service configuration.

```bash
cd ~
git clone https://github.com/sancheza/alsageist.git
cd alsageist
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

## 5. Downloading Voice & Speech Recognition Models

### Objective: Download the Piper ONNX voice weights and Vosk offline STT model into the `models/` directory.

Run the model downloader script to fetch both models:

```bash
python3 scripts/download_model.py --all
```

Expected output:
```text
Downloading https://huggingface.co/rhasspy/piper-voices/resolve/c10ece1aade47bb51c153c893d14e5bf8e5b7117/en/en_GB/alan/medium/en_GB-alan-medium.onnx -> /home/pi/alsageist/models/piper/en_GB-alan-medium.onnx
Successfully saved /home/pi/alsageist/models/piper/en_GB-alan-medium.onnx (SHA-256 verified)
Downloading https://huggingface.co/rhasspy/piper-voices/resolve/c10ece1aade47bb51c153c893d14e5bf8e5b7117/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json -> /home/pi/alsageist/models/piper/en_GB-alan-medium.onnx.json
Successfully saved /home/pi/alsageist/models/piper/en_GB-alan-medium.onnx.json (SHA-256 verified)
Downloading https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -> /home/pi/alsageist/models/vosk-model-small-en-us-0.15
Extracting verified archive into /home/pi/alsageist/models...
Successfully installed Vosk model to /home/pi/alsageist/models/vosk-model-small-en-us-0.15
```

Each file is checked against a pinned SHA-256 digest before it is installed; a mismatch stops the download with a non-zero exit status.

Alternatively, download individual components:
* `python3 scripts/download_model.py --voice alan`: Downloads Piper neural TTS voice only.
* `python3 scripts/download_model.py --vosk`: Downloads Vosk offline STT model only.

---

## 6. Configuration & Environment Variables

AlsaGeist uses offline intent matching and local TTS by default. Configure your environment using the provided template:

```bash
cp env.example .env
nano .env
```

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Groq API key for high-speed Whisper Large v3 STT | None |
| `GEMINI_API_KEY` | Google Gemini API key for Gemini Flash STT & LLM fallback | None |
| `PERSONALITY` | Active character personality (`data/personalities/<name>.json`) | `grimjaw` |
| `LLM_FALLBACK` | Fetch custom LLM response for indeterminate prompts instead of canned response | `false` |
| `ALSAGEIST_USER` | Target Linux user account for systemd service installation | `$USER` |
| `ALSAGEIST_DIR` | Installation directory path for systemd service installation | Repository path |
| `VOLUME` | Audio playback volume multiplier (`0.0` to `1.0`) | `0.5` |
| `MIC_GAIN` | Microphone digital input gain multiplier | `1.5` |
| `VAD_THRESHOLD` | RMS audio energy threshold for speech detection | `0.0025` |

```ini
# Minimal recommended .env
GROQ_API_KEY=gsk_your_groq_api_key_here
PERSONALITY=grimjaw
LLM_FALLBACK=false
VOLUME=0.5
MIC_GAIN=1.5
VAD_THRESHOLD=0.0035
```

---

## 7. Usage & Running the Display

### Objective: Start the interactive display daemon.

Run the main application:

```bash
./main.py
```

Expected output:
```text
======================================================================
 ALSA-GEIST CRYPT SKELETON IS ACTIVE & LISTENING
 Engine: INTENT | Voice: Piper Alan (Local ONNX)
 Interaction Logging: Active -> logs/interactions.jsonl
 Speak into your microphone to start a conversation.
 Press Ctrl+C to stop.
======================================================================

🟢 [LED: IDLE (Breathing ~ ~ ~)]
```

Speak into your microphone. Grimjaw will detect your voice, transcribe the input, match the intent, synthesize speech, and play the response while pulsing the LED.

For real-time CLI remote control (`alsageist_ctl.py`), noise calibration, online/offline operational modes, and custom personality creation, see the [User & Operations Guide](docs/usage_guide.md).

---

## 8. Command-Line Options

Inspect all runtime flags:

```bash
./main.py --help
```

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-v, --version` | Display application version and exit | Current version |
| `--engine` | Dialogue engine: `intent` (offline) or `llm` (cloud) | `intent` |
| `--llm-fallback` | Fetch custom LLM response for indeterminate prompts | `$LLM_FALLBACK` or `false` |
| `--no-llm-fallback` | Disable LLM fallback for indeterminate prompts | True if unset |
| `--model-path` | Path to the Piper ONNX model file | `models/piper/en_GB-alan-medium.onnx` |
| `--input-device` | Input audio hardware device name or index | System default |
| `--output-device` | Output audio hardware device name or index | System default |
| `--volume` | Audio playback volume multiplier (`0.0` to `1.0`) | `0.5` (or `$VOLUME`) |
| `--mic-gain` | Microphone digital input gain multiplier | `1.5` (or `$MIC_GAIN`) |
| `--threshold` | RMS audio energy threshold for speech detection | `0.0025` (or `$VAD_THRESHOLD`) |
| `--min-speech-sec` | Minimum sustained speech duration in seconds | `0.15` |
| `--silence-timeout` | Trailing silence in seconds before speech boundary | `0.9` |
| `--tempo` | Piper phoneme length scale (higher = slower speech) | `1.15` |
| `--noise` | Piper generator noise level for gravel texture | `0.85` |
| `--no-greeting` | Disable spoken startup greeting announcement | False |

---

## 9. Conversational Intent Architecture

The offline dialogue engine in `services/intent_service.py` provides deterministic responses across over 150 categories:
* **Candy Protection & Bowl Rules:** Limits visitors to one piece, detects theft inquiries, and warns against greedy hands.
* **Costume & Character Recognition:** 46 distinct 1-to-1 character rules (KPop Demon Hunters: Rumi, Mira, Zoey, Jinu; Bluey; Mario, Luigi, Peach, Yoshi; Baby Yoda; Eevee; Toy Story; Frozen; Moana; Mickey, Minnie, Cinderella; DC/Marvel heroes; Sonic; Minecraft Steve; SpongeBob; Barbie; TMNT; Harry Potter; PAW Patrol).
* **Costume Inquiry State Machine:** Once a conversation has started (turn 2+), Grimjaw has a 1-in-3 chance to ask the visitor who they are dressed as.
* **Classic Pop Culture & Cinema Quotes:** *I'm Gonna Git You Sucka*, *Blade Runner* (Tears in Rain), *Say Anything* (Hey Soul Classics), *Ali G*, *He-Man & Skeletor*.
* **Poetry & Music:** Edgar Allan Poe's *The Raven*, Robert Browning's *Childe Roland*, Michael Jackson's *Thriller*, *Monster Mash*, *The Time Warp*.
* **Anatomy & Skeleton Puns:** Banter regarding bones, teeth, absent organs, zero-calorie diets, and cold weather.
* **Indeterminate Prompt Handling:** When `LLM_FALLBACK=true`, unrecognized phrases trigger custom LLM responses via Groq/Gemini, with automatic fallback to generic canned lines if offline or on timeout.
* **Visitor Prompts Reference:** Run `./scripts/summarize_personality.py -p grimjaw` to dynamically inspect or export the catalog of supported spoken questions and answers for any personality profile.

---

## 10. Hardware Arcade Button & Status LED Signaling

### Arcade Button Functions

The lighted arcade pushbutton on GPIO 23 provides manual controls:

| Action | Duration | Result |
| :--- | :--- | :--- |
| **Short Press** | < 5.0 seconds | Cycles audio volume in 10% steps (`0.0` to `1.0`), stops ongoing playback, resets conversational context, and announces the new volume level. Ignored when disabled. |
| **Medium Hold** | 5.0 seconds | **Mute / Disable Toggle**: When active, holding for 5 seconds disables Grimjaw (turns off microphone capture and speaker output, stops playback, resets dialogue context, and turns the LED completely off). Holding for 5 seconds again re-enables Grimjaw, restores microphone capture and speaker output, sets the LED to breathing `IDLE`, and resumes normal operation. |
| **Long Hold** | 10.0 seconds | **Safe System Poweroff**: Speaks offline pre-cached "Powering off.", turns off the status LED, and cleanly powers down the Raspberry Pi operating system (supported natively and inside Docker). |

### Status LED Signaling

The integrated LED on GPIO 25 provides real-time operational feedback:

| State | Pattern | Meaning |
| :--- | :--- | :--- |
| `IDLE` | 0.5 Hz smooth breathing pulse (5%–85% duty cycle) | Resting ambient state, listening for visitors on the porch. |
| `LISTENING` | Solid 100% brightness | Speech detected; recording incoming audio. |
| `FLASH` | 3 rapid pulses (~7 Hz) | Acknowledgment: confirmed speech was captured. |
| `THINKING` | Heartbeat double-pulse (~1.1 Hz "lub-dub... pause") | Processing STT transcription, intent classification, LLM fallback, or TTS synthesis. |
| `SPEAKING` | 3–4 Hz organic audio-reactive flicker | Playing character speech through the speaker. |
| `ERROR` | 5 Hz rapid square-wave alert blink | Subsystem exception or network STT failure. |
| `OFF` | Dark (0% duty cycle) | Muted / disabled via 5-second button hold. |

---

## 11. Testing

### Objective: Run the automated test suite.

Install the test dependencies, then execute pytest across all unit tests:

```bash
pip install -e ".[dev]"
pytest tests/
```

Expected output:
```text
============================= 350 passed in 18.74s =============================
```

---

## 12. Raspberry Pi & Google AIY Voice Kit Deployment

For hardware selection, acoustic enclosure design, and OS flashing, see the [Project & Hardware Overview](docs/overview.md). For step-by-step native setup and service deployment, see the [Getting Started on Bare Metal Guide](docs/getting_started_rpi_bare_metal.md).

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

## 13. License

Copyright (c) 2026 sancheza. This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0). You may use, modify, and share it freely, but if you distribute it or offer it as a network service, you must publish your complete source, including your changes, under the same license.

### Commercial Licensing

To use AlsaGeist in a closed-source or commercial product without the AGPL's requirements, request a separate commercial license from the author through [GitHub](https://github.com/sancheza).

### Trademark

"AlsaGeist" is the name of this project. The license covers the code, not the name. Please do not use the name for commercial products or modified versions without permission.
