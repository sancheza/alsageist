# Getting Started with AlsaGeist on Bare Metal Raspberry Pi OS

Step-by-step guide for installing, configuring, and operating AlsaGeist natively within a Python virtual environment on Raspberry Pi OS.

---

## 1. Overview

Native bare-metal deployment executes AlsaGeist directly on the host operating system using a dedicated Python virtual environment. This method provides direct access to host ALSA audio devices, the Linux character-device GPIO subsystem (`/dev/gpiochip*`), and systemd process supervisors.

| Directory | Path | Purpose |
| :--- | :--- | :--- |
| **Source Root** | `~/alsageist` | Application daemon, utility scripts, and runtime configuration |
| **Virtual Environment** | `~/alsageist/venv` | Isolated Python 3.11+ binaries and package dependencies |
| **Personalities** | `~/alsageist/data/personalities` | Character JSON profiles and dialogue regex patterns |
| **Neural Voices** | `~/alsageist/models/piper` | Piper ONNX voice models and JSON voice configs |
| **Audio Cache** | `~/alsageist/models/cache` | Pre-warmed NumPy audio arrays for zero-latency playback |
| **Telemetry** | `~/alsageist/logs` | JSONL turn logs and operational logs |

---

## 2. Prerequisites

* Raspberry Pi 3 Model B, 3B+, 4, or 5 running **Raspberry Pi OS Lite (64-bit)** (Debian 12 Bookworm).
* Google AIY Voice HAT V1 or compatible USB speakerphone connected to the board.
* Network access and active SSH session to the Raspberry Pi.
* Operating system flashed and verified (see [AlsaGeist Project & Hardware Overview](overview.md)).

---

## 3. Audio Hardware & Driver Configuration

Hardware I2S audio devices (such as the Google AIY Voice HAT V1) require kernel device tree overlays and ALSA full-duplex PCM routing.

### 3.1 Automated Driver Configuration (Recommended)

#### Objective: Configure device tree overlays, ALSA routing, and hardware group permissions automatically.

1. Install Git and the GitHub CLI:

```bash
sudo apt update && sudo apt install -y git gh
```

2. Clone the repository:

```bash
cd ~
git clone https://github.com/sancheza/alsageist.git
cd alsageist
```

> [!NOTE]
> If accessing a private repository, authenticate first via GitHub CLI:
> ```bash
> gh auth login
> ```

3. Run the automated configuration script:

```bash
sudo ./scripts/setup_rpi_os.sh --profile aiy
```

*For USB speakerphones or generic USB audio adapters, pass `--profile usb` instead.*

Expected output:
```text
======================================================================
 AlsaGeist Raspberry Pi OS Configuration Tool
 Profile: AIY Voice HAT V1
======================================================================

[1/5] Applying device tree overlays in /boot/firmware/config.txt...
  Added: dtparam=audio=off
  Added: dtparam=i2s=on
  Added: dtoverlay=googlevoicehat-soundcard

[2/5] Configuring ALSA routing in /etc/asound.conf...
  Written full-duplex ALSA dmix/dsnoop configuration.

[3/5] Configuring user hardware group permissions...
  Added user to 'audio' group.
  Added user to 'gpio' group.

======================================================================
 Configuration complete. A reboot is required to load kernel drivers.
======================================================================
```

4. Reboot the Raspberry Pi to activate the kernel audio drivers:

```bash
sudo reboot
```

---

### 3.2 Manual Driver Configuration (Alternative)

If configuring drivers manually without the automated script:

1. Edit `/boot/firmware/config.txt` (or `/boot/config.txt` on legacy releases):

```bash
sudo nano /boot/firmware/config.txt
```

2. Disable onboard Broadcom audio and enable the Google Voice HAT overlay:

```ini
# Disable onboard audio
dtparam=audio=off

# Enable Google AIY Voice HAT V1 I2S sound card
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

3. Save the file and reboot:

```bash
sudo reboot
```

---

## 4. Verifying Audio Hardware in ALSA

### Objective: Confirm playback and capture devices are registered and operational.

1. Check playback hardware endpoints:

```bash
aplay -l
```

Expected output for Google AIY Voice HAT:
```text
**** List of PLAYBACK Hardware Devices ****
card 0: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: Google voiceHAT SoundCard HiFi voicehat-hifi-0 []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

2. Check capture hardware endpoints:

```bash
arecord -l
```

Expected output:
```text
**** List of CAPTURE Hardware Devices ****
card 0: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: Google voiceHAT SoundCard HiFi voicehat-hifi-0 []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

3. Test speaker playback with a test WAV file:

```bash
speaker-test -c 2 -r 48000 -D default -t wav
```
*Press `Ctrl+C` to stop playback.*

4. Test microphone capture (records 3 seconds of audio and immediately plays it back):

```bash
arecord -D default -d 3 -f S16_LE -r 16000 -c 1 /tmp/test.wav && aplay -D default /tmp/test.wav
```

> [!NOTE]
> The Google AIY Voice HAT V1 uses a fixed-gain I2S codec without hardware mixer registers. Running `alsamixer` will display `This sound device does not have any controls`. This is normal and expected; volume is controlled in software.

---

## 5. System Dependencies & Build Toolchain

### Objective: Install required operating system packages, build tools, and audio headers.

```bash
sudo apt update && sudo apt install -y \
  git \
  python3-pip \
  python3-venv \
  portaudio19-dev \
  libasound2-dev \
  alsa-utils \
  libatlas-base-dev \
  unzip
```

Expected output:
```text
Processing triggers for libc-bin...
0 upgraded, 8 newly installed, 0 to remove.
```

---

## 6. Repository & Virtual Environment Setup

### Objective: Configure the Python virtual environment and install dependencies.

1. Navigate to the cloned repository:

```bash
cd ~/alsageist
```

2. Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Upgrade packaging tools and install AlsaGeist with the Raspberry Pi GPIO libraries in editable mode, pinned to the tested dependency versions in `constraints.txt`:

```bash
pip install --upgrade pip setuptools wheel
pip install -c constraints.txt -e ".[rpi]"
```

4. Verify Python dependencies:

```bash
python3 -c "import sounddevice, numpy, gpiozero, vosk; print('Dependencies verified successfully.')"
```

Expected output:
```text
Dependencies verified successfully.
```

---

## 7. Voice & Offline Speech Recognition Model Downloads

### Objective: Download local ONNX voice model weights and Vosk offline STT model.

Download the default Alan voice model (Piper ONNX) and the Vosk English STT model:

```bash
python3 scripts/download_model.py --all
```

Expected output:
```text
Downloading https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx -> /home/pi/alsageist/models/piper/en_GB-alan-medium.onnx
Successfully saved /home/pi/alsageist/models/piper/en_GB-alan-medium.onnx (SHA-256 verified)
Downloading https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json -> /home/pi/alsageist/models/piper/en_GB-alan-medium.onnx.json
Successfully saved /home/pi/alsageist/models/piper/en_GB-alan-medium.onnx.json (SHA-256 verified)
Downloading https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -> /home/pi/alsageist/models/vosk-model-small-en-us-0.15
Extracting verified archive into /home/pi/alsageist/models...
Successfully installed Vosk model to /home/pi/alsageist/models/vosk-model-small-en-us-0.15
```

Verify downloaded model directories in `models/`:

```bash
ls -lh models/piper/
ls -d models/vosk-model-small-en-us-0.15/
```

Expected output:
```text
-rw-r--r-- 1 alsageist alsageist  61M en_GB-alan-medium.onnx
-rw-r--r-- 1 alsageist alsageist 4.8K en_GB-alan-medium.onnx.json
models/vosk-model-small-en-us-0.15/
```

---

## 8. Environment Configuration (.env)

### Objective: Configure runtime settings and API credentials.

1. Copy the example configuration template:

```bash
cp env.example .env
chmod 600 .env
nano .env
```

2. Configure environment settings:

| Parameter | Environment Variable | CLI Flag | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Persona** | `PERSONALITY` | `--personality` | `grimjaw` | Character profile from `data/personalities/<name>.json`. |
| **Output Volume** | `VOLUME` | `--volume` | `0.5` | Playback audio volume multiplier between `0.0` and `1.0`. |
| **Microphone Gain** | `MIC_GAIN` | `--mic-gain` | `1.5` | Digital input multiplier applied to raw captured audio frames. |
| **VAD Threshold** | `VAD_THRESHOLD` | `--threshold` | `0.0035` | Minimum RMS audio energy required to trigger speech capture. |
| **Cloud STT** | `GROQ_API_KEY` | None | None | Groq Whisper Large v3 authentication token. |
| **Cloud STT / LLM** | `GEMINI_API_KEY` | None | None | Google Gemini authentication token. |
| **Dialogue Mode** | `LLM_FALLBACK` | `--llm-fallback` | `false` | Enable dynamic LLM fallback for indeterminate prompts. |

Sample minimal `.env` configuration:

```ini
# Character Configuration
PERSONALITY=grimjaw
LLM_FALLBACK=false

# Cloud Speech-to-Text (Optional, leave blank for offline mode)
GROQ_API_KEY=
GEMINI_API_KEY=

# Audio Hardware Calibration
VOLUME=0.5
MIC_GAIN=1.5
VAD_THRESHOLD=0.0035
```

---

## 9. Interactive Testing & Audio Tuning

### Objective: Launch AlsaGeist in foreground mode to verify capture, synthesis, and LED behavior.

1. Start the main application:

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

2. Speak into the microphone. Grimjaw detects speech, processes the intent, synthesizes the voice, and illuminates the LED.

3. **Audio Calibration (if needed):**
   * **If the speaker is too loud or quiet:** Adjust `--volume` (e.g., `--volume 0.4` or `--volume 0.8`).
   * **If speech requires standing too close to the microphone:** Increase `--mic-gain` (e.g., `--mic-gain 2.0`).
   * **If room background noise triggers false activations:** Increase `--threshold` (e.g., `--threshold 0.0045`).

4. **Arcade Button Controls:**
   * **Short Press (< 5s):** Cycles playback volume in 10% steps and announces the new level.
   * **Medium Hold (5s):** Toggles Mute / Disabled mode (turns off microphone capture and extinguishes the LED).
   * **Long Hold (10s):** Initiates safe system shutdown. Speaks offline pre-cached "Powering off.", turns off the status LED, and cleanly powers down the Raspberry Pi.

---

## 10. Pre-Warming Dialogue Audio Cache

### Objective: Pre-render all character responses to disk for zero-latency (<200 ms) playback.

On the Raspberry Pi 3, neural TTS generation takes 4–6 seconds per uncached phrase. Pre-warming renders all dialogue responses in advance and stores them in `models/cache/`:

```bash
./scripts/prewarm_cache.py --all
```

Check the number of compiled audio cache files:

```bash
ls -1 models/cache/*.npy | wc -l
```

Expected output:
```text
401
```

---

## 11. Autonomous Deployment via Systemd

### Objective: Configure AlsaGeist as a background service that launches automatically on power-on.

1. Install and enable the systemd service using the automated installer:

```bash
./scripts/setup_systemd_service.sh --enable
```

> [!NOTE]
> The installer displays a warning confirming this is a bare-metal installation (not Docker) and prompts for confirmation (`y/N`). To bypass the prompt in automated scripts, pass `-y` / `--yes`.

Expected output:
```text
======================================================================
 AlsaGeist Systemd Service Configuration
======================================================================
  Target User:        pi
  Working Directory:  /home/pi/alsageist
  Template Source:    /home/pi/alsageist/alsageist.service
  Target Output:      /etc/systemd/system/alsageist.service
======================================================================

======================================================================
 WARNING: BARE-METAL SERVICE INSTALLATION
======================================================================
 This script generates and installs a systemd unit for running AlsaGeist
 directly on the host OS via a native Python virtual environment.

 This script is NOT required for Docker installations.
 Docker installations manage container lifecycle and autostart via
 Docker Compose (or scripts/setup_docker.sh).
======================================================================

Proceed with installing bare-metal systemd service? [y/N]: y
  Successfully installed service unit to /etc/systemd/system/alsageist.service.
  Reloading systemd daemon and enabling service...
  Service enabled and started.
Done.
```

2. Check service status:

```bash
sudo systemctl status alsageist.service
```

Expected output:
```text
● alsageist.service - AlsaGeist Autonomous Conversational Display Engine
     Loaded: loaded (/etc/systemd/system/alsageist.service; enabled; preset: enabled)
     Active: active (running)
```

3. Follow live operational logs:

```bash
journalctl -u alsageist.service -f
```

---

## 12. Troubleshooting Runbook

### Problem: `PortAudio / sounddevice: No default input/output device found`
Ensure the user account belongs to the `audio` group and specify devices explicitly:
```bash
sudo usermod -aG audio $USER
./main.py --input-device 0 --output-device 0
```

### Problem: Voice playback sounds high-pitched, sped up, or distorted
The hardware DAC requires 44.1 kHz or 48 kHz stereo streams, while Piper synthesizes at 22.05 kHz mono. Ensure full-duplex ALSA routing in `/etc/asound.conf` was generated via `scripts/setup_rpi_os.sh`.

### Problem: Microphone audio capture is silent or VAD does not trigger
On USB audio devices, run `alsamixer`, press `F4`, unmute capture channels (`Spacebar`), and increase microphone gain to 80%. Save settings:
```bash
sudo alsactl store
```

### Problem: `Permission denied accessing GPIO`
Add the user account to the `gpio` group and reboot:
```bash
sudo usermod -aG gpio $USER
sudo reboot
```

### Problem: `Speech recognition timed out / Network unreachable`
Verify internet connectivity to cloud APIs:
```bash
curl -I https://api.groq.com
curl -I https://generativelanguage.googleapis.com
```

### Problem: `alsamixer: This sound device does not have any controls`
This is expected behavior for the Google AIY Voice HAT V1 because it uses a fixed-gain I2S codec. Test speaker output directly:
```bash
speaker-test -c 2 -r 48000 -D default -t wav
```
