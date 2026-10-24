# Getting Started with AlsaGeist on Bare Metal Raspberry Pi OS

Step-by-step guide for installing, configuring, and operating AlsaGeist natively within a Python virtual environment on Raspberry Pi OS.

---

## 1. Overview

Native bare-metal deployment executes AlsaGeist directly on the host operating system using a dedicated Python virtual environment. This method provides direct access to host ALSA audio devices, the Linux character-device GPIO subsystem (`/dev/gpiochip*`), and systemd process supervisors.

| Directory | Path | Purpose |
| :--- | :--- | :--- |
| **Source Root** | `~/alsageist` | Application daemon, utility scripts, and runtime configuration |
| **Virtual Environment** | `~/alsageist/venv` | Isolated Python binaries (3.9+; Bookworm ships 3.11) and package dependencies |
| **Personalities** | `~/alsageist/data/personalities` | Character JSON profiles and dialogue regex patterns |
| **Neural Voices** | `~/alsageist/models/piper` | Piper ONNX voice models and JSON voice configs |
| **Audio Cache** | `~/alsageist/models/cache` | Pre-warmed NumPy audio arrays for zero-latency playback |
| **Telemetry** | `~/alsageist/logs` | JSONL turn logs and operational logs |

---

## 2. Prerequisites

* Raspberry Pi 3 Model B, 3B+, 4, or 5 running **Raspberry Pi OS Lite (64-bit)** (Debian 12 Bookworm), image `2026-10-06-raspios-bookworm-arm64-lite.img` or newer.
* Google AIY Voice HAT V1 or compatible USB speakerphone connected to the board.
* Network access and active SSH session to the Raspberry Pi.
* A power supply rated for the board (Raspberry Pi 3 Model B: 5 V, 2.5 A). An inadequate supply throttles the CPU and slows every step; see [Under-Voltage and CPU Throttling](usage_guide.md#56-under-voltage-and-cpu-throttling-raspberry-pi).
* A microSD card and a workstation to write it from (see section 3). For background on the hardware, see the [AlsaGeist Project & Hardware Overview](overview.md).

> [!IMPORTANT]
> Check your AIY Voice Kit version. V1 (Raspberry Pi 3, Voice HAT) is supported. V2 (Raspberry Pi Zero, Voice Bonnet) is untested; see [Profile C](overview.md#profile-c-google-aiy-voice-kit-v2-raspberry-pi-zero-untested) for limitations and suggestions.

---

## 3. Install the Operating System (on your workstation)

### Objective: Write Raspberry Pi OS Lite (64-bit) to a microSD card.

Do this on your Linux, macOS, or Windows computer:

1. Insert the microSD card into your workstation and open [Raspberry Pi Imager](https://www.raspberrypi.com/software/) (version 2.0 or newer). The wizard has these steps:
   * **Device:** select your Raspberry Pi model, then **Next**.
   * **OS:** select **Raspberry Pi OS (other)** → **Raspberry Pi OS Lite (64-bit)** (`2026-10-06-raspios-bookworm-arm64-lite.img` or newer), then **Next**.
   * **Storage:** select the microSD card (leave **Exclude system drives** enabled), then **Next**.
   * **Customisation:** complete each sub-step below, selecting **Next** after each. The **Wi-Fi** and **Remote Access** sub-steps are optional and let the Pi boot headless (onto your network with SSH enabled, no keyboard or monitor needed). Skip them only if you will attach a keyboard and monitor.
     * **Hostname:** e.g. `alsageist` (letters, numbers, and hyphens only).
     * **Localisation:** choose your capital city. This sets the time zone, keyboard layout, and Wi-Fi regulatory domain (Wi-Fi stays disabled without a regulatory domain).
     * **User:** choose a username (the examples in this guide use `pi`) and a password. There is no default account, so you must set one.
     * **Wi-Fi (optional, for headless boot):** choose a secure network, then enter the **SSID** and **password**. Turn on **Hidden SSID** only if your network does not broadcast its name. A Pi connected by Ethernet does not need Wi-Fi settings.
     * **Remote Access (optional, for headless boot):** turn on **Enable SSH**, then choose **Use password authentication**, or **Use public key authentication** and paste your public key.
     * **Raspberry Pi Connect:** optional; leave it off.
   * **Writing:** review the summary and write the card. Imager stores the customisation on the card's boot partition, and the Pi applies it on first boot.
2. **(Optional, disable CPU throttling)** After Imager finishes writing, but before first boot, re-insert the card into your workstation if it was ejected. Open the `bootfs` partition (the same file appears as `/boot/firmware/config.txt` once the Pi is running) and add this line at the end of `config.txt`:

   ```ini
   avoid_warnings=2
   ```

   This hides the low-voltage warning overlay and stops the firmware from lowering the CPU frequency when it detects low voltage, so the Pi keeps running at full speed under heavy load. It can keep speech recognition and synthesis response times more consistent. You can also add the line later by editing `/boot/firmware/config.txt` on the Pi and rebooting.

   > [!WARNING]
   > Skip this step unless you specifically want to disable throttling. Throttling exists to protect the Pi when its power supply cannot keep up, so turning it off can cause instability, crashes, or SD card corruption. Use a good power supply and adequate cooling, and remove the line if you see unexpected reboots.

   Eject the card safely when you are done.
3. Insert the card into the Raspberry Pi and power it on. The first boot takes a few minutes while the Pi expands the filesystem and, if you configured Wi-Fi, joins your network.
4. **(Optional, headless setup)** From your workstation (on the same network), connect over SSH. If you attached a keyboard and monitor instead, log in on the Pi directly.

```bash
ssh <username>@<hostname>.local
```

> [!TIP]
> If `<hostname>.local` does not resolve (common on Windows without Bonjour, or on networks that block mDNS), find the Pi's IP address in your router's client list and use `ssh <username>@<ip-address>`. A Pi connected by Ethernet cable needs no Wi-Fi settings and works the same way.

> [!NOTE]
> The Pi must be on a 2.4 GHz-capable Wi-Fi network (the original Raspberry Pi 3 Model B does not support 5 GHz; the 3B+, 4, and 5 do). If it never appears on your network, write the card again with corrected Wi-Fi settings, or connect it by Ethernet.

---

## 4. Audio Hardware & Driver Configuration

Hardware I2S audio devices (such as the Google AIY Voice HAT V1) require kernel device tree overlays and ALSA full-duplex PCM routing.

### 4.1 Automated Driver Configuration (Recommended)

#### Objective: Configure device tree overlays, ALSA routing, and hardware group permissions automatically.

> [!IMPORTANT]
> All steps from here on run **on the Raspberry Pi itself** (over SSH, or with a keyboard and monitor attached), not on your workstation.

1. Bring the freshly installed operating system up to date and install Git and the GitHub CLI. The update pulls in current kernel, firmware, and package fixes before the script configures the audio drivers:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y git gh
sudo reboot
```

Wait a minute for the Pi to restart, then reconnect with SSH.

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

### 4.2 Manual Driver Configuration (Alternative)

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

## 5. Verifying Audio Hardware in ALSA

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

## 6. System Dependencies & Build Toolchain

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

## 7. Repository & Virtual Environment Setup

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

## 8. Voice & Offline Speech Recognition Model Downloads

### Objective: Download local ONNX voice model weights and Vosk offline STT model.

Download the default Alan voice model (Piper ONNX) and the Vosk English STT model:

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

## 9. Environment Configuration (.env)

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

#### About the optional API keys

**Both API keys are optional. Leave either one blank**, and AlsaGeist runs fully offline. You can add them at any time later.

* **Groq API key** (`GROQ_API_KEY`): turns on cloud **STT** (speech-to-text, the step that converts what a visitor says into text). Groq runs Whisper Large v3, which is faster and more accurate than the offline Vosk model used when this is blank. A Groq key can also power the LLM fallback described below.
* **Gemini API key** (`GEMINI_API_KEY`): can power the **LLM fallback** (large language model), where Gemini improvises a reply when Grimjaw's built-in responses don't cover what a visitor said. When no key is set, he uses canned fallback lines instead.

Unlike the Docker setup script, nothing sets `LLM_FALLBACK` for you here. Set `LLM_FALLBACK=true` in `.env` yourself if you add a key and want the LLM fallback. After editing `.env` on a running service, restart it with `sudo systemctl restart alsageist.service`.

#### About the voice command password

Privileged spoken commands (shutdown, mode switches, fallback toggles) require a verbal password so visitors can't shut the device down or change its mode. The default is `alsageist`. This default is public, so **change it if you wish**. Set `VOICE_COMMAND_PASSWORD=` in `.env` to a word of your choice (or to `none` to disable the check), then restart the service with `sudo systemctl restart alsageist.service`.

---

## 10. Interactive Testing & Audio Tuning

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

2. Speak into the microphone. Grimjaw detects speech, processes the intent, synthesizes the voice, and illuminates the LED. Speak a full sentence near the microphone and pause; he answers when he hears you stop. Here are a few things to try:

   * **Get to know him:** *"What's your name?"*, *"Are you dead?"*, *"How old are you?"*, *"Where do you keep your brain?"*
   * **Candy:** *"Can I have some candy?"*, *"What is your favorite candy?"*
   * **Costumes:** *"I'm dressed up as Spider-Man."*, *"I'm dressed as Barbie."*, *"What is your favorite costume?"*
   * **Silly questions:** *"Do you get cold without any skin on?"*, *"Can you tickle yourself?"*, *"Do you own a pet?"*
   * **Pop culture:** *"Tell me a Seinfeld quote."*
   * **Check on the system:** *"What mode are you in?"*, *"What voice model are you using?"*

   See [Spoken Voice System Commands](usage_guide.md#43-spoken-voice-system-commands) for more spoken commands, including the ones that need the voice command password.

3. **Audio Calibration (if needed):**
   * **If the speaker is too loud or quiet:** Adjust `--volume` (e.g., `--volume 0.4` or `--volume 0.8`).
   * **If speech requires standing too close to the microphone:** Increase `--mic-gain` (e.g., `--mic-gain 2.0`).
   * **If room background noise triggers false activations:** Increase `--threshold` (e.g., `--threshold 0.0045`).

4. **Arcade Button Controls:**
   * **Short Press (< 5s):** Cycles playback volume in 10% steps and announces the new level.
   * **Medium Hold (5s):** Toggles Mute / Disabled mode (turns off microphone capture and extinguishes the LED).
   * **Long Hold (10s):** Initiates safe system shutdown. Speaks offline pre-cached "Powering off.", turns off the status LED, and cleanly powers down the Raspberry Pi.

---

## 11. Pre-Warming Dialogue Audio Cache

### Objective: Pre-render all character responses to disk for zero-latency (<200 ms) playback.

On the Raspberry Pi 3, neural TTS generation takes 4–6 seconds per uncached phrase. Pre-warming renders all dialogue responses in advance and stores them in `models/cache/`. It takes about 50 minutes on a Raspberry Pi 3. **This step is optional: you can skip it and run it later.** AlsaGeist works without it, but responses that aren't cached yet pause for 4–6 seconds while they are synthesized.

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

## 12. Autonomous Deployment via Systemd

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

## 13. Updating an Existing Installation

### Objective: Upgrade AlsaGeist to the latest release, update virtual environment dependencies, refresh the audio cache, and restart the background service.

1. Navigate to the installation directory:

```bash
cd ~/alsageist
```

2. Pull the latest release code:

```bash
git pull origin main
```

Expected output:
```text
Updating acdecae..35bab29
Fast-forward
```

3. Update dependencies in the virtual environment:

```bash
./venv/bin/pip install -c constraints.txt -e ".[rpi]"
```

Expected output:
```text
Requirement already satisfied: ...
Installing collected packages: ...
Successfully installed ...
```

4. Pre-warm dialogue audio cache:

```bash
./venv/bin/python3 scripts/prewarm_cache.py
```

Expected output:
```text
======================================================================
 Pre-Warming Dialogue Audio Cache
======================================================================
Pre-warming complete: 168 files cached.
```

5. Restart the background service:

```bash
sudo systemctl restart alsageist.service
```

6. Verify operational status:

```bash
./scripts/alsageist_ctl.py status
```

Expected output:
```text
================================================================
 AlsaGeist Runtime Status
================================================================
  Display State:     RUNNING
  Active Persona:    Grimjaw the Crypt Skeleton (grimjaw)
  Voice Model:       en_GB-alan-medium (Local ONNX)
  Audio Output Vol:  50% (0.5)
  Microphone State:  UNMUTED (Capturing)
  Network Mode:      ONLINE (auto) (Cloud STT / LLM: Disabled)
  Dialogue Engine:   INTENT
  Hardware LED:      IDLE
  Process PID:       1234 (Uptime: 0m 5s)
================================================================
```

---

## 14. Troubleshooting Runbook

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

### Problem: Startup, replies, or `say.py` take tens of seconds
The CPU is probably throttled by an inadequate power supply. Check the clock and the throttle flags:
```bash
vcgencmd measure_clock arm
vcgencmd get_throttled
```

Expected output when throttled:
```text
frequency(48)=600000000
throttled=0x50005
```

A clock of `1200000000` and a value of `0x0` mean the supply is not the cause. Otherwise see [Under-Voltage and CPU Throttling](usage_guide.md#56-under-voltage-and-cpu-throttling-raspberry-pi).

### Problem: `alsamixer: This sound device does not have any controls`
This is expected behavior for the Google AIY Voice HAT V1 because it uses a fixed-gain I2S codec. Test speaker output directly:
```bash
speaker-test -c 2 -r 48000 -D default -t wav
```
