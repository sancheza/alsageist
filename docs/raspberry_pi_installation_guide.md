# Raspberry Pi Installation & Deployment Guide

This guide covers hardware selection, driver configuration, system setup, and deployment for running AlsaGeist on Raspberry Pi.

---

## 1. System Requirements & Hardware Selection

### Minimum vs. Recommended Specifications

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Single Board Computer** | Raspberry Pi 3 Model B (1 GB RAM) | Raspberry Pi 4 Model B (2 GB+ RAM) or Raspberry Pi 5 |
| **Architecture** | 64-bit ARM (`aarch64`) | 64-bit ARM (`aarch64`) |
| **Storage** | 16 GB MicroSD Card (Class 10 / A1) | 32 GB MicroSD Card (Class 10 / A2) |
| **Power Supply** | 5V 2.5A Micro-USB (Pi 3) | Official 5V 3.0A+ USB-C (Pi 4) / 5V 5.0A (Pi 5) |
| **Operating System** | Raspberry Pi OS Lite (64-bit, Bookworm) | Raspberry Pi OS Lite (64-bit, Bookworm) |
| **Network** | Wi-Fi 802.11n or 10/100 Ethernet | Gigabit Ethernet or 5GHz 802.11ac Wi-Fi |
| **Internet Access** | Required during setup and for cloud STT API calls | Required during setup and for cloud STT API calls |

> [!NOTE]
> AlsaGeist executes intent recognition, Voice Activity Detection (VAD), and neural Piper TTS speech synthesis completely offline. An internet connection is strictly required for two phases: initial package/model installation, and streaming Speech-to-Text inference via Groq Whisper or Google Gemini Flash APIs during live interaction.

---

### Hardware Configuration Profiles

#### Profile A: Primary Target — Raspberry Pi 3 + Google AIY Voice Kit V1 (2017)
* **Single Board Computer:** Raspberry Pi 3 Model B (or 3 Model B+)
* **Audio HAT:** Google AIY Voice HAT V1 (attached to the 40-pin GPIO header)
* **Microphone:** AIY Voice Kit dual Knowles MEMS microphone daughterboard (connected via 5-pin ribbon cable)
* **Speaker:** 3-inch 4Ω 3W speaker wired to the Voice HAT screw terminal blocks
* **Status Indicator:** Arcade push button with integrated LED (wired to the Voice HAT button harness)
* **Pin Allocation:**
  * LED Indicator: GPIO 25
  * Arcade Push Button: GPIO 23

#### Profile B: Alternative Modern Hardware — Raspberry Pi 4 or 5 + USB Audio
* **Single Board Computer:** Raspberry Pi 4 Model B or Raspberry Pi 5
* **Audio Hardware:** USB Audio Class (UAC) speakerphone (e.g., Anker PowerConf, Jabra Speak 410) or USB sound card with 3.5mm microphone and powered speaker
* **Status Indicator:** Standard 5mm LED with 330Ω current-limiting resistor connected between GPIO 25 (Pin 22) and GND (Pin 20)
* **Arcade Button (Optional):** Normally-open momentary switch connected between GPIO 23 (Pin 16) and GND (Pin 14) with internal pull-up enabled

---

## 2. Operating System Installation

### Objective: Flash Raspberry Pi OS Lite (64-bit) to the MicroSD card.

1. Insert the MicroSD card into your workstation.
2. Launch **Raspberry Pi Imager**.
3. Choose OS: **Raspberry Pi OS (other)** -> **Raspberry Pi OS Lite (64-bit)**.
4. Choose Storage: Select your MicroSD card.
5. In OS Customization settings:
   * Set hostname: (e.g., `alsageist`, `raspberrypi`, or any preferred network name)
   * Set Linux username and password (e.g., `alsageist`, `pi`, or your preferred username)
   * Configure wireless LAN credentials (SSID and password)
   * Enable SSH with password authentication or authorized public key
6. Click **Write** and verify the write operation.
7. Insert the MicroSD card into the Raspberry Pi, connect network/power, and power on.

> [!NOTE]
> **Hostname & User Account Flexibility:** AlsaGeist has zero hardcoded hostname or username dependencies. You can choose any network hostname (e.g., `alsageist`, `crypt-skeleton`, or `raspberrypi`) and any Linux user account (e.g., `pi`, `alsageist`, or your personal account). The only requirements are that the user account belongs to the `audio` and `gpio` groups for hardware access, and that file paths in the systemd service match your user's home directory.

Verify SSH connectivity from your workstation (replacing `<username>` and `<hostname>` with your configured values):

```bash
ssh <username>@<hostname>.local
```

Example output:
```text
Linux alsageist 6.6.x+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.6.x aarch64
alsageist@alsageist:~$
```

---

## 3. Audio Hardware & Driver Configuration

### 3.1 Automated OS Configuration Script

AlsaGeist includes an automated setup script ([`scripts/configure_rpi_os.sh`](../scripts/configure_rpi_os.sh)) that configures device tree overlays in `config.txt`, writes full-duplex ALSA routing in `/etc/asound.conf`, and adds the user to the `audio` and `gpio` groups.

#### Objective: Clone the repository, configure OS audio drivers, and set user permissions automatically.

1. Install Git and GitHub CLI:

```bash
sudo apt update && sudo apt install -y git gh
```

> [!NOTE]
> **GitHub Authentication (Private Repository):**
> If the repository is private, authenticate using GitHub CLI (`gh auth login`) before cloning. This configures Git credentials automatically without requiring manual Personal Access Tokens:
>
> ```bash
> gh auth login
> ```
>
> Follow the interactive prompts:
> ```text
> ? What account do you want to log into? GitHub.com
> ? What is your preferred protocol for Git operations? HTTPS
> ? Authenticate Git with your GitHub credentials? Yes
> ? How would you like to authenticate GitHub CLI? Login with a web browser
> ```
>
> GitHub CLI outputs a one-time device code. Open `https://github.com/login/device` in any browser, enter the code, and confirm.
>
> *(Alternative methods: generate an SSH key on the Pi via `ssh-keygen -t ed25519` and add `~/.ssh/id_ed25519.pub` to GitHub Settings, or set the repository visibility to Public).*

2. Clone the repository into your home directory:

```bash
cd ~
git clone https://github.com/sancheza/alsageist.git
cd alsageist
```

3. Run the automated configuration script:

For **Google AIY Voice HAT V1 (Profile A)**:

```bash
sudo ./scripts/configure_rpi_os.sh --profile aiy
```

For **USB Audio Class Devices (Profile B)**:

```bash
sudo ./scripts/configure_rpi_os.sh --profile usb
```

By default, the script configures permissions for whichever Linux user invokes `sudo`. To configure for a specific user explicitly:

```bash
sudo ./scripts/configure_rpi_os.sh --profile aiy --user <your_username>
```

To preview all actions without writing files or making system modifications:

```bash
./scripts/configure_rpi_os.sh --dry-run
```

Expected output:
```text
======================================================================
 AlsaGeist Raspberry Pi OS Configuration Tool v1.0.0
 Profile: AIY | Target user: alsageist
 Target config: /boot/firmware/config.txt
======================================================================

[1/3] User Group Configuration for user 'alsageist'...
  User 'alsageist' is already a member of 'audio'.
  User 'alsageist' is already a member of 'gpio'.

[2/3] Kernel & Device Tree Configuration (/boot/firmware/config.txt)...
  Appending Voice HAT overlay to /boot/firmware/config.txt...
  Voice HAT configuration appended.

[3/3] ALSA Configuration (/etc/asound.conf)...
  Writing ALSA full-duplex configuration to /etc/asound.conf...
  ALSA configuration written successfully.

======================================================================
 OS configuration complete!
 A system reboot is required to load the Google Voice HAT device tree overlay:
   sudo reboot
======================================================================
```

4. Reboot the Raspberry Pi after running the script:

```bash
sudo reboot
```

> [!TIP]
> **What to Skip If Using the Automated Script:**
> The script automatically installs device tree overlays, configures `/etc/asound.conf`, and adds your user account to the `audio` and `gpio` hardware groups.
>
> After rebooting, **skip Section 3.2 (Manual Google AIY Voice HAT Driver Configuration) and Section 3.3 (USB Audio Hardware Configuration)** and proceed directly to **[Section 3.4: Verifying Audio Devices in ALSA](#34-verifying-audio-devices-in-alsa)**.

---

### 3.2 Manual Google AIY Voice HAT V1 Driver Configuration

If performing manual configuration instead of running the script:

#### Objective: Enable the AIY Voice HAT device tree overlay and disable onboard audio manually.

Edit `/boot/firmware/config.txt` (or `/boot/config.txt` on legacy OS releases):

```bash
sudo nano /boot/firmware/config.txt
```

Locate the audio section, disable the default Broadcom audio output, and add the Google Voice HAT overlay:

```ini
# Disable default onboard Broadcom audio
dtparam=audio=off

# Enable Google AIY Voice HAT V1 I2S sound card
dtoverlay=googlevoicehat-soundcard
dtparam=i2s=on
```

Save the file and reboot the Raspberry Pi:

```bash
sudo reboot
```

---

### 3.3 USB Audio Hardware Configuration

When using USB speakerphones or USB DAC adapters (Profile B), no device tree overlay is needed. Connect the USB audio device to any available USB port.

Verify device detection via kernel messages:

```bash
dmesg | grep -i "usb audio"
```

Expected output:
```text
usb 1-1.2: 1:1: cannot get min/max values for control 2 (id 2)
usbcore: registered new interface driver snd-usb-audio
```

---

### 3.4 Verifying Audio Devices in ALSA

#### Objective: Verify that the recording and playback endpoints are registered by ALSA.

Check playback devices:

```bash
aplay -l
```

Expected output for Google AIY Voice HAT:
```text
**** List of PLAYBACK Hardware Devices ****
card 0: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: Google voiceHAT SoundCard HiFi voicehat-hifi-0 [Google voiceHAT SoundCard HiFi voicehat-hifi-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

Check capture devices:

```bash
arecord -l
```

Expected output for Google AIY Voice HAT:
```text
**** List of CAPTURE Hardware Devices ****
card 0: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: Google voiceHAT SoundCard HiFi voicehat-hifi-0 [Google voiceHAT SoundCard HiFi voicehat-hifi-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

---

### 3.5 Audio Mixer Controls & Hardware Verification

#### Google AIY Voice HAT V1 (Profile A)
The Google Voice HAT V1 uses an I2S DAC and Knowles digital MEMS microphones with fixed hardware gain. It does not expose hardware mixer registers to ALSA.

When selecting `snd_rpi_googlevoicehat_soundcar` in `alsamixer`, the following message is **normal and expected**:
```text
This sound device does not have any controls
```
AlsaGeist streams digital audio directly to the I2S DAC at nominal volume and does not require mixer adjustments.

##### Objective: Test Voice HAT speaker output and microphone capture.

Test speaker output with spoken voice (press `Ctrl+C` to stop):

```bash
speaker-test -c 2 -r 48000 -D default -t wav
```

Alternatively, test with a 440 Hz audible tone:

```bash
speaker-test -c 2 -r 44100 -D default -t sine -f 440
```

Test microphone capture (records 3 seconds of audio, then plays it back):

```bash
arecord -D default -d 3 -f S16_LE -r 16000 -c 1 /tmp/test.wav && aplay -D default /tmp/test.wav
```

#### USB Audio Adapters & Speakerphones (Profile B)
USB audio cards that expose hardware mixer registers can be adjusted via `alsamixer`:

```bash
alsamixer
```

1. Press `F6` and select your USB audio device.
2. Use arrow keys to set Master Playback volume to `85%`.
3. Press `F4` to view Capture controls and set Microphone gain to `80%`.
4. Press `Esc` to exit, then save settings:

```bash
sudo alsactl store
```

---

## 4. System Toolchain & Dependencies Installation

### Objective: Install required operating system packages, build tools, and audio headers.

Update apt repositories and install dependencies:

```bash
sudo apt update && sudo apt install -y \
  git \
  python3-pip \
  python3-venv \
  portaudio19-dev \
  libasound2-dev \
  alsa-utils \
  libatlas-base-dev
```

Expected output:
```text
Processing triggers for libc-bin...
0 upgraded, 7 newly installed, 0 to remove.
```

---

## 5. AlsaGeist Repository & Environment Installation

### Objective: Clone the codebase, configure a Python virtual environment, and install package dependencies.

1. Ensure the repository is cloned and navigate into the directory:

* **If you already cloned the repository in Section 3.1:**
  ```bash
  cd ~/alsageist
  ```

* **If performing a manual setup without Section 3.1 (repository not yet cloned):**
  ```bash
  cd ~
  git clone https://github.com/sancheza/alsageist.git
  cd alsageist
  ```

2. Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Upgrade packaging tools and install AlsaGeist in editable mode:

```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

4. Install GPIO hardware libraries:

The required GPIO library depends on your **Raspberry Pi OS version**, not the board model:

* **Raspberry Pi OS Bookworm (Recommended, for any Pi 3, Pi 4, or Pi 5):**
  Bookworm uses the Linux kernel character-device GPIO interface (`lgpio`):
  ```bash
  pip install gpiozero rpi-lgpio
  ```

* **Legacy OS (Bullseye / Buster):**
  Older OS releases use direct memory access (`RPi.GPIO`):
  ```bash
  pip install gpiozero RPi.GPIO
  ```

Verify package installation:

```bash
python3 -c "import sounddevice, numpy, gpiozero; print('Dependencies verified successfully.')"
```

Expected output:
```text
Dependencies verified successfully.
```

---

## 6. Piper Neural Voice Model Installation

### Objective: Download the local ONNX voice model weights.

Run the model download script:

```bash
python3 scripts/download_model.py --voice alan
```

Expected output:
```text
Downloading https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx -> models/piper/en_GB-alan-medium.onnx
Downloading https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json -> models/piper/en_GB-alan-medium.onnx.json
Successfully saved models/piper/en_GB-alan-medium.onnx
Successfully saved models/piper/en_GB-alan-medium.onnx.json
```

Verify that model files exist in `models/piper/`:

```bash
ls -lh models/piper/
```

Expected output:
```text
-rw-r--r-- 1 alsageist alsageist  61M en_GB-alan-medium.onnx
-rw-r--r-- 1 alsageist alsageist 4.8K en_GB-alan-medium.onnx.json
```

---

## 7. Environment Configuration (.env)

AlsaGeist reads runtime settings, API credentials, service parameters, and hardware audio calibration from a local `.env` file in the project root.

### Objective: Create and populate the `.env` configuration file from the template.

Copy the template file to `.env`:

```bash
cp env.example .env
nano .env
```

### Configuration Parameters

| Category | Variable | Default | Purpose |
| :--- | :--- | :--- | :--- |
| **Cloud STT** | `GROQ_API_KEY` | None | Groq Whisper Large v3 authentication token (Recommended for sub-second cloud transcription). |
| **Cloud STT / LLM** | `GEMINI_API_KEY` | None | Google Gemini authentication token (Alternative STT and conversational LLM fallback). |
| **Persona** | `PERSONALITY` | `grimjaw` | Active character profile identifier matching `data/personalities/<name>.json`. |
| **Dialogue Mode** | `LLM_FALLBACK` | `false` | Enables dynamic LLM fallback for indeterminate prompts instead of snarky canned responses (`true`/`false`). |
| **Service User** | `ALSAGEIST_USER` | `$USER` | Linux user account configured by `scripts/install_service.sh` for background systemd execution. |
| **Service Directory** | `ALSAGEIST_DIR` | Repository root | Installation directory path written to systemd unit file by `scripts/install_service.sh`. |
| **Audio Volume** | `VOLUME` | `0.8` | Output playback volume scaling factor (`0.0` to `1.0`). |
| **Microphone Gain** | `MIC_GAIN` | `1.5` | Digital input multiplier applied to raw captured audio frames before threshold detection. |
| **VAD Threshold** | `VAD_THRESHOLD` | `0.0035` | Minimum RMS audio energy required to trigger speech capture. |

### Minimal Recommended `.env` File

```ini
# Cloud Speech-to-Text (At least one required for online mode)
GROQ_API_KEY=gsk_your_groq_api_key_here

# Persona Configuration
PERSONALITY=grimjaw
LLM_FALLBACK=false

# Systemd Service Deployment
ALSAGEIST_USER=pi
ALSAGEIST_DIR=/home/pi/alsageist

# Audio Output & Input Calibration
VOLUME=0.8
MIC_GAIN=1.5
VAD_THRESHOLD=0.0035
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`). AlsaGeist automatically loads `.env` upon startup.

---

## 8. Running & Testing AlsaGeist Interactively

### Objective: Launch the AlsaGeist runtime in foreground mode to verify audio capture, synthesis, and LED behavior.

Execute the main application:

```bash
./main.py --verbose
```

Expected output:
```text
======================================================================
 💀 ALSA-GEIST CRYPT SKELETON IS ACTIVE & LISTENING
 Engine: INTENT | Voice: Piper Alan (Local ONNX)
 MicroSD / Memory Mode: Zero Disk Audio I/O (Pure RAM)
 Platform: Linux (Raspberry Pi Hardware Mode)
 LED Controller: RaspberryPiLEDController (GPIO 25, Button 23)
======================================================================
[AudioIO] Resampling 22050Hz -> 44100Hz for ALSA hardware compatibility.
[LED] Set state: IDLE (Breathing PWM active)
```

### Verification Checklist:
1. **LED Status:** The button LED transitions into a soft breathing pulse (0.5 Hz).
2. **Audio Capture:** Speak within range of the microphones ("What is your name?"). The LED transitions to solid brightness (`LISTENING`), then flickers (`SPEAKING`).
3. **Audio Output:** Grimjaw responds through the speaker: *"They call me Grimjaw the Sentinel!"*
4. **Mute State:** The microphone automatically mutes during speech synthesis to prevent audio feedback loops.
5. **Clean Exit:** Press `Ctrl+C` to terminate the process.

---

### 8.1 Audio Tuning & Calibration (Volume & Sensitivity)

Because hardware I2S DACs (such as the Google AIY Voice HAT) do not provide ALSA mixer gain controls, AlsaGeist provides software volume attenuation, digital microphone gain boosting, and voice activity energy calibration.

#### Configuration Options

| Parameter | Environment Variable | CLI Flag | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Output Volume** | `VOLUME` | `--volume` | `0.8` | Output playback scaling multiplier between `0.0` and `1.0` (e.g. `0.8` for 20% reduction). |
| **Microphone Gain** | `MIC_GAIN` | `--mic-gain` | `1.5` | Digital input multiplier applied to captured audio frames before threshold detection. |
| **VAD Threshold** | `VAD_THRESHOLD` | `--threshold` | `0.0035` | Minimum audio RMS energy required to trigger speech capture. |

#### Tuning via `.env` File

Add or modify these variables in your `.env` file:

```ini
# Audio volume & microphone sensitivity tuning
VOLUME=0.8
MIC_GAIN=1.5
VAD_THRESHOLD=0.0035
```

#### Tuning via Command-Line Flags

Override settings interactively during testing:

```bash
./main.py --volume 0.8 --mic-gain 2.0 --threshold 0.0030 --verbose
```

* **If the speaker is too loud:** Decrease `VOLUME` (e.g., `0.7` or `0.6`).
* **If you have to speak too close to the microphone:** Increase `MIC_GAIN` (e.g., `1.8` or `2.0`) and/or decrease `VAD_THRESHOLD` (e.g., `0.0030`).
* **If ambient room noise triggers false activations:** Increase `VAD_THRESHOLD` (e.g., `0.0050`).

---

### 8.2 Pre-Warming Dialogue Audio Cache (0ms TTFB Latency)

On low-power single-board computers like the Raspberry Pi 3, neural TTS synthesis for an unfamiliar phrase can take 6–10 seconds of CPU calculation. AlsaGeist includes a pre-warming utility that synthesizes dialogue phrases in advance and saves them into `models/cache/` as NumPy audio arrays.

> [!NOTE]
> Pre-warming is a **one-time setup step**. Because cache files are stored permanently on your MicroSD card in `models/cache/`, they persist across system reboots and power cycles. You do not need to run this command again unless you delete the cache directory or add new custom dialogue lines.

#### Objective: Pre-warm all dialogue responses into persistent disk cache.

Execute the pre-warming script:

```bash
cd ~/alsageist
./scripts/prewarm_cache.py --all
```

#### Checking Cache Build Progress

If running the pre-warming script in the background or over SSH, check the number of compiled cache files:

```bash
ls -1 ~/alsageist/models/cache/*.npy | wc -l
```

Expected output (counts toward the 383 total responses):
```text
383
```

To inspect running pre-warm processes and CPU activity:

```bash
ps aux | grep prewarm
```

---

### 8.3 Arcade Button Controls & Status LED States

The physical illuminated arcade button on the AIY Voice Kit (GPIO 23 for switch, GPIO 25 for LED) provides runtime controls and visual feedback without requiring an SSH terminal or keyboard.

#### Arcade Button Functions

| Action | Duration | Result |
| :--- | :--- | :--- |
| **Short Press** | < 5.0 seconds | Cycles audio volume in 10% steps (`0.0` to `1.0`), stops ongoing playback, resets conversational context, and speaks the new volume level ("Volume 80 percent."). Ignored when disabled. |
| **Long Hold** | 5.0 seconds | **Mute / Disable Toggle**: When active, holding for 5 seconds disables Grimjaw (turns off microphone capture and speaker output, stops ongoing playback, resets dialogue context, and turns off the button LED). Holding for 5 seconds again re-enables Grimjaw, restores microphone capture and speaker output, returns the LED to breathing `IDLE`, and resumes normal operation. |

#### Status LED Signaling

| State | Pattern | Meaning |
| :--- | :--- | :--- |
| `IDLE` | 0.5 Hz smooth breathing pulse (5%–85% duty cycle) | Resting ambient state; listening for porch visitors. |
| `LISTENING` | Solid 100% brightness | Speech detected; recording incoming microphone frames. |
| `FLASH` | 3 rapid pulses (~7 Hz) | Acknowledgment: confirmed speech was captured. |
| `THINKING` | Heartbeat double-pulse (~1.1 Hz "lub-dub... pause") | Active processing: transcribing speech, classifying intent, fetching LLM fallback, or synthesizing audio. |
| `SPEAKING` | 3–4 Hz organic audio-reactive flicker | Playing character response audio through the speaker. |
| `ERROR` | 5 Hz rapid square-wave alert blink | Exception occurred or cloud STT network failure. |
| `OFF` | Dark (0% duty cycle) | Muted / disabled via 5-second button hold. |

---

## 9. Autonomous Deployment via Systemd Service

### Objective: Configure AlsaGeist as a background systemd service that starts automatically upon power-on.

1. Create a systemd service definition file:

```bash
sudo nano /etc/systemd/system/alsageist.service
```

2. Add the following unit configuration (replace `<username>` with your actual Linux user, e.g. `alsageist` or `pi`):

```ini
[Unit]
Description=AlsaGeist Autonomous Conversational Display Engine
After=sound.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<username>
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=-/home/<username>/alsageist/.env
WorkingDirectory=/home/<username>/alsageist
ExecStart=/home/<username>/alsageist/venv/bin/python3 /home/<username>/alsageist/main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

3. Reload the systemd daemon, enable the service, and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable alsageist.service
sudo systemctl start alsageist.service
```

Alternatively, use the installer script to automatically configure and enable the unit from `.env`:

```bash
./scripts/install_service.sh --enable
```

4. Verify service status:

```bash
sudo systemctl status alsageist.service
```

Expected output:
```text
● alsageist.service - AlsaGeist Autonomous Conversational Display Engine
     Loaded: loaded (/etc/systemd/system/alsageist.service; enabled; preset: enabled)
     Active: active (running) since Sat 2026-10-10 15:45:00 UTC; 5s ago
   Main PID: 1248 (python3)
      Tasks: 4 (limit: 1024)
     Memory: 142.0M
        CPU: 1.821s
     CGroup: /system.slice/alsageist.service
             └─1248 /home/alsageist/alsageist/venv/bin/python3 /home/alsageist/alsageist/main.py
```

To follow live operational logs:

```bash
journalctl -u alsageist.service -f
```

---

## 10. Troubleshooting & Operational Runbook

### Issue: `PortAudio / sounddevice: No default input/output device found`
* **Cause:** ALSA has not initialized sound cards or user lacks permissions for `/dev/snd`.
* **Fix:** Add the user to the `audio` group and specify device flags explicitly:
  ```bash
  sudo usermod -aG audio $USER
  ./main.py --input-device 0 --output-device 0
  ```

### Issue: Voice playback sounds high-pitched, sped up, or distorted
* **Cause:** The ALSA hardware DAC only supports 44.1 kHz or 48 kHz stereo streams, while Piper TTS synthesizes at 22.05 kHz mono.
* **Fix:** AlsaGeist includes automatic linear interpolation resampling (`np.interp`). If overriding default streams, ensure output sample rate is set to 44100 Hz in `core/audio_io.py`.

### Issue: Microphone audio capture is silent or VAD does not trigger
* **Cause:** Hardware capture volume is muted or at 0% gain in ALSA.
* **Fix:** Run `alsamixer`, press `F4`, unmute capture channels (`Spacebar`), and raise microphone gain to 80%. Save settings via `sudo alsactl store`.

### Issue: `GPIO Zero / lgpio: Permission denied accessing GPIO`
* **Cause:** User account does not belong to the `gpio` group on Raspberry Pi OS.
* **Fix:** Add user to the `gpio` group and reboot:
  ```bash
  sudo usermod -aG gpio $USER
  sudo reboot
  ```

### Issue: `Speech recognition timed out / Network unreachable`
* **Cause:** Wi-Fi connection dropped or DNS resolution failed during STT inference.
* **Fix:** Verify internet routing with `curl -I https://api.groq.com` or `curl -I https://generativelanguage.googleapis.com`. If using Wi-Fi on a front porch, verify RSSI signal strength with `iwconfig wlan0`.

### Issue: `Git clone prompts for password` or `Support for password authentication was removed`
* **Cause:** The repository is private, and GitHub requires authentication credentials rather than an account password.
* **Fix:** Authenticate using GitHub CLI (`gh auth login`) before cloning:
  ```bash
  sudo apt install -y gh
  gh auth login
  ```
  Answer the prompts (`GitHub.com` -> `HTTPS` -> `Yes` -> `Login with a web browser`), enter the displayed code at `https://github.com/login/device`, and run `git clone` again. Alternatively, clone via SSH key (`git clone git@github.com:sancheza/alsageist.git`).

### Issue: `alsamixer: This sound device does not have any controls`
* **Cause:** The Google AIY Voice HAT V1 uses a fixed-gain I2S codec driver without hardware mixer registers.
* **Fix:** This is normal and expected behavior for the Voice HAT. AlsaGeist outputs audio directly to the DAC and requires no mixer adjustment. Verify that hardware audio works by running:
  ```bash
  speaker-test -c 2 -r 48000 -D default -t wav
  ```
