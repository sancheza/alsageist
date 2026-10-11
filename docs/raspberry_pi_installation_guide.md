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

1. Clone the repository into your home directory:

```bash
cd ~
git clone https://github.com/sancheza/alsageist.git
cd alsageist
```

2. Run the automated configuration script:

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

3. Reboot the Raspberry Pi after running the script:

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

### 3.5 Setting Mixer Volume Levels

#### Objective: Unmute audio channels and set hardware gain.

Launch the ALSA mixer:

```bash
alsamixer
```

1. Press `F6` to select the sound card (`snd_rpi_googlevoicehat_soundcar` or your USB audio card).
2. Use arrow keys to raise Playback Master volume to `85%`.
3. Press `F4` to switch to Capture controls.
4. Set Microphone capture level to `80%`.
5. Press `Esc` to exit.

Persist the volume configuration across reboots:

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

1. Navigate into the repository directory (or clone if performing a manual setup):

```bash
cd ~/alsageist
```

If you did not run the automated setup script in Section 3.1, clone the repository first:

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

4. Install GPIO controller libraries for Raspberry Pi:

On **Raspberry Pi OS Bookworm / Raspberry Pi 5 / Pi 4**:

```bash
pip install gpiozero rpi-lgpio
```

On **Legacy OS (Bullseye / Buster / Pi 3 RPi.GPIO backend)**:

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

## 7. Speech-to-Text Environment Configuration

AlsaGeist streams audio directly from memory to your chosen cloud STT provider (Groq Whisper or Gemini Flash) for intent recognition.

### Objective: Set API authentication tokens in your environment.

Choose one of the following providers:

#### For Groq Whisper Large v3 (Fastest Latency):

```bash
export GROQ_API_KEY="gsk_your_groq_api_key_here"
```

#### For Google Gemini Flash STT:

```bash
export GEMINI_API_KEY="AIzaSy_your_gemini_api_key_here"
```

To persist these variables across terminal sessions, append them to your profile:

```bash
echo 'export GROQ_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

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

## 9. Autonomous Deployment via Systemd Service

### Objective: Configure AlsaGeist as a background systemd service that starts automatically upon power-on.

1. Create a systemd service definition file:

```bash
sudo nano /etc/systemd/system/alsageist.service
```

2. Add the following unit configuration (replace `<username>` with your actual Linux user, e.g. `pi` or `alsageist`):

```ini
[Unit]
Description=AlsaGeist Autonomous Conversational Halloween Display
After=sound.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<username>
WorkingDirectory=/home/<username>/alsageist
Environment="PYTHONUNBUFFERED=1"
Environment="GROQ_API_KEY=gsk_your_groq_api_key_here"
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

4. Verify service status:

```bash
sudo systemctl status alsageist.service
```

Expected output:
```text
● alsageist.service - AlsaGeist Autonomous Conversational Halloween Display
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
