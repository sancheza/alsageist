# Getting Started with AlsaGeist in Docker on Raspberry Pi

Autonomous voice-driven conversational animatronic display engine container deployment for Raspberry Pi OS.

---

## 1. Overview

AlsaGeist runs as an isolated, containerized application with direct hardware access to ALSA audio devices and GPIO pin signaling. Externalized volumes for character personalities, Piper neural voice weights, and interaction logs ensure updates preserve user data and pre-warmed audio caches.

| Component | Container Path | Host External Path | Description |
| :--- | :--- | :--- | :--- |
| **Personalities** | `/app/data` | `./data` | Custom character JSON profiles and local regex intents |
| **Models & Cache** | `/app/models` | `./models` | Piper ONNX voice weights and zero-latency audio `.npy` files |
| **Telemetry** | `/app/logs` | `./logs` | Turn interaction logs (`interactions.jsonl`, `conversations.log`) |
| **Configuration** | `/app/.env` | `./.env` | Audio calibration and optional cloud STT/LLM API keys |

> [!NOTE]
> The container runs as non-root user `alsageist` with host networking, `/dev/snd` device access, and the host system D-Bus socket mounted for clean system poweroff.

---

## 2. Prerequisites

* Raspberry Pi 3 Model B, 3B+, 4, or 5 running **Raspberry Pi OS Lite (64-bit)** (Debian 12 Bookworm).
* Google AIY Voice HAT V1 (or compatible USB soundcard/speakerphone).
* Network connection for initial image pull and voice model download.
* A power supply rated for the board (Raspberry Pi 3 Model B: 5 V, 2.5 A). An inadequate supply throttles the CPU and slows every step; see [Under-Voltage and CPU Throttling](usage_guide.md#55-under-voltage-and-cpu-throttling-raspberry-pi).

---

## 3. Automated Onboarding (One-Step Setup)

### Objective: Configure host audio drivers, install Docker, provision external data directories, and autostart AlsaGeist.

1. Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/sancheza/alsageist.git
cd alsageist
```

2. Run the automated Docker deployment script:

```bash
./scripts/setup_docker.sh
```

Expected output:
```text
======================================================================
 AlsaGeist Docker Deployment & Raspbian Onboarding v1.0.0
 Profile: AIY | Target User: pi
 Install Path: /home/pi/alsageist
======================================================================

[1/5] Configuring Host Audio Drivers & System Permissions...
[2/5] Verifying Docker and Docker Compose...
[3/5] Setting Up Externalized Directories...
  Downloading Grimjaw neural voice model (en_GB-alan-medium)...
  Voice model downloaded successfully.
  Downloading Vosk offline speech recognition model...
  Vosk offline STT model installed successfully.

[4/5] Configuring Environment Variables...
  Enter Groq API Key [leave empty for offline STT]: 
  Enter Gemini API Key [leave empty for offline fallback]: 
  Configured /home/pi/alsageist/.env (mode 600).

  Pre-warm audio dialogue cache now (~25 responses, ~15 min on Pi 3)? (y/N): n

[5/5] Configuring Systemd Autostart Service...
  Enabled alsageist.service autostart on system boot.

======================================================================
 AlsaGeist Setup Completed Successfully!

 Dialogue Speech Cache:
   - To pre-warm offline dialogue responses with zero playback latency, run:
     docker compose exec alsageist python3 scripts/prewarm_cache.py --all

 IMPORTANT: A system reboot is required to activate the AIY Voice HAT overlay:
   sudo reboot
======================================================================
```

3. If using the Google AIY Voice HAT, reboot the host to activate the kernel soundcard driver overlay:

```bash
sudo reboot
```

AlsaGeist starts automatically on boot via systemd.

---

## 4. Manual Deployment with Docker Compose

### Objective: Build or pull and run the container manually using Docker Compose.

1. Create required external host directories:

```bash
mkdir -p data/personalities models/piper models/cache logs
```

2. Download default personality, Piper TTS voice, and Vosk offline speech model:

```bash
cp data/personalities/grimjaw.json data/personalities/
python3 scripts/download_model.py --all
```

3. Create the environment configuration file:

```bash
cp env.example .env
chmod 600 .env
```

4. Launch the container in the background:

```bash
docker compose up -d
```

Expected output:
```text
[+] Running 1/1
 ✔ Container alsageist  Started
```

5. Verify running status and inspect logs:

```bash
docker compose logs -f
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
```

---

## 5. Pre-Warming Dialogue Audio Cache (0ms Latency)

### Objective: Pre-render all character responses to disk for zero-latency (<1 ms) playback.

On low-power hardware like the Raspberry Pi 3, neural TTS generation requires 4–6 seconds of CPU time per uncached phrase. Running the pre-warming utility synthesizes all dialogue lines in advance and saves them into the mounted `./models/cache/` volume on the host SD card:

```bash
docker compose exec alsageist python3 scripts/prewarm_cache.py --all
```

Check the number of compiled audio cache files on the host:

```bash
ls -1 models/cache/*.npy | wc -l
```

Expected output:
```text
401
```

---

## 6. Seamless Container Updates

### Objective: Upgrade to the latest container image without losing character profiles, voice models, or interaction logs.

1. Pull the latest image:

```bash
docker compose pull
```

2. Recreate the container with zero state loss:

```bash
docker compose up -d --remove-orphans
```

All character configurations in `data/`, cached speech in `models/cache/`, and turn logs in `logs/` remain intact on the host.

---

## 7. Real-Time Host CLI Control

### Objective: Inspect status or hot-swap personalities from the host command line.

The control socket is private to the container user (`/tmp/alsageist-0/alsageist.sock`, mode 0600). Run `alsageist_ctl.py` inside the container:

**Starting Host & Directory:** Raspberry Pi, any directory

```bash
docker exec alsageist python3 scripts/alsageist_ctl.py status
```

Expected output:
```text
{
  "active_personality": "grimjaw",
  "engine": "intent",
  "online": false,
  "uptime_sec": 342.1
}
```

To initiate an orderly host shutdown from the CLI:
```bash
docker exec alsageist python3 scripts/alsageist_ctl.py shutdown --yes
```

---

## 8. Physical Controls & Host Shutdown

The containerized service has direct access to the GPIO button pins. Host poweroff needs the host's system D-Bus socket, which `docker-compose.shutdown.yml` mounts:

**Starting Host & Directory:** Raspberry Pi, repository root

```bash
docker compose -f docker-compose.yml -f docker-compose.shutdown.yml up -d
```

To keep the override after reboots, add this line to `.env`; `docker compose` and the systemd unit from `scripts/setup_docker.sh` then both use it:

```text
COMPOSE_FILE=docker-compose.yml:docker-compose.shutdown.yml
```

> [!WARNING]
> The D-Bus mount gives the container access to the host's system services. Without it, the long hold and the `shutdown` command announce power-off and then return to normal operation.


* **Short Press (< 5s):** Cycles playback volume in 10% steps.
* **Medium Hold (5s):** Mutes / disables Grimjaw (turns off audio and LED).
* **Long Hold (10s):** Initiates safe host poweroff. The container speaks offline pre-cached "Powering off.", extinguishes the status LED, and signals the host systemd daemon over the D-Bus socket to cleanly shut down the Raspberry Pi. `BUTTON_SHUTDOWN_ENABLED=false` turns the long hold into an ordinary hold (mute / disable); the `shutdown` control command still works.

---

## 9. Troubleshooting

### Problem: Audio input or output fails to register in container.

Verify ALSA devices on the host:

```bash
aplay -l && arecord -l
```

Expected output:
```text
card 0: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcard], device 0: Google voiceHAT SoundCard HiFi voicehat-hifi-0 []
```

If no soundcards appear, ensure the kernel overlay is present in `/boot/firmware/config.txt`:

```text
dtparam=audio=off
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

### Problem: Startup or replies take tens of seconds.

The CPU is probably throttled by an inadequate power supply. Run these commands on the host, not in the container:

```bash
vcgencmd measure_clock arm
vcgencmd get_throttled
```

Expected output when throttled:
```text
frequency(48)=600000000
throttled=0x50005
```

A clock of `1200000000` and a value of `0x0` mean the supply is not the cause. Otherwise see [Under-Voltage and CPU Throttling](usage_guide.md#55-under-voltage-and-cpu-throttling-raspberry-pi).

### Problem: Arcade button or LED does not respond.

Ensure the target user is a member of the `gpio` group on the host:

```bash
groups $USER | grep gpio
```
