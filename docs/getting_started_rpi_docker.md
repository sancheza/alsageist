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

* Raspberry Pi 3 Model B, 3B+, 4, or 5 running **Raspberry Pi OS Lite (64-bit)** (Debian 12 Bookworm), image `2026-10-06-raspios-bookworm-arm64-lite.img` or newer.
* Google AIY Voice HAT V1 (or compatible USB soundcard/speakerphone).
* Network connection for initial image pull and voice model download.
* A power supply rated for the board (Raspberry Pi 3 Model B: 5 V, 2.5 A). An inadequate supply throttles the CPU and slows every step; see [Under-Voltage and CPU Throttling](usage_guide.md#56-under-voltage-and-cpu-throttling-raspberry-pi).

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

   This hides the low-voltage warning overlay and stops the firmware from lowering the CPU frequency when it detects low voltage, so the Pi keeps running at full speed under heavy load. It can keep speech recognition and synthesis response times more consistent.

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

## 4. Automated Setup

### Objective: Configure host audio drivers, install Docker, provision external data directories, and autostart AlsaGeist.

> [!IMPORTANT]
> All steps from here on run **on the Raspberry Pi itself** (over SSH, or with a keyboard and monitor attached), not on your workstation. `setup_docker.sh` configures the Pi's operating system and does not run on Linux desktops, macOS, or Windows.

1. On the Raspberry Pi, bring the freshly installed operating system up to date and make sure `git` is installed. The update pulls in current kernel, firmware, and package fixes before the setup script configures the audio drivers:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y git
sudo reboot
```

Wait a minute for the Pi to restart, then reconnect with SSH.

2. Clone the repository into your home directory (`~`, which is `/home/<username>`, e.g. `/home/pi`) and navigate into the project directory:

```bash
cd ~
git clone https://github.com/sancheza/alsageist.git
cd alsageist
```

3. Run the automated Docker deployment script:

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

  Pre-warm audio dialogue cache now (all responses, ~50 min on Pi 3)? (y/N): n

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

#### About the optional API keys

During step 4 of the script, you are asked for two API keys. **Both are optional. Press Enter to leave either one blank**, and AlsaGeist runs fully offline. You can add them at any time later.

* **Groq API key** (`GROQ_API_KEY`): turns on cloud **STT** (speech-to-text, the step that converts what a visitor says into text). Groq runs Whisper Large v3, which is faster and more accurate than the offline Vosk model used when this is blank. A Groq key can also power the LLM fallback described below.
* **Gemini API key** (`GEMINI_API_KEY`): can power the **LLM fallback** (large language model), where Gemini improvises a reply when Grimjaw's built-in responses don't cover what a visitor said. When no key is set, he uses canned fallback lines instead.

If you enter either key, the script also sets `LLM_FALLBACK=true` in `.env`.

To add or change keys later, edit `.env` on the Pi, then recreate the container so it picks up the change:

```bash
cd ~/alsageist
nano .env
docker compose up -d
```

Set `GROQ_API_KEY=` and/or `GEMINI_API_KEY=` to your key, and set `LLM_FALLBACK=true` if you want the LLM fallback.

#### About the voice command password

Privileged spoken commands (shutdown, mode switches, fallback toggles) require a verbal password so visitors can't shut the device down or change its mode. The default is `alsageist`. This default is public, so **change it if you wish**. Set `VOICE_COMMAND_PASSWORD=` in `.env` to a word of your choice (or to `none` to disable the check), then recreate the container with `docker compose up -d`.

#### About the optional audio cache pre-warm

The script also asks whether to pre-warm the dialogue audio cache, which takes about 50 minutes on a Raspberry Pi 3. **It is safe to answer `n` (the default) and do it later.** Pre-warming removes the 4–6 second synthesis pause on a Raspberry Pi 3 for responses that aren't cached yet; AlsaGeist works without it.

To pre-warm later, once AlsaGeist is running (after the reboot, or after `docker compose up -d`), run this on the Pi:

```bash
cd ~/alsageist
docker compose exec alsageist python3 scripts/prewarm_cache.py --all
```

See [Pre-Warming Dialogue Audio Cache](#6-pre-warming-dialogue-audio-cache-0ms-latency) for details.

4. If using the Google AIY Voice HAT, reboot the host to activate the kernel soundcard driver overlay:

```bash
sudo reboot
```

AlsaGeist starts automatically on boot via systemd.

> [!IMPORTANT]
> **Setup is complete after this step.** Skip section 5 (Manual Deployment): it is an alternative to the automated script, for people who do not use `setup_docker.sh`. Continue to the checks below to confirm AlsaGeist is running.

#### Confirm AlsaGeist is running

**The first boot after setup can take a long time.** If the pre-built image cannot be pulled from the registry, Docker builds it on the Pi itself. On a Raspberry Pi 3 this takes well over 10 minutes, during which `docker ps` shows no containers. This is normal. Follow the progress from any SSH session on the Pi:

```bash
journalctl -u alsageist -f
```

Press `Ctrl+C` to stop following. When the build finishes, check the service and the container:

```bash
sudo systemctl status alsageist
docker ps
```

You should see `Active: active (exited)` for the service (it starts the container and then exits, which is expected) and a container named `alsageist` with a status of `Up ...` in `docker ps`.

Then confirm the application itself is responding:

```bash
docker exec alsageist python3 scripts/alsageist_ctl.py status
docker logs alsageist --tail 30
```

The status command prints JSON that includes the active personality (`grimjaw`). The logs list your audio output and capture devices; with the AIY Voice HAT you should see `sndrpigooglevoi`. If you don't, see [Troubleshooting](#10-troubleshooting).

#### Try talking to Grimjaw

Once the service is running, speak a full sentence near the microphone and pause. Grimjaw answers when he hears you stop. Here are a few things to try:

* **Get to know him:** *"What's your name?"*, *"Are you dead?"*, *"How old are you?"*, *"Where do you keep your brain?"*
* **Candy:** *"Can I have some candy?"*, *"What is your favorite candy?"*
* **Costumes:** *"I'm dressed up as Spider-Man."*, *"I'm dressed as Barbie."*, *"What is your favorite costume?"*
* **Silly questions:** *"Do you get cold without any skin on?"*, *"Can you tickle yourself?"*, *"Do you own a pet?"*
* **Pop culture:** *"Tell me a Seinfeld quote."*
* **Check on the system:** *"What mode are you in?"*, *"What voice model are you using?"*

See [Spoken Voice System Commands](usage_guide.md#43-spoken-voice-system-commands) for more spoken commands, including the ones that need the voice command password.

---

## 5. Manual Setup with Docker Compose

### Objective: Build or pull and run the container manually using Docker Compose.

> [!NOTE]
> Skip this section if you ran `./scripts/setup_docker.sh` in section 4. It already did everything below. This section is an alternative for people who do not use the setup script.

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

## 6. Pre-Warming Dialogue Audio Cache (0ms Latency)

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

## 7. Seamless Container Updates

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

## 8. Real-Time Host CLI Control

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

## 9. Physical Controls & Host Shutdown

The containerized service has direct access to the GPIO button pins. Host poweroff needs the host's system D-Bus socket and root privilege (to satisfy polkit authorization for systemd-logind PowerOff), which `docker-compose.shutdown.yml` configures:

**Starting Host & Directory:** Raspberry Pi, repository root

```bash
docker compose -f docker-compose.yml -f docker-compose.shutdown.yml up -d
```

To keep the override after reboots, add this line to `.env`; `docker compose` and the systemd unit from `scripts/setup_docker.sh` then both use it:

```text
COMPOSE_FILE=docker-compose.yml:docker-compose.shutdown.yml
```

To verify a finished deployment (container health, hardware access, and whether the container is authorized to power off the host), run this on the Pi from the repository root. It is read-only; add `--shutdown` to also do a live power-off test:

```bash
./scripts/verify_docker_deploy.sh
```

> [!WARNING]
> The D-Bus mount and root user override give the container access to host system management. Without them, the long hold and the `shutdown` command announce power-off and then return to normal operation.


* **Short Press (< 5s):** Cycles playback volume in 10% steps.
* **Medium Hold (5s):** Mutes / disables Grimjaw (turns off audio and LED).
* **Long Hold (10s):** Initiates safe host poweroff. The container speaks offline pre-cached "Powering off.", extinguishes the status LED, and signals the host systemd daemon over the D-Bus socket to cleanly shut down the Raspberry Pi. `BUTTON_SHUTDOWN_ENABLED=false` turns the long hold into an ordinary hold (mute / disable); the `shutdown` control command still works.

---

## 10. Troubleshooting

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

A clock of `1200000000` and a value of `0x0` mean the supply is not the cause. Otherwise see [Under-Voltage and CPU Throttling](usage_guide.md#56-under-voltage-and-cpu-throttling-raspberry-pi).

### Problem: Arcade button or LED does not respond.

Ensure the target user is a member of the `gpio` group on the host:

```bash
groups $USER | grep gpio
```
