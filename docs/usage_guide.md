# AlsaGeist User & Operations Guide

Instructional guide for operating, configuring, and customizing AlsaGeist across all operational scenarios.

---

## 1. Feature Highlights

AlsaGeist is a real-time conversational display engine designed for interactive props, skull animatronics, and Halloween candy bowl sentinels.

| Feature | Description |
| :--- | :--- |
| **Hands-Free Conversational Loop** | Real-time Voice Activity Detection (VAD) samples microphone audio in RAM and detects speech boundaries without requiring push-to-talk. |
| **Instant Offline Banter** | Over 150 deterministic intents respond in under 5 milliseconds with zero internet connection. |
| **Local Neural TTS** | High-fidelity character speech synthesized locally via Piper ONNX neural voices. |
| **Hybrid Cloud Fallback** | Optional streaming cloud STT (Groq Whisper Large v3 or Google Gemini) and dynamic LLM fallback for open-ended conversation. |
| **Physical Controls** | Arcade push-button supports software volume cycling, hold-to-mute, and 10-second safe host poweroff. |
| **Dynamic Status LED** | Pulse-width modulation (PWM) feedback indicates ambient listening, speech detection, thinking, and speaking. |
| **Live Socket IPC** | Local Unix domain socket allows real-time adjustments (volume, personality hot-swap, speech broadcast) without daemon restarts. |
| **Zero Disk Wear** | Turn audio capture and synthesis execute in memory arrays to prevent SD card degradation. |

---

## 2. Running AlsaGeist

AlsaGeist supports three execution models: interactive foreground mode, Docker container mode, and background systemd service mode.

### 2.1 Interactive Terminal Mode (Foreground)

#### Objective: Launch AlsaGeist directly in the terminal for real-time testing and immediate console logging.

1. Activate your Python virtual environment:

```bash
cd ~/alsageist
source venv/bin/activate
```

2. Start the application:

```bash
./main.py
```

Expected output:
```text
======================================================================
 ALSA-GEIST [GRIMJAW THE SKELETON] IS ACTIVE & LISTENING
 Engine: INTENT | Voice: Alan (Local ONNX)
 Interaction Logging: Active -> logs/interactions.jsonl
 Speak into your microphone to start a conversation.
 Press Ctrl+C to stop.
======================================================================

🟢 [LED: IDLE (Breathing ~ ~ ~)]
```

Speak into the microphone. Press `Ctrl+C` to cleanly exit.

---

### 2.2 Docker Container Mode

#### Objective: Start, monitor, and stop AlsaGeist running inside an isolated Docker container.

1. Start the container in background daemon mode:

```bash
docker compose up -d
```

Expected output:
```text
[+] Running 1/1
 ✔ Container alsageist  Started
```

2. Inspect live container logs:

```bash
docker compose logs -f
```

3. Stop the container:

```bash
docker compose down
```

4. Restart the container:

```bash
docker compose restart
```

---

### 2.3 Autonomous Background Service (systemd)

#### Objective: Control the native background service on Raspberry Pi OS.

1. Check current service state:

```bash
sudo systemctl status alsageist.service
```

Expected output:
```text
● alsageist.service - AlsaGeist Autonomous Conversational Display Engine
     Loaded: loaded (/etc/systemd/system/alsageist.service; enabled)
     Active: active (running)
```

2. Restart the service:

```bash
sudo systemctl restart alsageist.service
```

3. Follow real-time systemd journal logs:

```bash
journalctl -u alsageist.service -f
```

---

## 3. Real-Time Remote Management (`alsageist_ctl.py`)

The control client communicates with a running AlsaGeist instance (native or Docker) over a local Unix domain socket at `/tmp/alsageist-<uid>/alsageist.sock` (directory mode 0700, socket mode 0600). Only the user running the daemon can connect. For Docker, prefix each command with `docker exec alsageist`.

### 3.1 Monitoring Live Health & Dashboard

#### Objective: Inspect active personality, volume, mode, and process uptime.

Execute the `status` command:

```bash
python3 scripts/alsageist_ctl.py status
```

Expected output:
```text
======================================================================
 AlsaGeist Runtime Status
======================================================================
 State:           ACTIVE
 Personality:     grimjaw (Grimjaw the Skeleton)
 Voice Model:     Alan
 Dialogue Engine: INTENT (Offline regex matcher)
 LLM Fallback:    Disabled (canned fallbacks)
 Network Mode:    OFFLINE
 Volume:          50% (0.5)
 Microphone:      UNMUTED
 LED State:       IDLE
 Uptime:          384s
 Process PID:     1482
======================================================================
```

*Pass `--json` to output raw JSON formatted data for external automation.*

---

### 3.2 Adjusting Audio Volume

#### Objective: Query or alter playback volume without restarting the daemon.

1. Query current volume:

```bash
python3 scripts/alsageist_ctl.py volume
```

Expected output:
```text
Volume: 50% (0.5)
```

2. Set absolute volume (0% to 100%):

```bash
python3 scripts/alsageist_ctl.py volume 75
```

3. Increase volume by 10%:

```bash
python3 scripts/alsageist_ctl.py volume +10
```

4. Decrease volume by 15%:

```bash
python3 scripts/alsageist_ctl.py volume -15
```

---

### 3.3 Muting & Disabling the Display (Sleep/Wake)

#### Objective: Silence audio and pause listening while keeping the process running.

1. Mute microphone capture only:

```bash
python3 scripts/alsageist_ctl.py mute
```

2. Unmute microphone capture:

```bash
python3 scripts/alsageist_ctl.py unmute
```

3. Put the prop into full standby sleep (mutes mic and speaker, turns off LED):

```bash
python3 scripts/alsageist_ctl.py disable
```

4. Wake the prop from standby sleep:

```bash
python3 scripts/alsageist_ctl.py enable
```

5. Toggle between sleep and active states:

```bash
python3 scripts/alsageist_ctl.py toggle
```

---

### 3.4 Operational Mode Switching (Online vs Offline)

#### Objective: Switch dialogue routing dynamically.

1. Force offline mode (local regex intents and local Piper TTS only):

```bash
python3 scripts/alsageist_ctl.py mode offline
```

2. Enable online mode (cloud STT and LLM fallback):

```bash
python3 scripts/alsageist_ctl.py mode online
```

3. Return to automatic detection (connectivity is probed every 15 minutes):

```bash
python3 scripts/alsageist_ctl.py mode auto
```

> [!NOTE]
> A forced `online` or `offline` mode stays in effect until `mode auto` is sent or the service restarts. `status` shows `(forced)` or `(auto)` next to the network mode.

---

### 3.5 Broadcasting Custom Spoken Announcements

#### Objective: Make the prop speak an immediate custom phrase through the hardware speaker.

```bash
python3 scripts/alsageist_ctl.py say "Happy Halloween, mortals! Beware the crypt!"
```

Expected output:
```text
Spoke: "Happy Halloween, mortals! Beware the crypt!"
```

---

### 3.6 Hot-Swapping Personalities

#### Objective: Change the active character persona without restarting AlsaGeist.

```bash
python3 scripts/alsageist_ctl.py personality grimjaw
```

Expected output:
```text
Switched personality to 'grimjaw' (Grimjaw the Skeleton).
```

---

### 3.7 Safe System Shutdown

#### Objective: Orderly poweroff the Raspberry Pi host.

```bash
python3 scripts/alsageist_ctl.py shutdown
```

Expected output:
```text
Host system shutdown initiated.
```

---

## 4. Hardware Button & LED Signaling

The physical illuminated arcade button on GPIO 23 and LED on GPIO 25 provide direct hardware control without terminal access.

### 4.1 Physical Arcade Button Functions

| Action | Hold Duration | System Response |
| :--- | :--- | :--- |
| **Short Press** | < 5.0 seconds | Cycles audio volume in 10% steps (`0.0` to `1.0`), cancels active speech playback, resets conversation context, and announces the new level (e.g. *"Volume 60 percent"*). |
| **Medium Hold** | 5.0 seconds | **Mute / Sleep Toggle**: Disables microphone capture, halts audio output, and extinguishes the button LED. Holding for 5 seconds again re-enables normal operation and restores the breathing idle LED. |
| **Long Hold** | 10.0 seconds | **Safe System Poweroff**: Speaks offline pre-cached *"Powering off."*, turns off the status LED, and cleanly powers down the Raspberry Pi host (works natively and in Docker). |

---

### 4.2 Status LED Signaling Reference

| State | Pattern | Operational Meaning |
| :--- | :--- | :--- |
| `IDLE` | 0.5 Hz smooth breathing pulse (5%–85% brightness) | Resting state on the porch, actively listening for visitor speech. |
| `LISTENING` | Solid 100% illumination | Speech energy detected; actively streaming microphone frames into RAM. |
| `FLASH` | 3 rapid pulses (~7 Hz) | Acknowledgment: confirmed speech capture complete. |
| `THINKING` | Heartbeat double-pulse (~1.1 Hz "lub-dub... pause") | Processing: transcribing speech, matching intent, or rendering TTS audio. |
| `SPEAKING` | 3–4 Hz organic audio-reactive flicker | Playing character response audio through the amplified speaker. |
| `ERROR` | 5 Hz rapid alert blink | Subsystem exception or network STT timeout. |
| `OFF` | Dark (0% brightness) | Muted / sleeping via 5-second button hold or powered down. |

---

## 5. Offline vs. Online Operational Modes

AlsaGeist adapts automatically based on network connectivity and API keys configured in `.env`.

### 5.1 Pure Offline Mode (Default)

In Pure Offline Mode, AlsaGeist operates with **zero internet connection**:
* Spoken phrases are matched against deterministic regex rules in `data/personalities/grimjaw.json`.
* Voice responses are loaded instantly (<1 ms) from pre-warmed audio cache files in `models/cache/`.
* Any unrecognized phrase triggers one of Grimjaw's snarky canned fallback lines.
* **Requirements:** No API keys required. Ideal for outdoor installations with weak Wi-Fi.

### 5.2 Hybrid Online Mode

When internet access and API keys are present, AlsaGeist enables cloud acceleration:
* **Cloud STT:** Uses Groq Whisper Large v3 (sub-second audio transcription) or Google Gemini Flash STT.
* **Dynamic LLM Fallback:** When a visitor asks an unfamiliar question not covered by the 150+ offline intents, Grimjaw queries an LLM to generate a witty, in-character response under 2 sentences.
* **Automatic Resilience:** If every cloud transcription service fails, the local Vosk model answers that question and AlsaGeist probes the network immediately (at most once every 30 seconds). If the network is down it switches to offline mode, announces "NOW IN OFFLINE MODE", and later questions skip the cloud entirely. If only a cloud service is down while the network is up, it stays online and tries the next service. Connectivity is also probed every 15 minutes, which brings it back online and announces "NOW IN ONLINE MODE". A mode set with `alsageist_ctl.py mode online` or `mode offline` is never changed automatically.

### 5.3 Configuring Operational Parameters in `.env`

Edit your `.env` file to select modes:

```ini
# Personality Profile
PERSONALITY=grimjaw

# Enable or disable dynamic LLM fallback for unrecognized questions
LLM_FALLBACK=true

# Cloud API Keys (Leave blank for 100% offline mode)
GROQ_API_KEY=gsk_your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here

# Hardware Calibration
VOLUME=0.5
MIC_GAIN=1.5
VAD_THRESHOLD=0.0035
```

### 5.4 Interaction Logs & Privacy

| Setting | Default | Effect |
| :--- | :--- | :--- |
| `LOG_MAX_MB` | `5` | Size at which `logs/interactions.jsonl` and `logs/conversations.log` rotate |
| `LOG_BACKUPS` | `3` | Rotated copies kept per file (`.1` is the newest) |
| `LOG_TRANSCRIPTS` | `true` | `false` stores `[not stored]` instead of visitor speech and replies; timings, intents and audio levels are still logged |

* Log files are readable only by the user running AlsaGeist (mode 0600).
* In online mode each utterance is sent to the configured cloud service (Groq or Google) for transcription, and unrecognized questions go to the LLM. In offline mode audio stays on the device.

### 5.5 Under-Voltage and CPU Throttling (Raspberry Pi)

#### Objective: Recognize a weak power supply, understand its effect on response speed, and find the cause.

> [!IMPORTANT]
> When the power supply cannot hold the Raspberry Pi's voltage, the firmware throttles the CPU. Every CPU-bound step then runs slower: loading the voice model, synthesizing a new line, offline speech recognition, and starting playback. Nothing fails and no error appears; replies are only slower.

**Effect on a Raspberry Pi 3 Model B:**

| Condition | ARM clock |
| :--- | :--- |
| Normal | 1200 MHz |
| Throttled by under-voltage | 600 MHz |

Timings measured on a Raspberry Pi 3 Model B while it was throttled to 600 MHz:

| Step | Time |
| :--- | :--- |
| Loading the Piper voice model (`say.py` with a new phrase) | about 35 s |
| Synthesizing a new sentence | 11 to 13 s |
| Offline (Vosk) recognition of a short question | about 7 s |

#### Detect throttling

| Method | Command | What it shows |
| :--- | :--- | :--- |
| Service log (automatic) | `journalctl -u alsageist.service \| grep -i throttl` | Warnings written by the service (see below) |
| Firmware flags | `vcgencmd get_throttled` | Bit field; values and bits are listed below |
| Actual CPU clock | `vcgencmd measure_clock arm` | `600000000` when throttled, `1200000000` at full speed on a Raspberry Pi 3 Model B |
| Kernel log | `journalctl -k \| grep -i voltage` | Timestamped `Undervoltage detected!` and `Voltage normalised` events |
| Live alarm | `cat /sys/class/hwmon/hwmon*/in0_lcrit_alarm 2>/dev/null` | `1` while the supply is under-voltage, `0` otherwise |

Expected output on a throttled Raspberry Pi:
```text
$ vcgencmd get_throttled
throttled=0x50005
$ vcgencmd measure_clock arm
frequency(48)=600000000
$ journalctl -k | grep -i voltage
Oct 17 11:03:51 raspberrypi kernel: hwmon hwmon1: Undervoltage detected!
Oct 17 11:03:57 raspberrypi kernel: hwmon hwmon1: Voltage normalised
```

Values of `vcgencmd get_throttled`:

| Value | Meaning |
| :--- | :--- |
| `0x0` | No under-voltage or throttling since boot |
| `0x50000` | Under-voltage and throttling happened earlier since boot; not happening now |
| `0x50005` | Under-voltage and throttling are happening now, and have happened since boot |

Each bit:

| Bit | Value | Meaning |
| :--- | :--- | :--- |
| 0 | `0x1` | Under-voltage detected now |
| 1 | `0x2` | ARM frequency capped now |
| 2 | `0x4` | Currently throttled |
| 3 | `0x8` | Soft temperature limit active now |
| 16 | `0x10000` | Under-voltage has occurred since boot |
| 17 | `0x20000` | ARM frequency capping has occurred since boot |
| 18 | `0x40000` | Throttling has occurred since boot |
| 19 | `0x80000` | Soft temperature limit has occurred since boot |

> [!NOTE]
> `/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq` can report the maximum clock while the firmware has the CPU at half speed. Use `vcgencmd measure_clock arm` to read the real clock.

> [!NOTE]
> In the Docker deployment, run these commands on the host. The container image does not include `vcgencmd`.

#### Fix under-voltage

1. Use a power supply rated for the board (Raspberry Pi 3 Model B: 5 V, 2.5 A; see the [hardware overview](overview.md)).
2. Use a short, thick power cable, and power the board from a wall adapter, not a computer or hub port.
3. Disconnect USB devices that are not needed.
4. Reboot, then check `vcgencmd get_throttled` while the service is working, for example while it answers a question. Bits 16 to 19 stay set until the next boot, so `0x0` after a reboot under load confirms the fix.

#### Service warnings

The service checks the firmware flags once a minute. While the CPU is throttled or under-voltage it logs a warning at most every 15 minutes, even when the supply keeps dipping and recovering. After the CPU has stayed clear for 5 minutes it logs a notice, and the next problem warns immediately. A condition that happened earlier since boot is logged once.

| Setting | Default | Effect |
| :--- | :--- | :--- |
| `THROTTLE_CHECK_SEC` | `60` | Seconds between checks; `0` disables the monitor |
| `THROTTLE_WARN_SEC` | `900` | Minimum seconds between repeated warnings while throttling continues |
| `LOW_POWER_ANNOUNCE` | `true` | `false` turns off the spoken low-power line (the log warnings stay) |
| `LOW_POWER_ANNOUNCE_SEC` | `3600` | Minimum seconds between spoken low-power lines |

Example log lines (`journalctl -u alsageist.service`):
```text
Warning: CPU throttling: under-voltage detected, currently throttled, CPU 61.2C. Check the power supply and USB cable (the Pi 3 needs a solid 5 V, 2.5 A).
CPU is no longer throttled, CPU 55.0C.
Warning: since boot the CPU reported: under-voltage has occurred, throttling has occurred, CPU 55.0C. Check the power supply and USB cable (the Pi 3 needs a solid 5 V, 2.5 A).
```

* The monitor reads `vcgencmd get_throttled`, falling back to the kernel's under-voltage alarm. On machines with neither it does nothing.

#### Spoken low-power line

While the firmware reports under-voltage, the persona says one line from the profile's `low_power_lines` list out loud, such as "My power levels are low. I need to frighten more children to recharge."

| Rule | Behavior |
| :--- | :--- |
| Frequency | At most once per `LOW_POWER_ANNOUNCE_SEC` (one hour by default) |
| Trigger | Under-voltage detected now; a thermal limit alone does not trigger it |
| Timing | Not spoken during a conversation or within 10 seconds of one, or while the display is asleep. The attempt is repeated at the next check, so a skipped line does not use up the hour |
| Wording | A random line from `low_power_lines` in the active personality profile. A profile without the list says a plain sentence about the power supply |

Turn it off in `.env`:
```ini
LOW_POWER_ANNOUNCE=false
```

After editing `low_power_lines`, run `python3 scripts/prewarm_cache.py --all` so the new lines play without a synthesis delay.

---

## 6. Audio Calibration & Noise Tuning

Environmental conditions (a quiet indoor testing room vs a front porch with wind, rustling leaves, and street traffic) require calibrating audio levels.

### 6.1 Volume Calibration

The AIY Voice HAT uses digital software volume attenuation:
* **Recommended Indoor / Bench Testing:** `VOLUME=0.4` to `0.5`
* **Recommended Outdoor Porch Prop:** `VOLUME=0.8` to `1.0`

Adjust via flag:
```bash
./main.py --volume 0.7
```

---

### 6.2 Microphone Input Sensitivity (`MIC_GAIN`)

`MIC_GAIN` multiplies raw audio buffer amplitudes before Voice Activity Detection evaluates energy levels:
* **Standard speaking distance (1–2 feet):** `MIC_GAIN=1.5` (Default)
* **Distanced visitors (4–6 feet away from the prop):** `MIC_GAIN=2.0` to `2.5`
* **Loud amplified surroundings:** `MIC_GAIN=1.0`

Adjust via flag:
```bash
./main.py --mic-gain 2.0
```

---

### 6.3 Voice Activity Threshold (`VAD_THRESHOLD`)

`VAD_THRESHOLD` sets the minimum Root Mean Square (RMS) energy needed to transition from `IDLE` to `LISTENING`:
* **Quiet indoor room:** `VAD_THRESHOLD=0.0025` (Very sensitive)
* **Standard porch installation:** `VAD_THRESHOLD=0.0035` (Default balance)
* **High-noise environments (wind, street traffic, ambient music):** `VAD_THRESHOLD=0.0045` to `0.0060`

Adjust via flag:
```bash
./main.py --threshold 0.0045
```

---

## 7. Personality Customization & Intent Management

### 7.1 Anatomy of a Personality Profile

Personalities live in `data/personalities/<name>.json`. A profile defines character identity, default voice, intent categories, and canned fallbacks:

```json
{
  "personality": "grimjaw",
  "display_name": "Grimjaw the Skeleton",
  "description": "Sarcastic, dry crypt skeleton guarding the Halloween candy bowl.",
  "voice": "Alan",
  "intents": [
    {
      "intent": "candy_take_one",
      "patterns": [
        "\\b(can i have|take|grab)\\b.*\\b(candy|sweet|treat)\\b"
      ],
      "responses": [
        "Just one piece, mortal! Greedy hands get turned to bone!",
        "Take one. If you take two, my skull won't be the only one on this porch."
      ]
    }
  ],
  "fallbacks": [
    "My ancient skull cannot decipher your mortal tongue.",
    "Speak clearly, mortal, or step away from the crypt."
  ],
  "low_power_lines": [
    "My power levels are low. I need to frighten more children to recharge."
  ]
}
```

`fallbacks`, `escalation_lines`, and `low_power_lines` are lists of lines the persona speaks; `low_power_lines` is spoken while the power supply is weak (see section 5.5).

---

### 7.2 Checking Existing Intents

#### Objective: Test whether a question is already matched by an existing intent rule.

```bash
./scripts/add_intent.py --check "Can I take two candies?"
```

Expected output:
```text
Pattern matched intent 'candy_take_one' in profile 'grimjaw':
Response: "Just one piece, mortal! Greedy hands get turned to bone!"
```

---

### 7.3 Adding New Custom Dialogue (`add_intent.py`)

#### Objective: Add a new spoken question, intent regex, and character response.

1. Add a single intent with pre-warmed audio cache:

```bash
./scripts/add_intent.py \
  -q "What is your favorite board game?" \
  -r "Chess! I play with skulls instead of pawns." \
  --prewarm
```

Expected output:
```text
Created intent 'favorite_board_game':
  Pattern: \b(favorite|play|like)\b.*\b(board\s*game|chess)\b
  Response: "Chess! I play with skulls instead of pawns."
  Pre-warmed audio cache: models/cache/4f8b9...npy
Profile 'data/personalities/grimjaw.json' updated successfully.
```

2. Run the interactive intent builder wizard:

```bash
./scripts/add_intent.py --interactive
```

---

### 7.4 Pre-Warming Dialogue Audio Cache (`prewarm_cache.py`)

#### Objective: Pre-render neural voice audio arrays for zero-latency response on Raspberry Pi.

On Raspberry Pi 3 hardware, synthesizing a new phrase on the CPU takes 4–6 seconds. Pre-warming pre-renders audio to disk cache (`models/cache/*.npy`) so playback starts in under 1 millisecond.

1. Pre-warm dialogue on bare metal:

```bash
# High-frequency dialogue only
./scripts/prewarm_cache.py

# All responses across all intents (~401 cached phrases)
./scripts/prewarm_cache.py --all
```

2. Pre-warm dialogue inside Docker container:

```bash
docker compose exec alsageist python3 scripts/prewarm_cache.py --all
```

3. Check the number of cached response arrays:

```bash
ls -1 models/cache/*.npy | wc -l
```

Expected output:
```text
401
```

---

## 8. Standalone Utilities

### 8.1 Direct Speech Testing (`scripts/say.py`)

#### Objective: Test local Piper neural voice output directly through the speaker.

| Goal | Command |
| :--- | :--- |
| Speak a line through the running service (fastest; reuses its loaded voice) | `python3 scripts/alsageist_ctl.py say "Testing."` |
| Preview a voice, personality, tempo, or volume without touching the service | `python3 scripts/say.py "Testing." --tempo 1.3` |
| Speak when the service is not running (for example on a development machine) | `python3 scripts/say.py "Testing."` |

```bash
python3 scripts/say.py "Halt! Who dares disturb the crypt?"
```

Specify custom volume:
```bash
python3 scripts/say.py "Testing volume." --volume 0.5
```

Show how long each phase took:
```bash
python3 scripts/say.py "Testing speed." --debug
```

Expected output when the phrase is already in the audio cache (Raspberry Pi 3):
```text
Grimjaw the Crypt Skeleton speaking (volume=0.4): "Testing speed."
Timing (--debug):
  python imports       1.61 s   51.0%  numpy, sounddevice, project modules
  load personality     0.66 s   20.8%
  resolve voice        0.00 s    0.0%
  init audio           0.00 s    0.0%
  load voice model     0.00 s    0.0%  skipped (phrase was cached)
  synthesize           0.01 s    0.2%  disk cache hit
  playback             0.88 s   27.9%  0.72 s of audio
  total                3.15 s
```

Expected output when the phrase is new (Raspberry Pi 3):
```text
  load voice model    35.31 s   59.3%  en_GB-alan-medium.onnx
  synthesize          12.93 s   21.7%  synthesized (cache miss)
```

| Phase | What it covers |
| :--- | :--- |
| `python imports` | Loading numpy, sounddevice, and the project modules |
| `load voice model` | Creating the ONNX Runtime session for the Piper voice. `say.py` loads it only when the phrase is not cached; the running service loads it once at startup |
| `synthesize` | Text to audio, or `disk cache hit` when the phrase is already in `models/cache` |
| `playback` | Time to play the clip, including its length |

> [!NOTE]
> Loading the voice took about 35 seconds on a Raspberry Pi 3 whose CPU was throttled to 600 MHz (see section 5.5). Phrases pre-warmed with `scripts/prewarm_cache.py` skip it.

---

### 8.2 Dynamic Personality Prompt Catalog (`scripts/summarize_personality.py`)

#### Objective: Inspect, search, or export the catalog of supported questions and answers for any character personality.

1. Display all supported dialogue intents and sample questions for Grimjaw:

```bash
python3 scripts/summarize_personality.py --personality grimjaw
```

2. Filter to candy bowl rules or costume questions:

```bash
python3 scripts/summarize_personality.py --personality grimjaw --category candy
```

3. Export a complete character markdown reference doc:

```bash
python3 scripts/summarize_personality.py --personality grimjaw -o docs/grimjaw_catalog.md
```

---

## 9. Related Guides

* **[Project & Hardware Overview](overview.md):** Complete hardware specifications, AIY Voice Kit pinouts, acoustic decoupling, and OS flashing.
* **[Getting Started with Docker on Raspberry Pi](getting_started_rpi_docker.md):** Automated containerized deployment, external volume layout, and container lifecycle.
* **[Getting Started on Bare Metal](getting_started_rpi_bare_metal.md):** Native Python virtual environment setup, ALSA full-duplex configuration, and systemd service installation.
* **[Design Specification](design_spec.md):** Technical design specification covering electrical interfaces, acoustic performance budgets, and operational architecture.
