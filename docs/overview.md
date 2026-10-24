# AlsaGeist Project & Hardware Overview

Autonomous voice-driven conversational animatronic display engine for Halloween porch installations and interactive props.

The name **AlsaGeist** combines **ALSA** (Advanced Linux Sound Architecture, the kernel-level audio subsystem driving low-latency microphone capture and playback on Linux) and **Geist** (German for *ghost* or *spirit*), characterizing an autonomous vocal spirit inhabiting animatronic display hardware.

---

## 1. Project Capabilities

AlsaGeist transforms a prop skull, skeleton, or animatronic figure into a real-time conversational character. The primary personality, **Grimjaw the Crypt Skeleton**, acts as an interactive candy bowl guardian with quick, snarky banter.

* **Hands-Free Conversational Loop:** Real-time Voice Activity Detection (VAD) continuously samples room audio in RAM, detects speech onset, and triggers conversational processing without push-to-talk requirements.
* **Deterministic Intent Matching:** Offline banter engine matching over 150 conversational intents (candy protection, costume recognition, classic pop culture, poetry, and skeleton puns) with sub-second response times.
* **Local Neural Speech Synthesis:** High-quality local voice synthesis using Piper TTS (`en_GB-alan-medium.onnx`). Runs fully offline on single-board computers with zero cloud API latency or ongoing costs.
* **Hybrid Cloud Fallback (Optional):** Optional streaming cloud Speech-to-Text (Groq Whisper Large v3 or Google Gemini Flash) and dynamic LLM fallback for prompts not covered by local rules.
* **Physical Hardware Controls:** Integrated arcade button support for software volume cycling (10% increments), 5-second mute/disable toggle, and 10-second hold for safe host poweroff.
* **Dynamic LED Signaling:** Pulse-width modulation (PWM) LED states indicating resting ambient idle, listening, speech capture acknowledgment, reasoning/synthesis, and spoken audio-reactive flicker.
* **Zero Disk I/O Runtime:** Audio capture, token transmission, and playback buffering operate entirely in RAM buffers to protect MicroSD cards from wear.

---

## 2. System Specifications & Board Comparison

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Single Board Computer** | Raspberry Pi 3 Model B (1 GB RAM) | Raspberry Pi 4 Model B (2 GB+ RAM) or Raspberry Pi 5 |
| **Architecture** | 64-bit ARM (`aarch64`) | 64-bit ARM (`aarch64`) |
| **Storage** | 16 GB MicroSD Card (Class 10 / A1) | 32 GB MicroSD Card (Class 10 / A2) |
| **Power Supply** | 5V 2.5A Micro-USB (Pi 3) | Official 5V 3.0A USB-C (Pi 4) / 5V 5.0A (Pi 5) |
| **Operating System** | Raspberry Pi OS Lite (64-bit, Bookworm) | Raspberry Pi OS Lite (64-bit, Bookworm) |
| **Audio Hardware** | Google AIY Voice HAT V1 | Google AIY Voice HAT V1 or USB Audio Class (UAC) Speakerphone |
| **Network** | Wi-Fi 802.11n or 10/100 Ethernet | Gigabit Ethernet or 5 GHz 802.11ac Wi-Fi |

### Board Performance Profiles

* **Raspberry Pi 3 Model B / B+:** Runs local intent matching and playback of pre-warmed audio cache instantly (<200 ms). Uncached neural TTS synthesis takes 4–6 seconds of CPU time. Pre-warming dialogue responses via `scripts/prewarm_cache.py` is essential for responsive real-time banter.
* **Throttled Raspberry Pi 3:** A board with an inadequate power supply runs its CPU at half clock speed (600 MHz instead of 1200 MHz) and is slower than the figures above. On a throttled Raspberry Pi 3 Model B a new sentence took 11 to 13 seconds to synthesize. See [Under-Voltage and CPU Throttling](usage_guide.md#56-under-voltage-and-cpu-throttling-raspberry-pi).
* **Raspberry Pi 4 Model B / Raspberry Pi 5:** Synthesizes uncached Piper neural TTS phrases on the fly in ~1.0–1.5 seconds, enabling dynamic LLM fallback banter with low latency.

---

## 3. Hardware Configuration Profiles

### Profile A: Primary Target, Raspberry Pi 3 + Google AIY Voice Kit V1 (2017)

The original 2017 Google AIY Voice Kit V1 provides a complete audio and physical control system in a single HAT:

* **Audio Interface:** Google AIY Voice HAT V1 attached to the 40-pin GPIO header. Uses the `googlevoicehat-soundcard` device tree overlay for I2S audio.
* **Microphone:** Stereo Knowles MEMS microphone daughterboard connected to the Voice HAT via the 5-pin ribbon cable.
* **Speaker:** 3-inch 4Ω 3W speaker wired directly to the amplified screw terminal blocks on the Voice HAT.
* **Push Button:** Illuminated arcade button wired to the Voice HAT button header.
* **Pin Allocation:**
  * **Status LED:** GPIO 25 (PWM-driven)
  * **Arcade Push Button:** GPIO 23 (Active-low with internal pull-up)

```
Raspberry Pi 3B Header
  └── Google AIY Voice HAT V1
        ├── Screw Terminals  ──> 3-inch 4Ω Speaker
        ├── 5-pin Ribbon     ──> Dual MEMS Microphone Daughterboard
        └── Button Harness   ──> Arcade Push Button (GPIO 23) + LED (GPIO 25)
```

### Profile B: Alternative Modern Hardware, Raspberry Pi 4 or 5 + USB Audio

For installations using modern Raspberry Pi boards without the legacy AIY Voice HAT:

* **Audio Interface:** Standard USB Audio Class (UAC) speakerphone (e.g., Anker PowerConf, Jabra Speak 410) or a USB sound card with a 3.5 mm microphone and powered speaker.
* **Status Indicator:** Standard 5 mm LED with a 330Ω current-limiting resistor connected between GPIO 25 (Pin 22) and GND (Pin 20).
* **Arcade Button (Optional):** Normally-open momentary pushbutton connected between GPIO 23 (Pin 16) and GND (Pin 14).

```
Raspberry Pi 4 / 5
  ├── USB Port           ──> USB Speakerphone or USB DAC (Mic + Speaker)
  ├── GPIO 25 (Pin 22)   ──> 330Ω Resistor ──> Status LED (+) ──> GND (Pin 20)
  └── GPIO 23 (Pin 16)   ──> Momentary Push Button ──> GND (Pin 14)
```

### Profile C: Google AIY Voice Kit V2 (Raspberry Pi Zero, Untested)

> [!IMPORTANT]
> **Check which AIY Voice Kit you have before following a guide.**
>
> | Kit | Board in the kit | Audio board | Status |
> | :--- | :--- | :--- | :--- |
> | **Voice Kit V1 (2017)** | Raspberry Pi 3 | Voice HAT | Supported (Profile A) |
> | **Voice Kit V2 (2018)** | Raspberry Pi Zero | Voice Bonnet | Untested; see the limitations below |

The sections below apply only to a Voice Kit V2 or any other Raspberry Pi Zero board. AlsaGeist has not been run on a Voice Kit V2.

| Board | CPU / OS | RAM | Status |
| :--- | :--- | :--- | :--- |
| **Raspberry Pi Zero / Zero W / Zero WH** | ARMv6, 32-bit only | 512 MB | Does not install (bare metal or Docker) |
| **Raspberry Pi Zero 2 W** | Cortex-A53, 64-bit capable | 512 MB | Untested; expected to install, memory is tight |

**Limitations:**

| Limitation | Detail |
| :--- | :--- |
| **No 32-bit packages** | `numpy`, `onnxruntime`, and `piper-tts` (pinned in `constraints.txt`) publish Linux wheels for `aarch64` and `x86_64` only. `pip install` fails on a 32-bit OS, including inside the Docker image. |
| **Voice Bonnet audio** | `scripts/setup_rpi_os.sh` and `scripts/setup_docker.sh` configure only the Voice HAT V1 overlay (`googlevoicehat-soundcard`). The Voice Bonnet is not configured. |
| **Memory** | The daemon uses 280–380 MB of resident memory on a Raspberry Pi 3 Model B. A 512 MB board leaves little for the operating system, and Docker adds its own overhead. |
| **Speed** | The Zero 2 W clocks at 1.0 GHz against 1.2 GHz on the Raspberry Pi 3 Model B. Uncached Piper synthesis takes longer than the 4-6 seconds documented for the Pi 3. |

**Suggestions:**

* **Move to a supported board.** Use a Raspberry Pi 3 Model B, 3B+, 4, or 5 with a USB speakerphone (Profile B), or with a Voice HAT V1 if you have one.
* **Keep the Zero board with USB audio.** Use a Zero 2 W with Raspberry Pi OS Lite (64-bit) and a USB speakerphone (Profile B wiring). This combination is untested.
* **Reduce memory on a Zero 2 W:**
  * Run bare metal rather than Docker.
  * Set `GROQ_API_KEY` or `GEMINI_API_KEY` in `.env` so cloud speech-to-text replaces the local Vosk model.
  * Add swap.
  * Run `scripts/prewarm_cache.py` before deployment so common replies play from the cache.

---

## 4. Acoustic Decoupling & Prop Installation

When mounting components inside a prop skull or animatronic figure:

1. **Microphone Isolation:** Mechanically decouple the microphone daughterboard from the speaker enclosure using soft foam or rubber standoffs. Direct mechanical coupling transmits speaker vibrations into the microphone, causing feedback loops and false speech triggers.
2. **Port Openings:** Route the microphone daughterboard ports toward the exterior of the prop (e.g. nasal cavity or under the jaw) with an unobstructed sound path.
3. **Speaker Enclosure:** Mount the speaker with a sealed rear cavity to maximize vocal clarity and volume from the 3W amplifier.
4. **Thermal Dissipation:** Ensure the prop shell has passive ventilation holes above the Raspberry Pi SoC to prevent thermal throttling during extended operation.

---

## 5. Operating System Preparation

AlsaGeist runs on **Raspberry Pi OS Lite (64-bit)** based on Debian 12 (Bookworm). The Lite image eliminates desktop overhead and maximizes CPU and RAM availability for audio processing.

### Objective: Flash Raspberry Pi OS Lite (64-bit) to a MicroSD card.

1. Insert the MicroSD card into your workstation.
2. Open **Raspberry Pi Imager**.
3. Choose OS: **Raspberry Pi OS (other)** $\rightarrow$ **Raspberry Pi OS Lite (64-bit)**.
4. Choose Storage: Select your MicroSD card.
5. In OS Customization settings:
   * **Hostname:** Enter your preferred hostname (e.g., `alsageist`).
   * **Username and Password:** Set a Linux username and password (e.g., `alsageist`).
   * **Wireless LAN:** Enter your Wi-Fi SSID and password.
   * **Services:** Enable SSH with password authentication or authorized public keys.
6. Select **Write** and wait for verification to complete.
7. Insert the MicroSD card into the Raspberry Pi and connect power.

Verify network connectivity from your workstation:

```bash
ssh <username>@<hostname>.local
```

Expected output:
```text
Linux alsageist 6.6.x+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.6.x aarch64
alsageist@alsageist:~$
```

---

## 6. Deployment Guides

Choose the installation method suited to your environment:

| Method | Recommended For | Guide Link |
| :--- | :--- | :--- |
| **Docker (Containerized)** | Fast, clean deployment with isolated dependencies, automated soundcard configuration, and zero-risk updates. | [Getting Started with Docker on Raspberry Pi](getting_started_rpi_docker.md) |
| **Bare Metal (Native Venv)** | Direct Python virtual environment execution for development, manual dependency control, or systems without Docker. | [Getting Started on Bare Metal Raspberry Pi OS](getting_started_rpi_bare_metal.md) |

---

## 7. Additional Documentation

* **[User & Operations Guide](usage_guide.md):** Complete guide for day-to-day operation, socket CLI commands (`alsageist_ctl.py`), audio tuning, and creating custom personalities.
* **[Design Specification](design_spec.md):** Technical design specification covering electrical interfaces, acoustic performance budgets, and operational architecture.
* **[System Architecture](architecture.md):** Deep technical reference on concurrency models, state machines, and audio pipeline internals.
