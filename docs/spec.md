# Conversational Display Specification & Architecture

## 1. Project Requirements Summary

### Hardware Scope
* **Base Controller:** Raspberry Pi 3 Model B (Quad-Core 1.2 GHz, 1 GB RAM).
* **Audio System:** Google AIY Voice Kit V1 (2017 Full-Size Voice HAT):
  * **Driver Overlay:** `googlevoicehat-soundcard` via Raspberry Pi 40-pin GPIO.
  * **Input:** AIY stereo MEMS microphone daughterboard connected via ribbon cable.
  * **Output:** AIY 3-inch speaker wired directly to the Voice HAT amplified screw terminals.
  * **Controls & Indicators:** AIY illuminated arcade push-button with integrated status LED wired to the HAT button header.
* **Enclosure:** Hollow prop shell (e.g., life-sized resin skull) with internal acoustic decoupling.

### Conversational & Acoustic Scope
* **Character Profile:** **Grimjaw the Crypt Skeleton** — dry, gravelly rasp, hollow tomb presence, and sarcastic candy-guardian porch banter.
* **Voice Engine:** **Piper Neural TTS (`en_GB-alan-medium.onnx`)** running locally in RAM with zero disk I/O, zero API rate limits, and zero external network latency for audio generation.
* **Interaction Flow:** Hands-free continuous conversational streaming with Voice Activity Detection (VAD) and silence boundary detection, with fallback arcade button trigger and barge-in reset.
* **Status LED States:**
  * **Idle / Ready:** Rhythmic breathing pulse (0.5 Hz).
  * **Listening / Speech Detected:** Solid 100% illumination.
  * **Speaking / Audio Playback:** Dynamic audio-reactive flicker / pulse.
  * **Error / Disconnected:** Rapid 5 Hz alert blink.
* **Zero Disk I/O:** Audio capture, token transmission, and playback buffering operate entirely in RAM numpy/PCM arrays without writing temporary files to the SD card.
* **Headless Autostart:** Autonomous boot and recovery via systemd service.

---

## 2. System Architecture

```
                      ┌────────────────────────────────────────┐
                      │    Raspberry Pi 3B / Mac (RAM Buffer)  │
                      │                                        │
                      │  [Microphone Stream (16kHz PCM)]       │
                      │          │                             │
                      │          ▼                             │
                      │  [Core: Real-Time VAD]                 │
                      │  • Detects speech onset & silence gap  │
                      │          │                             │
                      │          ▼ (In-Memory WAV Buffer)      │
                      └──────────┬─────────────────────────────┘
                                 │
                   Internet / WiFi Connection
                                 │
                                 ▼
         ┌──────────────────────────────────────────────────┐
         │       High-Speed Cloud Inference Services        │
         │                                                  │
         │  1. Speech-to-Text:                              │
         │     • Groq Whisper Large v3 (Free: 7,200s/day)   │
         │     • Fallback: Gemini Flash STT in RAM          │
         │                                                  │
         │  2. Dialogue Reasoning:                          │
         │     • Groq Llama 3.3 70B (Free: 14,400 req/day)  │
         │     • Fallback: Gemini Flash Text                │
         │     • Prompt: Grimjaw Porce Guardian (<2 sent.)  │
         └───────────────────────┬──────────────────────────┘
                                 │
                                 ▼ (Creature Spoken Text)
                      ┌────────────────────────────────────────┐
                      │  [Services: Local Piper Neural TTS]    │
                      │  • Model: en_GB-alan-medium.onnx       │
                      │  • Synthesizes raw PCM in memory       │
                      │  • Zero API limits / 100% Free         │
                      │          │                             │
                      │          ▼ (22.05kHz PCM Audio Stream) │
                      │  [Core: AudioIO Speaker Playback]      │
                      │          │                             │
                      │  [Core: LED Controller]                │
                      │  • Dynamic PWM flicker during speech   │
                      └────────────────────────────────────────┘
```

---

## 3. Modular Codebase Layout

```
alsageist/
├── core/
│   ├── __init__.py
│   ├── audio_io.py          # Cross-platform in-memory capture & playback (CoreAudio / ALSA)
│   ├── led_controller.py    # Hardware RPi.GPIO PWM & terminal visualizer
│   └── vad.py               # Energy-based voice activity & silence boundary detector
├── services/
│   ├── __init__.py
│   ├── llm_service.py       # Grimjaw dialogue reasoning engine (Groq / Gemini)
│   ├── stt_service.py       # In-memory speech-to-text (Groq Whisper / Gemini)
│   └── tts_service.py       # Local Piper neural TTS engine in RAM
├── models/
│   └── piper/
│       ├── en_GB-alan-medium.onnx
│       └── en_GB-alan-medium.onnx.json
├── scripts/
│   ├── preview_voice.py     # Live audio synthesis preview tool
│   ├── audition_all_voices.py # Automated voice library audition tool
│   ├── generate_all_mockups.py # Batch mockup generator across engines
│   └── play_mockups.py      # Terminal playlist player
├── tests/
│   ├── test_vad.py
│   ├── test_led_controller.py
│   ├── test_tts_service.py
│   └── test_llm_service.py
├── main.py                  # Real-time conversational agent entry point
├── spec.md                  # Project specification & architecture
└── pyproject.toml           # Packaging and dependency declarations
```

---

## 4. End-to-End Raspberry Pi 3 Deployment Guide

### 1. Operating System Installation & Configuration
1. Flash **Raspberry Pi OS Lite (64-bit)** to a high-endurance microSD card.
2. Configure initial headless settings (WiFi credentials, hostname, SSH user) via Raspberry Pi Imager.
3. Boot the Raspberry Pi 3B and connect via SSH:
   ```bash
   ssh pi@<raspberry-pi-ip>
   ```
4. Enable the AIY Voice HAT V1 audio overlay by editing `/boot/firmware/config.txt`:
   ```ini
   # Disable default Broadcom audio
   dtparam=audio=off

   # Enable AIY Voice HAT V1 soundcard
   dtoverlay=googlevoicehat-soundcard
   ```
5. Reboot the device:
   ```bash
   sudo reboot
   ```
6. Verify audio hardware registration:
   * Playback DAC: `aplay -l` (card: `sndrpigooglevoi`)
   * Capture ADC: `arecord -l` (card: `sndrpigooglevoi`)

### 2. Audio Subsystem & System Dependencies
1. Install required system packages:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-venv portaudio19-dev libasound2-dev libatlas-base-dev alsa-utils
   ```

### 3. Application Environment & Model Setup
1. Clone the repository and navigate to `/home/pi/alsageist`:
   ```bash
   cd /home/pi/alsageist
   ```
2. Initialize the Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install sounddevice numpy piper-tts onnxruntime rpi-gpio pytest
   ```
3. Download the Piper Alan voice model:
   ```bash
   mkdir -p models/piper
   curl -L -o models/piper/en_GB-alan-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx
   curl -L -o models/piper/en_GB-alan-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json
   ```

### 4. Production Daemon (systemd)
1. Create `/etc/systemd/system/alsageist.service`:
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
   Environment="GEMINI_API_KEY=your_key_here"
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```
2. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now alsageist.service
   ```
