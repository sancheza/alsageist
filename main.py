#!/usr/bin/env python3
"""AlsaGeist - Real-Time Conversational Halloween Display.

Autonomous voice-driven Halloween prop running in-memory streaming dialogue
with local Piper neural TTS on macOS and Raspberry Pi.

Usage:
    ./main.py
    ./main.py --input-device "Microphone" --output-device "Speaker"
    ./main.py --verbose
"""

import argparse
import os
import signal
import sys
import time

from core.audio_io import AudioIO
from core.led_controller import LEDState, create_led_controller
from core.logger import InteractionLogger
from core.vad import VoiceActivityDetector
from services.llm_service import ConversationalLLMService
from services.stt_service import SpeechToTextService
from services.tts_service import PiperTTSService

VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="AlsaGeist: Autonomous Conversational Halloween Display."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed diagnostic logging.",
    )
    parser.add_argument(
        "--engine",
        choices=["intent", "llm"],
        default="intent",
        help="Dialogue engine: 'intent' (fast offline banter engine) or 'llm' (neural LLM). (default: intent)",
    )
    parser.add_argument(
        "--model-path",
        default="models/piper/en_GB-alan-medium.onnx",
        help="Path to Piper ONNX model file (default: models/piper/en_GB-alan-medium.onnx).",
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="Optional input audio hardware device name or index.",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help="Optional output audio hardware device name or index.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.006,
        help="Minimum audio RMS energy threshold for speech detection (default: 0.006). Increase if false-triggering in loud rooms.",
    )
    parser.add_argument(
        "--min-speech-sec",
        type=float,
        default=0.35,
        help="Minimum sustained speech duration in seconds required to accept an utterance (default: 0.35s).",
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=1.1,
        help="Trailing silence in seconds required before speech is considered finished (default: 1.1s).",
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=1.15,
        help="Piper phoneme length scale / tempo (default: 1.15 for slower gravel).",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.85,
        help="Piper generator noise level (default: 0.85).",
    )
    return parser.parse_args()


class AlsaGeistAgent:
    """Autonomous conversational agent orchestrator."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize all subsystems.

        Args:
            args: Command line arguments.
        """
        self.args = args
        self.running = False

        print(f"Initializing AlsaGeist subsystems (Dialogue Engine: {args.engine.upper()})...")

        # 1. LED & Hardware Controller
        self.led = create_led_controller(verbose=True)

        # 2. In-memory Audio I/O
        self.audio = AudioIO(
            sample_rate=16000,
            channels=1,
            block_size=480,  # 30ms frames
            device_input=args.input_device,
            device_output=args.output_device,
        )

        # 3. Voice Activity Detector (Configured for robust ambient noise rejection)
        self.vad = VoiceActivityDetector(
            sample_rate=16000,
            frame_duration_ms=30,
            energy_threshold_multiplier=2.2,
            min_speech_frames=3,
            silence_timeout_sec=args.silence_timeout,
            min_energy_threshold=args.threshold,
            min_active_speech_sec=args.min_speech_sec,
        )

        # 4. Speech-to-Text
        self.stt = SpeechToTextService(sample_rate=16000)

        # 5. Dialogue Engine (Intent vs LLM)
        if args.engine == "intent":
            from services.intent_service import GrimjawIntentEngine
            self.dialogue_engine = GrimjawIntentEngine()
        else:
            self.dialogue_engine = ConversationalLLMService()

        # 6. Local Piper Neural TTS
        self.tts = PiperTTSService(
            model_path=args.model_path,
            sample_rate=22050,
            length_scale=args.tempo,
            noise_scale=args.noise,
        )

        # 7. Interaction Logger
        self.logger = InteractionLogger(log_dir="logs")

        self.led.set_button_callback(self._on_button_press)

    def _reset_dialogue(self) -> None:
        """Reset conversational context for the next trick-or-treater."""
        if hasattr(self.dialogue_engine, "reset_conversation"):
            self.dialogue_engine.reset_conversation()
        elif hasattr(self.dialogue_engine, "reset_history"):
            self.dialogue_engine.reset_history()

    def _on_button_press(self) -> None:
        """Handle arcade button press interrupt."""
        print("\n[BUTTON] Arcade button pressed: interrupting/triggering!")
        self.audio.stop_playback()
        self._reset_dialogue()
        self.vad.reset()

    def start(self) -> None:
        """Run the real-time hands-free conversational loop."""
        self.running = True
        self.led.start()
        self.audio.start_capture()
        self.led.set_state(LEDState.IDLE)
        last_activity_time = time.time()

        print("\n" + "=" * 70)
        print(" 💀 ALSA-GEIST CRYPT SKELETON IS ACTIVE & LISTENING")
        print(f" Engine: {self.args.engine.upper()} | Voice: Piper Alan (Local ONNX)")
        print(f" Interaction Logging: Active -> {self.logger.jsonl_path}")
        print(" Speak into your microphone to start a conversation.")
        print(" Press Ctrl+C to stop.")
        print("=" * 70 + "\n")

        try:
            while self.running:
                frame = self.audio.read_frame(timeout=0.05)
                if frame is None:
                    # Check for 15s idle reset
                    if time.time() - last_activity_time > 15.0:
                        self._reset_dialogue()
                    continue

                is_speech, completed, utterance = self.vad.process_frame(frame)

                if is_speech:
                    last_activity_time = time.time()
                    if self.led.current_state != LEDState.LISTENING:
                        self.led.set_state(LEDState.LISTENING)

                if completed and utterance is not None and len(utterance) > 6000:
                    last_activity_time = time.time()
                    self._handle_utterance(utterance)
                    self.led.set_state(LEDState.IDLE)

        except KeyboardInterrupt:
            print("\nShutting down AlsaGeist...")
        finally:
            self.stop()

    def _handle_utterance(self, utterance_pcm) -> None:
        """Process completed user utterance through STT -> Brain -> TTS -> Playback."""
        t_start = time.time()
        audio_dur = len(utterance_pcm) / self.audio.sample_rate
        audio_rms = self.vad.calculate_rms(utterance_pcm)
        print(f"\n🎤 [USER SPEAKS] Processing utterance in RAM ({audio_dur:.2f}s, RMS: {audio_rms:.4f})...")

        # 1. Speech-to-Text
        try:
            user_text = self.stt.transcribe(utterance_pcm)
        except Exception as exc:
            print(f"STT Error: {exc}")
            self.led.set_state(LEDState.ERROR)
            self.logger.log_interaction(
                transcribed_text="",
                response_text="",
                engine=self.args.engine,
                matched_intent="stt_error",
                audio_duration_sec=audio_dur,
                audio_rms=audio_rms,
                latencies_ms={"stt": int((time.time() - t_start) * 1000)},
                status="ERROR",
                notes=f"STT Exception: {exc}",
            )
            time.sleep(1.0)
            return

        t_stt = time.time()
        stt_ms = int((t_stt - t_start) * 1000)

        if not user_text or len(user_text.strip()) < 2:
            print("  (Silence / ambient noise filtered)")
            self.logger.log_interaction(
                transcribed_text=user_text or "",
                response_text="",
                engine=self.args.engine,
                matched_intent="empty_silence",
                audio_duration_sec=audio_dur,
                audio_rms=audio_rms,
                latencies_ms={"stt": stt_ms},
                status="FILTERED_SILENCE",
            )
            return

        clean_test = user_text.strip().lower().rstrip(".,!?")
        if clean_test in ["you", "thank you", "shh", "the", "a", "subtitles by", "bye", "uh", "um", ""] and audio_rms < 0.010:
            print(f"  (Filtered ambient background artifact: \"{user_text}\")")
            self.logger.log_interaction(
                transcribed_text=user_text,
                response_text="",
                engine=self.args.engine,
                matched_intent="artifact_filter",
                audio_duration_sec=audio_dur,
                audio_rms=audio_rms,
                latencies_ms={"stt": stt_ms},
                status="FILTERED_ARTIFACT",
            )
            return

        print(f"👤 User: \"{user_text}\"  ({stt_ms}ms)")

        # 2. Dialogue Reasoning
        matched_intent = "llm_generated"
        try:
            if hasattr(self.dialogue_engine, "classify_and_respond"):
                creature_reply, matched_intent = self.dialogue_engine.classify_and_respond(user_text)
            else:
                creature_reply = self.dialogue_engine.generate_response(user_text)
        except Exception as exc:
            print(f"Dialogue Error: {exc}")
            self.led.set_state(LEDState.ERROR)
            time.sleep(1.0)
            return

        t_llm = time.time()
        llm_ms = int((t_llm - t_stt) * 1000)
        print(f"💀 Grimjaw ({matched_intent}): \"{creature_reply}\"  ({llm_ms}ms)")

        # 3. Local Piper TTS Synthesis in RAM
        try:
            audio_data, sample_rate = self.tts.synthesize(creature_reply)
        except Exception as exc:
            print(f"TTS Error: {exc}")
            self.led.set_state(LEDState.ERROR)
            time.sleep(1.0)
            return

        t_tts = time.time()
        tts_ms = int((t_tts - t_llm) * 1000)
        total_latency_ms = int((t_tts - t_start) * 1000)
        print(f"⚡ [Total TTFB Latency: {total_latency_ms}ms] Playing response...")

        # 4. Log Interaction Record
        self.logger.log_interaction(
            transcribed_text=user_text,
            response_text=creature_reply,
            engine=self.args.engine,
            matched_intent=matched_intent,
            audio_duration_sec=audio_dur,
            audio_rms=audio_rms,
            latencies_ms={"stt": stt_ms, "dialogue": llm_ms, "tts": tts_ms, "total": total_latency_ms},
            status="PROCESSED",
        )

        # 5. Audio Playback to Speaker (Mute mic capture to prevent self-triggering)
        self.led.set_state(LEDState.SPEAKING)
        self.audio.mute_capture()
        try:
            self.audio.play_audio(audio_data, sample_rate=sample_rate, blocking=True)
        finally:
            # 6. Cooldown to allow room resonance to dissipate, then unmute and reset VAD
            time.sleep(0.25)
            self.audio.unmute_capture()
            self.vad.reset()

    def stop(self) -> None:
        """Clean up all audio streams and LED threads."""
        self.running = False
        self.audio.stop_playback()
        self.audio.stop_capture()
        self.led.stop()
        print("AlsaGeist terminated cleanly.")


def main() -> int:
    """Main application entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    agent = AlsaGeistAgent(args)

    def signal_handler(signum, frame):
        agent.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    agent.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
