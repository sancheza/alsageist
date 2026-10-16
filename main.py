#!/usr/bin/env python3
"""AlsaGeist - Autonomous Voice-Driven Conversational Display Engine.

Autonomous voice-driven animatronic and display engine running in-memory streaming dialogue
with pluggable personality profiles, low-latency VAD, and local Piper neural TTS.

Usage:
    ./main.py
    ./main.py --personality snowman
    ./main.py --input-device "Microphone" --output-device "Speaker"
    ./main.py --verbose
"""

import argparse
import os
import re
import signal
import sys
import time
from typing import Optional, Tuple

from core import system_utils
from core.audio_io import AudioIO
from core.config import VERSION, parse_args
from core.led_controller import LEDState, create_led_controller
from core.logger import InteractionLogger
from core.system_utils import load_env_file
from core.vad import VoiceActivityDetector
from services.basic_info_service import BasicInfoService
from services.llm_service import ConversationalLLMService
from services.stt_service import SpeechToTextService
from services.tts_service import PiperTTSService


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

        # 0. Personality Profile
        from core.intent_builder import PersonalityProfile, find_personality_file
        personality = getattr(args, "personality", "grimjaw")
        self.profile: Optional[PersonalityProfile] = None
        try:
            p_file = find_personality_file(personality)
            if os.path.isfile(p_file):
                self.profile = PersonalityProfile.load_from_file(p_file)
        except Exception:
            pass

        # 1. LED & Hardware Controller
        self.led = create_led_controller(
            verbose=True,
            led_pin=getattr(args, "led_pin", None),
            button_pin=getattr(args, "button_pin", None),
            hold_time_sec=getattr(args, "button_hold_sec", None),
            long_hold_time_sec=getattr(args, "button_shutdown_sec", None),
        )

        # 2. In-memory Audio I/O
        self.audio = AudioIO(
            sample_rate=16000,
            channels=1,
            block_size=480,  # 30ms frames
            device_input=args.input_device,
            device_output=args.output_device,
            volume=args.volume,
            mic_gain=args.mic_gain,
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
        try:
            self.stt = SpeechToTextService(sample_rate=16000)
        except ValueError as exc:
            sys.stderr.write(
                f"\n[STT Configuration Error] {exc}\n"
                "Please configure GROQ_API_KEY or GEMINI_API_KEY in your .env file,\n"
                "or place a Vosk offline speech model in models/vosk-model-small-en-us-0.15\n\n"
            )
            sys.exit(1)

        # 5. Dialogue Engine (Intent vs LLM)
        self.llm_fallback_enabled = getattr(args, "llm_fallback", False)
        self.llm_fallback_service: Optional[ConversationalLLMService] = None

        if args.engine == "intent":
            from services.intent_service import PersonaIntentEngine
            self.dialogue_engine = PersonaIntentEngine(
                personality_name_or_path=self.profile or personality
            )
            if self.llm_fallback_enabled:
                try:
                    profile = getattr(self.dialogue_engine, "profile", self.profile)
                    if profile:
                        persona_prompt = (
                            f"You are {profile.display_name}. {profile.description}\n"
                            "Rules you must strictly obey:\n"
                            "1. Keep every response under 2 sentences maximum.\n"
                            "2. Be witty, dry, and stay in character.\n"
                            "3. Never output emojis, markdown formatting, asterisks, or sound effect tags.\n"
                            "4. Return plain spoken dialogue only."
                        )
                        self.llm_fallback_service = ConversationalLLMService(system_prompt=persona_prompt)
                    else:
                        self.llm_fallback_service = ConversationalLLMService()
                    print("  [LLM Fallback]: Active for indeterminate prompts.")
                except Exception as exc:
                    print(f"  [LLM Fallback Warning]: Could not initialize LLM service: {exc}. Generic fallback will be used.")
                    self.llm_fallback_service = None
        else:
            try:
                if self.profile:
                    persona_prompt = (
                        f"You are {self.profile.display_name}. {self.profile.description}\n"
                        "Rules you must strictly obey:\n"
                        "1. Keep every response under 2 sentences maximum.\n"
                        "2. Be witty, dry, and stay in character.\n"
                        "3. Never output emojis, markdown formatting, asterisks, or sound effect tags.\n"
                        "4. Return plain spoken dialogue only."
                    )
                    self.dialogue_engine = ConversationalLLMService(system_prompt=persona_prompt)
                else:
                    self.dialogue_engine = ConversationalLLMService()
            except Exception:
                self.dialogue_engine = ConversationalLLMService()

        repo_root = os.path.dirname(os.path.abspath(__file__))
        voice_id = self.profile.voice if self.profile else None
        resolved_model = system_utils.resolve_piper_model_path(
            model_path=args.model_path or os.environ.get("MODEL_PATH"),
            voice_name=voice_id,
            repo_root=repo_root,
        )

        tempo = args.tempo
        if tempo is None:
            tempo = (
                self.profile.voice_settings.get("tempo", 1.15)
                if self.profile and self.profile.voice_settings
                else 1.15
            )

        noise = args.noise
        if noise is None:
            noise = (
                self.profile.voice_settings.get("noise", 0.85)
                if self.profile and self.profile.voice_settings
                else 0.85
            )

        log_dir = os.environ.get("LOG_DIR", os.path.join(repo_root, "logs"))

        # 6. Local Piper Neural TTS
        self.tts = PiperTTSService(
            model_path=resolved_model,
            sample_rate=22050,
            length_scale=tempo,
            noise_scale=noise,
        )

        # 7. Interaction Logger
        self.logger = InteractionLogger(log_dir=log_dir)

        # 8. Basic Info Service (Live local time and weather)
        self.basic_info = BasicInfoService(profile=self.profile)

        # 9. Network status & Offline STT tracking
        self.is_online = self._check_internet()
        self.last_net_check_time = time.time()
        self.net_check_interval = 15.0 * 60.0  # 15 minutes
        self.is_disabled = False

        self.led.set_button_callbacks(
            on_press=self._on_button_press,
            on_hold=self._on_button_hold,
            on_long_hold=self._on_button_long_hold,
            hold_time_sec=getattr(args, "button_hold_sec", 5.0),
            long_hold_time_sec=getattr(args, "button_shutdown_sec", 10.0),
        )

        # 10. Control Server (Unix Domain Socket IPC)
        from core.control_server import ControlServer
        self.control_server = ControlServer(agent=self)
        self.start_time = time.time()


    @staticmethod
    def _check_internet(timeout: float = 2.5) -> bool:
        """Check if external internet connectivity is available."""
        return system_utils.check_internet_connectivity(timeout=timeout)

    def _play_speech_audio(
        self,
        audio_data,
        sample_rate: int,
        blocking: bool = True,
        cooldown_sec: float = 0.25,
    ) -> None:
        """Play synthesized audio while muting capture and managing LED state.

        Args:
            audio_data: Synthesized audio PCM or numpy buffer.
            sample_rate: Sample rate in Hz.
            blocking: Whether audio playback should block execution.
            cooldown_sec: Trailing pause in seconds to allow room resonance to dissipate.
        """
        if self.is_disabled:
            return
        self.led.set_state(LEDState.SPEAKING)
        self.audio.mute_capture()
        try:
            self.audio.play_audio(audio_data, sample_rate=sample_rate, blocking=blocking)
        finally:
            if cooldown_sec > 0:
                time.sleep(cooldown_sec)
            if not self.is_disabled:
                self.audio.unmute_capture()
                self.vad.reset()
                self.led.set_state(LEDState.IDLE)

    def _speak_response(
        self,
        text: str,
        blocking: bool = True,
        cooldown_sec: float = 0.25,
    ) -> bool:
        """Synthesize and play speech text.

        Args:
            text: Plain-text string to speak.
            blocking: Whether audio playback should block execution.
            cooldown_sec: Trailing pause in seconds.

        Returns:
            True if synthesis and playback succeeded, False otherwise.
        """
        if self.is_disabled:
            return False
        try:
            audio_data, sample_rate = self.tts.synthesize(text)
            self._play_speech_audio(audio_data, sample_rate, blocking=blocking, cooldown_sec=cooldown_sec)
            return True
        except Exception as exc:
            print(f"Speech playback error: {exc}")
            return False

    def _announce_mode(self, online: bool) -> None:
        """Announce current online or offline operational mode."""
        msg = "NOW IN ONLINE MODE" if online else "NOW IN OFFLINE MODE"
        print(f"\n💀 Grimjaw (network): \"{msg}\"")
        self._speak_response(msg, blocking=True, cooldown_sec=0.0)

    def _reset_dialogue(self) -> None:
        """Reset conversational context for the next trick-or-treater."""
        if hasattr(self.dialogue_engine, "reset_conversation"):
            self.dialogue_engine.reset_conversation()
        elif hasattr(self.dialogue_engine, "reset_history"):
            self.dialogue_engine.reset_history()
        if getattr(self, "llm_fallback_service", None) is not None:
            self.llm_fallback_service.reset_history()

    def _on_button_press(self) -> None:
        """Handle arcade button press: cycle volume by 10% and announce new level."""
        if self.is_disabled:
            return

        # 1. Calculate new volume level (step by 0.10, cycle 1.0 -> 0.0)
        current_vol = round(self.audio.volume, 2)
        if current_vol >= 0.99:
            new_vol = 0.0
        else:
            new_vol = min(1.0, round(current_vol + 0.10, 2))

        self.audio.volume = new_vol
        pct = int(round(new_vol * 100))
        print(f"\n[BUTTON] Volume button pressed: {pct}%")

        # 2. Stop ongoing playback, interrupt speech, and state new volume
        self.audio.stop_playback()
        self._reset_dialogue()
        self.vad.reset()

        announcement = f"Volume {pct} percent."
        self._speak_response(announcement, blocking=True, cooldown_sec=0.0)

    def _on_button_hold(self) -> None:
        """Handle 5-second arcade button hold: toggle mute/disabled state."""
        if not self.is_disabled:
            # 1. Disable / Mute Grimjaw (Mic and Speaker off, LED off)
            print("\n[BUTTON] 5-second hold detected: Disabling and muting Grimjaw (Mic & Speaker OFF)...")
            self.is_disabled = True
            self.audio.stop_playback()
            self.audio.mute_capture()
            self._reset_dialogue()
            self.vad.reset()
            self.led.set_state(LEDState.OFF)
        else:
            # 2. Re-enable / Unmute display (Mic and Speaker on, LED idle/on)
            print("\n[BUTTON] 5-second hold detected: Re-enabling and unmuting display (Normal operation)...")
            self.is_disabled = False
            self.audio.unmute_capture()
            self.vad.reset()
            self.led.set_state(LEDState.IDLE)

    def _on_button_long_hold(self) -> None:
        """Handle 10-second arcade button hold: announce Powering Off and shutdown the pi."""
        print("\n[BUTTON] 10-second hold detected: Announcing shutdown and powering off...")
        self.audio.stop_playback()
        self._reset_dialogue()
        self.vad.reset()

        self.is_disabled = True
        self.audio.mute_capture()
        if self.audio.volume <= 0.05:
            self.audio.volume = 0.5

        announcement = "Powering off."
        try:
            audio_data, sample_rate = self.tts.synthesize(announcement)
            self.led.set_state(LEDState.SPEAKING)
            self.audio.play_audio(audio_data, sample_rate=sample_rate, blocking=True)
        except Exception as exc:
            print(f"Shutdown announcement error: {exc}")
        finally:
            self.led.set_state(LEDState.OFF)

        system_utils.trigger_host_shutdown()

    def start(self) -> None:
        """Run the real-time hands-free conversational loop."""
        self.running = True
        self.led.start()
        self.audio.start_capture()
        self.led.set_state(LEDState.IDLE)
        if hasattr(self, "control_server"):
            self.control_server.start()
        last_activity_time = time.time()

        profile_title = "CONVERSATIONAL DISPLAY"
        voice_desc = "Local ONNX"
        if hasattr(self.dialogue_engine, "profile") and self.dialogue_engine.profile:
            profile_title = self.dialogue_engine.profile.display_name.upper()
            voice_desc = f"{self.dialogue_engine.profile.voice} (Local ONNX)"
        elif hasattr(self.args, "model_path") and self.args.model_path:
            voice_desc = f"{os.path.basename(self.args.model_path)} (Local ONNX)"

        print("\n" + "=" * 70)
        print(f" 🎙️  ALSA-GEIST [{profile_title}] IS ACTIVE & LISTENING")
        print(f" Engine: {self.args.engine.upper()} | Voice: {voice_desc}")
        print(f" Interaction Logging: Active -> {self.logger.jsonl_path}")
        print(" Speak into your microphone to start a conversation.")
        print(" Press Ctrl+C to stop.")
        print("=" * 70 + "\n")

        # Spoken announcements upon startup
        if not getattr(self.args, "no_greeting", False):
            self._play_startup_greeting()
            if not self.is_online:
                self._announce_mode(online=False)

        try:
            while self.running:
                if self.is_disabled:
                    time.sleep(0.05)
                    last_activity_time = time.time()
                    continue

                # Check internet access every 15 minutes (or switch modes dynamically)
                now = time.time()
                if now - self.last_net_check_time > self.net_check_interval:
                    self.last_net_check_time = now
                    currently_online = self._check_internet()
                    if currently_online != self.is_online:
                        self.is_online = currently_online
                        self._announce_mode(online=self.is_online)

                frame = self.audio.read_frame(timeout=0.05)
                if frame is None:
                    # Check for 10s idle reset between visitors
                    if now - last_activity_time > 10.0:
                        self._reset_dialogue()
                    continue

                is_speech, completed, utterance = self.vad.process_frame(frame)

                if self.vad.is_speaking:
                    last_activity_time = time.time()
                    if self.led.current_state != LEDState.LISTENING:
                        self.led.set_state(LEDState.LISTENING)
                elif self.led.current_state == LEDState.LISTENING and not completed:
                    self.led.set_state(LEDState.IDLE)

                if completed and utterance is not None:
                    last_activity_time = time.time()
                    self._handle_utterance(utterance)
                    if not self.is_disabled:
                        self.led.set_state(LEDState.IDLE)

        except KeyboardInterrupt:
            print("\nShutting down AlsaGeist...")
        finally:
            self.stop()

    def _filter_utterance(
        self, user_text: Optional[str], audio_rms: float
    ) -> Tuple[bool, str, str]:
        """Check if transcribed user text should be filtered as silence or ambient artifact.

        Args:
            user_text: Transcribed string from STT.
            audio_rms: Root-mean-square energy of the audio segment.

        Returns:
            Tuple of (should_filter, matched_intent, filter_status).
        """
        if not user_text or len(user_text.strip()) < 2:
            return True, "empty_silence", "FILTERED_SILENCE"

        clean_test = user_text.strip().lower().rstrip(".,!?")
        is_timecode = bool(re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", clean_test))
        if (
            (clean_test in ["you", "thank you", "shh", "the", "a", "subtitles by", "bye", "uh", "um", ""] and audio_rms < 0.010)
            or is_timecode
        ):
            return True, "artifact_filter", "FILTERED_ARTIFACT"

        return False, "", ""

    def _resolve_dialogue_response(self, user_text: str) -> Tuple[str, str]:
        """Resolve creature reply and matched intent from dialogue engine.

        Args:
            user_text: Transcribed user prompt.

        Returns:
            Tuple of (response_text, matched_intent).
        """
        if hasattr(self.dialogue_engine, "classify_and_respond"):
            creature_reply, matched_intent = self.dialogue_engine.classify_and_respond(user_text)

            # Dynamic basic info handlers: Time and Weather
            if matched_intent == "time_and_clock":
                creature_reply = self.basic_info.get_time_response()
            elif matched_intent == "weather_and_temperature":
                creature_reply = self.basic_info.get_weather_response()
            # Escalate indeterminate prompts to custom LLM response if enabled
            elif (
                matched_intent == "fallback_unrecognized"
                and self.llm_fallback_enabled
                and self.llm_fallback_service is not None
                and self.is_online
            ):
                try:
                    profile = getattr(self, "profile", None)
                    persona_name = profile.display_name if profile else "Grimjaw"
                    print(f"💀 {persona_name}: Indeterminate prompt — fetching custom LLM response...")
                    custom_reply = self.llm_fallback_service.generate_response(user_text)
                    if custom_reply and custom_reply.strip():
                        creature_reply = custom_reply.strip()
                        matched_intent = "llm_fallback_custom"
                except Exception as llm_exc:
                    print(f"  (LLM fallback fetch failed: {llm_exc}; using generic fallback)")
            return creature_reply, matched_intent
        else:
            return self.dialogue_engine.generate_response(user_text), "llm_generated"

    def _handle_utterance(self, utterance_pcm) -> None:
        """Process completed user utterance through STT -> Brain -> TTS -> Playback."""
        t_start = time.time()
        audio_dur = len(utterance_pcm) / self.audio.sample_rate
        audio_rms = self.vad.calculate_rms(utterance_pcm)
        print(f"\n🎤 [USER SPEAKS] Processing utterance in RAM ({audio_dur:.2f}s, RMS: {audio_rms:.4f})...")

        # Visual acknowledgement: Flash button LED 3 times rapidly immediately upon hearing speech
        self.led.flash(count=3)
        self.led.set_state(LEDState.THINKING)

        # 1. Speech-to-Text
        try:
            user_text = self.stt.transcribe(utterance_pcm, offline=not self.is_online)
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

        should_filter, matched_intent, filter_status = self._filter_utterance(user_text, audio_rms)
        if should_filter:
            if filter_status == "FILTERED_SILENCE":
                print("  (Silence / ambient noise filtered)")
            else:
                print(f"  (Filtered ambient background artifact / timecode: \"{user_text}\")")
            self.led.set_state(LEDState.IDLE)
            self.logger.log_interaction(
                transcribed_text=user_text or "",
                response_text="",
                engine=self.args.engine,
                matched_intent=matched_intent,
                audio_duration_sec=audio_dur,
                audio_rms=audio_rms,
                latencies_ms={"stt": stt_ms},
                status=filter_status,
            )
            return

        print(f"👤 User: \"{user_text}\"  ({stt_ms}ms)")

        # 2. Dialogue Reasoning
        try:
            creature_reply, matched_intent = self._resolve_dialogue_response(user_text)
        except Exception as exc:
            print(f"Dialogue Error: {exc}")
            self.led.set_state(LEDState.ERROR)
            time.sleep(1.0)
            return

        t_llm = time.time()
        llm_ms = int((t_llm - t_stt) * 1000)
        profile = getattr(self, "profile", None)
        persona_name = profile.display_name if profile else "Grimjaw"
        print(f"💀 {persona_name} ({matched_intent}): \"{creature_reply}\"  ({llm_ms}ms)")

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
        self._play_speech_audio(audio_data, sample_rate, blocking=True, cooldown_sec=0.25)

    def _play_startup_greeting(self) -> None:
        """Play an initial greeting announcement when the agent first wakes up."""
        profile = getattr(self.dialogue_engine, "profile", None)
        if profile and getattr(profile, "startup_greeting", None):
            greeting_text = profile.startup_greeting
            persona_name = profile.display_name
        elif profile:
            greeting_text = f"I am {profile.display_name}. System online and ready."
            persona_name = profile.display_name
        else:
            persona_name = getattr(self.args, "personality", "AlsaGeist")
            greeting_text = f"I am {persona_name}. System online and ready."

        print(f"\n💀 {persona_name} (startup): \"{greeting_text}\"")
        self._speak_response(greeting_text, blocking=True, cooldown_sec=0.0)

    def stop(self) -> None:
        """Clean up all audio streams, control server, and LED threads."""
        self.running = False
        if hasattr(self, "control_server"):
            try:
                self.control_server.stop()
            except Exception:
                pass
        self.audio.stop_playback()
        self.audio.stop_capture()
        self.led.stop()
        print("AlsaGeist terminated cleanly.")


def main() -> int:
    """Main application entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    load_env_file()
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
