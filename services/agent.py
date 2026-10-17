"""Autonomous conversational agent orchestrator.

Owns the subsystem lifecycle (audio, VAD, STT, dialogue, TTS, LED, control server)
and exposes a thread-safe control API used by the button handlers and the IPC
control server. Entry points (main.py) only parse arguments and call start().
"""

import argparse
import os
import random
import re
import threading
import time
from typing import Optional, Tuple

from core import system_utils
from core.audio_io import AudioIO
from core.control_server import ControlServer
from core.intent_builder import PersonalityProfile, load_personality, resolve_voice_settings
from core.led_controller import LEDState, create_led_controller
from core.logger import InteractionLogger
from core.power_monitor import PowerMonitor
from core.vad import VoiceActivityDetector
from services.basic_info_service import BasicInfoService
from services.intent_service import PersonaIntentEngine
from services.llm_service import ConversationalLLMService
from services.stt_service import SpeechToTextService
from services.tts_service import PiperTTSService

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDLE_RESET_SEC = 10.0  # silence after which the next visitor starts a fresh conversation
LOW_POWER_QUIET_SEC = 10.0  # no spoken low-power line within this time of a conversation
DEFAULT_LOW_POWER_LINE = "My power supply is too weak. Please check the power adapter."


class AlsaGeistAgent:
    """Autonomous conversational agent orchestrator."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize all subsystems.

        Args:
            args: Command line arguments.
        """
        self.args = args
        self.running = False
        self.is_disabled = False
        self.start_time = time.time()
        self._state_lock = threading.RLock()
        self._audio_lock = threading.RLock()
        self._switch_lock = threading.Lock()
        self._playback_generation = 0
        self._mic_lock = threading.Lock()  # guards only the short mute/unmute decisions, never playback
        self._last_activity_time = time.time()
        self._idle_reset_done = False
        self._mode_forced: Optional[bool] = None
        self._last_check_request = 0.0

        print(f"Initializing AlsaGeist subsystems (Dialogue Engine: {args.engine.upper()})...")

        personality = getattr(args, "personality", "grimjaw")
        self.profile: Optional[PersonalityProfile] = self._load_profile(personality)

        self.led = create_led_controller(
            verbose=True,
            led_pin=getattr(args, "led_pin", None),
            button_pin=getattr(args, "button_pin", None),
            hold_time_sec=getattr(args, "button_hold_sec", None),
            long_hold_time_sec=getattr(args, "button_shutdown_sec", None),
        )
        self.audio = self._build_audio()
        self.vad = self._build_vad()
        self.stt = self._build_stt()

        self.llm_fallback_enabled = getattr(args, "llm_fallback", False)
        self.dialogue_engine, self.llm_fallback_service = self._build_dialogue(self.profile, personality)
        self.tts = self._build_tts(self.profile)

        log_dir = os.environ.get("LOG_DIR", os.path.join(REPO_ROOT, "logs"))
        self.logger = InteractionLogger.from_env(log_dir=log_dir, persona_name=self.persona_name)
        self.basic_info = BasicInfoService(profile=self.profile)

        self.is_online = self._check_internet()
        self.last_net_check_time = time.time()
        self.net_check_interval = 15.0 * 60.0  # 15 minutes
        self._net_check_thread: Optional[threading.Thread] = None
        self._net_check_result: Optional[bool] = None

        self.led.set_button_callbacks(
            on_press=self._on_button_press,
            on_hold=self._on_button_hold,
            on_long_hold=self._on_button_long_hold,
            hold_time_sec=getattr(args, "button_hold_sec", 5.0),
            long_hold_time_sec=getattr(args, "button_shutdown_sec", 10.0),
        )

        self.control_server = ControlServer(agent=self)
        off_values = ("false", "0", "no", "off")
        monitor_enabled = os.environ.get("LOW_POWER_MONITOR", "true").strip().lower() not in off_values
        announce_enabled = os.environ.get("LOW_POWER_ANNOUNCE", "true").strip().lower() not in off_values
        self.power_monitor: Optional[PowerMonitor] = (
            PowerMonitor.from_env(announce=self._announce_low_power if announce_enabled else None)
            if monitor_enabled
            else None
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_profile(name_or_path: str, strict: bool = False) -> Optional[PersonalityProfile]:
        """Load a personality profile.

        Args:
            name_or_path: Personality name or JSON path.
            strict: If True, raise when the profile cannot be loaded.

        Returns:
            The loaded profile, or None when loading fails and strict is False.

        Raises:
            FileNotFoundError: If strict and the profile file does not exist.
        """
        try:
            return load_personality(name_or_path)
        except Exception as exc:
            if strict:
                raise
            print(f"Warning: Could not load personality '{name_or_path}': {exc}")
            return None

    def _build_audio(self) -> AudioIO:
        """Create the in-memory audio I/O manager."""
        return AudioIO(
            sample_rate=16000,
            channels=1,
            block_size=480,  # 30ms frames
            device_input=self.args.input_device,
            device_output=self.args.output_device,
            volume=self.args.volume,
            mic_gain=self.args.mic_gain,
        )

    def _build_vad(self) -> VoiceActivityDetector:
        """Create the voice activity detector (tuned for ambient noise rejection)."""
        return VoiceActivityDetector(
            sample_rate=16000,
            frame_duration_ms=30,
            energy_threshold_multiplier=2.2,
            min_speech_frames=3,
            silence_timeout_sec=self.args.silence_timeout,
            min_energy_threshold=self.args.threshold,
            min_active_speech_sec=self.args.min_speech_sec,
        )

    @staticmethod
    def _build_stt() -> SpeechToTextService:
        """Create the speech-to-text service, exiting with guidance when unconfigured."""
        try:
            return SpeechToTextService(sample_rate=16000)
        except ValueError as exc:
            raise SystemExit(
                f"\n[STT Configuration Error] {exc}\n"
                "Please configure GROQ_API_KEY or GEMINI_API_KEY in your .env file,\n"
                "or place a Vosk offline speech model in models/vosk-model-small-en-us-0.15\n"
            ) from exc

    def _build_dialogue(
        self,
        profile: Optional[PersonalityProfile],
        personality: str,
    ) -> Tuple[object, Optional[ConversationalLLMService]]:
        """Create the dialogue engine and optional LLM fallback for a persona.

        Args:
            profile: Loaded personality profile, or None.
            personality: Personality name or path, used when no profile is loaded.

        Returns:
            Tuple of (dialogue engine, LLM fallback service or None).
        """
        if self.args.engine != "intent":
            return self._build_llm(profile), None

        engine = PersonaIntentEngine(personality_name_or_path=profile or personality)
        fallback = None
        if self.llm_fallback_enabled:
            try:
                fallback = self._build_llm(getattr(engine, "profile", None) or profile)
                print("  [LLM Fallback]: Active for indeterminate prompts.")
            except Exception as exc:
                print(f"  [LLM Fallback Warning]: Could not initialize LLM service: {exc}. Generic fallback will be used.")
        return engine, fallback

    @staticmethod
    def _build_llm(profile: Optional[PersonalityProfile]) -> ConversationalLLMService:
        """Create an LLM dialogue service whose system prompt matches the persona.

        Args:
            profile: Personality profile supplying the prompt, or None for the default prompt.

        Returns:
            Configured ConversationalLLMService.
        """
        if profile:
            # In-character canned lines for when every backend fails (an explicit env override wins)
            fallback = None
            if profile.fallbacks and not os.environ.get("LLM_FALLBACK_RESPONSE"):
                lines = list(profile.fallbacks)
                fallback = lambda: random.choice(lines)  # noqa: E731
            try:
                return ConversationalLLMService(
                    system_prompt=profile.build_system_prompt(),
                    fallback_response=fallback,
                )
            except Exception:
                pass
        return ConversationalLLMService()

    def _tts_settings(self, profile: Optional[PersonalityProfile]) -> Tuple[str, float, float]:
        """Resolve the voice model path, tempo and noise for a persona.

        Command-line values take precedence over profile voice settings.

        Args:
            profile: Personality profile supplying voice and voice_settings, or None.

        Returns:
            Tuple of (model_path, tempo, noise).
        """
        tempo, noise = resolve_voice_settings(profile, self.args.tempo, self.args.noise)
        model_path = system_utils.resolve_piper_model_path(
            model_path=self.args.model_path or os.environ.get("MODEL_PATH"),
            voice_name=profile.voice if profile else None,
            repo_root=REPO_ROOT,
        )
        return model_path, tempo, noise

    def _build_tts(self, profile: Optional[PersonalityProfile]) -> PiperTTSService:
        """Create the Piper TTS service for a persona's voice and voice settings.

        Args:
            profile: Personality profile supplying voice and voice_settings, or None.

        Returns:
            Configured PiperTTSService.
        """
        model_path, tempo, noise = self._tts_settings(profile)
        return PiperTTSService(
            model_path=model_path,
            sample_rate=22050,
            length_scale=tempo,
            noise_scale=noise,
        )

    @staticmethod
    def _check_internet(timeout: float = 2.5) -> bool:
        """Check if external internet connectivity is available."""
        return system_utils.check_internet_connectivity(timeout=timeout)

    # ------------------------------------------------------------------
    # Persona accessors
    # ------------------------------------------------------------------

    @property
    def active_profile(self) -> Optional[PersonalityProfile]:
        """Return the profile of the active dialogue engine, else the loaded profile."""
        for candidate in (getattr(self.dialogue_engine, "profile", None), getattr(self, "profile", None)):
            if isinstance(candidate, PersonalityProfile):
                return candidate
        return None

    @property
    def persona_name(self) -> str:
        """Return the active persona's display name."""
        profile = self.active_profile
        if profile:
            return profile.display_name
        return getattr(self.args, "personality", "AlsaGeist")

    def _announcement(self, key: str, default: str) -> str:
        """Return a system announcement for the active persona."""
        profile = self.active_profile
        return profile.announcement(key, default) if profile else default

    def _announce_low_power(self, status) -> bool:
        """Say an in-character line about the weak power supply, if nobody is talking to the agent.

        Called by the power monitor while the supply is under-voltage. It declines (returns
        False, so the monitor tries again at its next check) during or just after a conversation
        and while the agent is asleep.

        Args:
            status: The PowerStatus that triggered the call.

        Returns:
            True if the line was spoken.
        """
        if self.is_disabled or time.time() - self._last_activity_time < LOW_POWER_QUIET_SEC:
            return False
        profile = self.active_profile
        lines = list(profile.low_power_lines) if profile and profile.low_power_lines else [DEFAULT_LOW_POWER_LINE]
        line = random.choice(lines)
        print(f"{self.persona_name} (low power): \"{line}\"")
        return self.speak(line, blocking=True)

    # ------------------------------------------------------------------
    # Control API (thread-safe; used by button handlers and the IPC server)
    # ------------------------------------------------------------------

    @property
    def volume(self) -> float:
        """Return the playback volume (0.0 to 1.0)."""
        return self.audio.volume

    def set_volume(self, volume: float) -> float:
        """Set the playback volume.

        Args:
            volume: Requested volume; values outside 0.0 to 1.0 are clamped.

        Returns:
            The volume that was applied.
        """
        applied = round(min(1.0, max(0.0, float(volume))), 2)
        self.audio.volume = applied
        return applied

    @property
    def mic_muted(self) -> bool:
        """Return True while microphone capture is muted."""
        return self.audio.is_muted

    @property
    def led_state_name(self) -> str:
        """Return the current LED state name, or UNKNOWN."""
        state = getattr(self.led, "current_state", None)
        return getattr(state, "name", "UNKNOWN") if state is not None else "UNKNOWN"

    def set_mic_muted(self, muted: bool) -> None:
        """Mute or unmute microphone capture."""
        if muted:
            self.audio.mute_capture()
        else:
            self.audio.unmute_capture()

    @property
    def mode_forced(self) -> bool:
        """Return True while the online/offline mode is fixed by a control command."""
        return self._mode_forced is not None

    def set_online(self, online: Optional[bool]) -> None:
        """Force the online or offline operating mode, or return to automatic detection.

        A forced mode stays until set_online(None); the periodic connectivity check
        does not change it.

        Args:
            online: True or False to force a mode, None for automatic detection.
        """
        with self._state_lock:
            self._mode_forced = None if online is None else bool(online)
            if online is None:
                self.last_net_check_time = 0.0  # probe on the next loop pass
            else:
                self.is_online = bool(online)

    def disable(self) -> None:
        """Enter sleep mode: stop playback, mute the microphone, turn the LED off."""
        with self._state_lock:
            self.is_disabled = True
            self.audio.stop_playback()
            self.audio.mute_capture()
            self.reset_dialogue()
            self.vad.reset()
            self.led.set_state(LEDState.OFF)

    def enable(self) -> None:
        """Leave sleep mode: unmute the microphone and resume the idle LED."""
        with self._state_lock:
            self.is_disabled = False
            self.audio.unmute_capture()
            self.vad.reset()
            self.led.set_state(LEDState.IDLE)

    def toggle(self) -> bool:
        """Toggle between sleep and active mode.

        Returns:
            True if the agent is disabled after the toggle.
        """
        with self._state_lock:
            if self.is_disabled:
                self.enable()
            else:
                self.disable()
            return self.is_disabled

    def reset_dialogue(self) -> None:
        """Reset conversational context for the next visitor."""
        if hasattr(self.dialogue_engine, "reset_conversation"):
            self.dialogue_engine.reset_conversation()
        elif hasattr(self.dialogue_engine, "reset_history"):
            self.dialogue_engine.reset_history()
        if getattr(self, "llm_fallback_service", None) is not None:
            self.llm_fallback_service.reset_history()

    def switch_personality(self, name_or_path: str) -> PersonalityProfile:
        """Replace the active persona: profile, dialogue engine, LLM prompt, voice and info service.

        Every replacement component is built before any is installed, so a failure
        leaves the running persona unchanged. Building (loading a voice model can take
        tens of seconds on a Raspberry Pi) happens outside the state lock, which is held
        only for the swap, so button and control commands stay responsive. The current
        voice is reused when the new persona resolves to the same model and settings.

        Args:
            name_or_path: Personality name or JSON path.

        Returns:
            The newly active profile.

        Raises:
            FileNotFoundError: If the personality profile cannot be found.
        """
        with self._switch_lock:  # one switch at a time
            profile = self._load_profile(name_or_path, strict=True)
            engine, fallback = self._build_dialogue(profile, name_or_path)
            tts = self._reuse_or_build_tts(profile)
            info = BasicInfoService(profile=profile)

            with self._state_lock:
                self.profile = profile
                self.dialogue_engine = engine
                self.llm_fallback_service = fallback
                self.tts = tts
                self.basic_info = info
                self.logger.persona_name = profile.display_name
            return profile

    def _reuse_or_build_tts(self, profile: PersonalityProfile) -> PiperTTSService:
        """Return the running TTS when the persona needs the same voice, else build a new one."""
        model_path, tempo, noise = self._tts_settings(profile)
        current = self.tts
        if (
            getattr(current, "model_path", None) == model_path
            and getattr(current, "length_scale", None) == tempo
            and getattr(current, "noise_scale", None) == noise
        ):
            return current
        return self._build_tts(profile)

    def request_shutdown(self) -> None:
        """Announce power-off and shut down the host, without blocking the caller."""
        threading.Thread(target=self._on_button_long_hold, daemon=True).start()

    # ------------------------------------------------------------------
    # Speech output
    # ------------------------------------------------------------------

    def _play_speech_audio(
        self,
        audio_data,
        sample_rate: int,
        blocking: bool = True,
        cooldown_sec: float = 0.25,
    ) -> None:
        """Play synthesized audio while muting capture and managing LED state.

        Only one caller plays at a time. The trailing cooldown runs outside the audio
        lock so other callers are not held up by it. The caller that started the most
        recent playback is the one that unmutes the microphone, so a clip that starts
        during another's cooldown is never cut off by an early unmute. The mute and
        unmute decisions share a short lock so they cannot interleave.

        Args:
            audio_data: Synthesized audio PCM or numpy buffer.
            sample_rate: Sample rate in Hz.
            blocking: Whether audio playback should block execution.
            cooldown_sec: Trailing pause in seconds to allow room resonance to dissipate.
        """
        if self.is_disabled:
            return
        generation = -1
        try:
            with self._audio_lock:
                with self._mic_lock:
                    self._playback_generation += 1
                    generation = self._playback_generation
                    self.led.set_state(LEDState.SPEAKING)
                    self.audio.mute_capture()
                self.audio.play_audio(audio_data, sample_rate=sample_rate, blocking=blocking)
        finally:
            if generation >= 0:
                if cooldown_sec > 0:
                    time.sleep(cooldown_sec)
                with self._mic_lock:
                    if generation == self._playback_generation and not self.is_disabled:
                        self.audio.unmute_capture()
                        self.vad.reset()
                        self.led.set_state(LEDState.IDLE)

    def speak(
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
            True if synthesis and playback succeeded, False otherwise (including when disabled).
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
        if online:
            msg = self._announcement("online_mode", "NOW IN ONLINE MODE")
        else:
            msg = self._announcement("offline_mode", "NOW IN OFFLINE MODE")
        print(f"\n{self.persona_name} (network): \"{msg}\"")
        self.speak(msg, blocking=True, cooldown_sec=0.0)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_button_press(self) -> None:
        """Handle arcade button press: cycle volume by 10% and announce new level."""
        if self.is_disabled:
            return

        current_vol = round(self.audio.volume, 2)
        if current_vol >= 0.99:
            new_vol = 0.0
        else:
            new_vol = min(1.0, round(current_vol + 0.10, 2))

        self.set_volume(new_vol)
        pct = int(round(new_vol * 100))
        print(f"\n[BUTTON] Volume button pressed: {pct}%")

        # Stop ongoing playback, interrupt speech, and state new volume
        self.audio.stop_playback()
        self.reset_dialogue()
        self.vad.reset()

        self.speak(f"Volume {pct} percent.", blocking=True, cooldown_sec=0.0)

    def _on_button_hold(self) -> None:
        """Handle 5-second arcade button hold: toggle mute/disabled state."""
        if not self.is_disabled:
            print(f"\n[BUTTON] 5-second hold detected: Disabling and muting {self.persona_name} (Mic & Speaker OFF)...")
        else:
            print("\n[BUTTON] 5-second hold detected: Re-enabling and unmuting display (Normal operation)...")
        self.toggle()

    def _on_button_long_hold(self) -> None:
        """Handle 10-second arcade button hold: announce Powering Off and shutdown the pi."""
        print("\n[BUTTON] 10-second hold detected: Announcing shutdown and powering off...")
        self.audio.stop_playback()
        self.reset_dialogue()
        self.vad.reset()

        self.is_disabled = True
        self.audio.mute_capture()
        if self.audio.volume <= 0.05:
            self.audio.volume = 0.5

        announcement = self._announcement("powering_off", "Powering off.")
        try:
            audio_data, sample_rate = self.tts.synthesize(announcement)
            with self._audio_lock:
                self.led.set_state(LEDState.SPEAKING)
                self.audio.play_audio(audio_data, sample_rate=sample_rate, blocking=True)
        except Exception as exc:
            print(f"Shutdown announcement error: {exc}")
        finally:
            self.led.set_state(LEDState.OFF)

        if not system_utils.trigger_host_shutdown():
            print("Shutdown failed: no poweroff method succeeded. Returning to normal operation.")
            self.enable()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the real-time hands-free conversational loop."""
        self.running = True
        self.led.start()
        self.audio.start_capture()
        self.led.set_state(LEDState.IDLE)
        if hasattr(self, "control_server"):
            self.control_server.start()
        if self.is_online:
            self.basic_info.prefetch_weather()
        self._note_activity()

        profile_title = "CONVERSATIONAL DISPLAY"
        voice_desc = "Local ONNX"
        profile = self.active_profile
        if profile:
            profile_title = profile.display_name.upper()
            voice_desc = f"{profile.voice} (Local ONNX)"
        elif getattr(self.args, "model_path", None):
            voice_desc = f"{os.path.basename(self.args.model_path)} (Local ONNX)"

        print("\n" + "=" * 70)
        print(f" ALSA-GEIST [{profile_title}] IS ACTIVE & LISTENING")
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

        # Start the power monitor last: its first check may speak, and that must come after the
        # greeting and outside the quiet period that follows startup
        self._note_activity()
        if getattr(self, "power_monitor", None) is not None:
            self.power_monitor.start()

        try:
            while self.running:
                if self.is_disabled:
                    time.sleep(0.05)
                    self._note_activity()
                    continue

                now = time.time()
                self._poll_network(now)
                self._reset_if_idle(now)

                frame = self.audio.read_frame(timeout=0.05)
                if frame is None:
                    continue

                is_speech, completed, utterance = self.vad.process_frame(frame)

                if self.vad.is_speaking:
                    self._note_activity()
                    if self.led.current_state != LEDState.LISTENING:
                        self.led.set_state(LEDState.LISTENING)
                elif self.led.current_state == LEDState.LISTENING and not completed:
                    self.led.set_state(LEDState.IDLE)

                if completed and utterance is not None:
                    self._note_activity()
                    self._handle_utterance(utterance)
                    self._note_activity()  # idle time counts from the end of the reply
                    if not self.is_disabled:
                        self.led.set_state(LEDState.IDLE)

        except KeyboardInterrupt:
            print("\nShutting down AlsaGeist...")
        finally:
            self.stop()

    def _note_activity(self) -> None:
        """Record visitor activity and re-arm the idle conversation reset."""
        self._last_activity_time = time.time()
        self._idle_reset_done = False

    def _reset_if_idle(self, now: float) -> None:
        """Reset the conversation once after IDLE_RESET_SEC without activity.

        Args:
            now: Current time in seconds since the epoch.
        """
        if not self._idle_reset_done and now - self._last_activity_time > IDLE_RESET_SEC:
            self.reset_dialogue()
            self._idle_reset_done = True

    def _check_network_if_cloud_stt_failed(self) -> None:
        """Ask for an immediate connectivity probe when every cloud STT backend just failed.

        The probe, not the error, decides whether to go offline: if only the cloud service
        is down while the network is up, the mode stays online.
        """
        if self.stt.last_cloud_failed is True:
            self._request_network_check()

    def _request_network_check(self, min_interval: float = 30.0) -> None:
        """Make the main loop probe connectivity now instead of waiting for the next interval.

        Does nothing when the mode is forced or the agent is already offline (the periodic
        check brings it back online), and at most one request per min_interval seconds.

        Args:
            min_interval: Minimum seconds between requests.
        """
        if self._mode_forced is not None or not self.is_online:
            return
        now = time.time()
        if now - self._last_check_request < min_interval:
            return
        self._last_check_request = now
        self.last_net_check_time = 0.0

    def _poll_network(self, now: float) -> None:
        """Switch between online and offline modes without blocking the audio loop.

        Every net_check_interval seconds a background thread probes connectivity.
        The loop applies the result on a later iteration and announces any change.

        Args:
            now: Current time in seconds since the epoch.
        """
        result, self._net_check_result = self._net_check_result, None
        if self._mode_forced is not None:
            return
        if result is not None and result != self.is_online:
            self.is_online = result
            self._announce_mode(online=result)

        checking = self._net_check_thread is not None and self._net_check_thread.is_alive()
        if not checking and now - self.last_net_check_time > self.net_check_interval:
            self.last_net_check_time = now
            self._net_check_thread = threading.Thread(
                target=self._probe_network, daemon=True, name="AlsaGeist-NetCheck"
            )
            self._net_check_thread.start()

    def _probe_network(self) -> None:
        """Probe connectivity and publish the result for the main loop."""
        self._net_check_result = self._check_internet()

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
        """Resolve persona reply and matched intent from dialogue engine.

        Args:
            user_text: Transcribed user prompt.

        Returns:
            Tuple of (response_text, matched_intent).
        """
        if hasattr(self.dialogue_engine, "classify_and_respond"):
            reply, matched_intent = self.dialogue_engine.classify_and_respond(user_text)

            # Dynamic basic info handlers: Time and Weather
            if matched_intent == "time_and_clock":
                reply = self.basic_info.get_time_response()
            elif matched_intent == "weather_and_temperature":
                reply = self.basic_info.get_weather_response(allow_network=self.is_online)
            # Escalate indeterminate prompts to custom LLM response if enabled
            elif (
                matched_intent == "fallback_unrecognized"
                and self.llm_fallback_enabled
                and self.llm_fallback_service is not None
                and self.is_online
            ):
                try:
                    print(f"{self.persona_name}: Indeterminate prompt, fetching custom LLM response...")
                    custom_reply = self.llm_fallback_service.generate_response(user_text)
                    if custom_reply and custom_reply.strip():
                        reply = custom_reply.strip()
                        matched_intent = "llm_fallback_custom"
                except Exception as llm_exc:
                    print(f"  (LLM fallback fetch failed: {llm_exc}; using generic fallback)")
            return reply, matched_intent
        else:
            return self.dialogue_engine.generate_response(user_text), "llm_generated"

    def _handle_utterance(self, utterance_pcm) -> None:
        """Process completed user utterance through STT -> Brain -> TTS -> Playback."""
        t_start = time.time()
        audio_dur = len(utterance_pcm) / self.audio.sample_rate
        audio_rms = self.vad.calculate_rms(utterance_pcm)
        print(f"\n[USER SPEAKS] Processing utterance in RAM ({audio_dur:.2f}s, RMS: {audio_rms:.4f})...")

        # Visual acknowledgement: Flash button LED 3 times rapidly immediately upon hearing speech
        self.led.flash(count=3)
        self.led.set_state(LEDState.THINKING)

        # 1. Speech-to-Text
        try:
            user_text = self.stt.transcribe(utterance_pcm, offline=not self.is_online)
            self._check_network_if_cloud_stt_failed()
        except Exception as exc:
            self._check_network_if_cloud_stt_failed()
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

        print(f"User: \"{user_text}\"  ({stt_ms}ms)")

        # 2. Dialogue Reasoning
        try:
            reply, matched_intent = self._resolve_dialogue_response(user_text)
        except Exception as exc:
            print(f"Dialogue Error: {exc}")
            self.led.set_state(LEDState.ERROR)
            time.sleep(1.0)
            return

        t_llm = time.time()
        llm_ms = int((t_llm - t_stt) * 1000)
        print(f"{self.persona_name} ({matched_intent}): \"{reply}\"  ({llm_ms}ms)")

        # 3. Local Piper TTS Synthesis in RAM
        try:
            audio_data, sample_rate = self.tts.synthesize(reply)
        except Exception as exc:
            print(f"TTS Error: {exc}")
            self.led.set_state(LEDState.ERROR)
            time.sleep(1.0)
            return

        t_tts = time.time()
        tts_ms = int((t_tts - t_llm) * 1000)
        total_latency_ms = int((t_tts - t_start) * 1000)
        print(f"[Total TTFB Latency: {total_latency_ms}ms] Playing response...")

        # 4. Log Interaction Record
        self.logger.log_interaction(
            transcribed_text=user_text,
            response_text=reply,
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
        profile = self.active_profile
        if profile and profile.startup_greeting:
            greeting_text = profile.startup_greeting
        else:
            greeting_text = f"I am {self.persona_name}. System online and ready."

        print(f"\n{self.persona_name} (startup): \"{greeting_text}\"")
        self.speak(greeting_text, blocking=True, cooldown_sec=0.0)

    def stop(self) -> None:
        """Clean up all audio streams, control server, and LED threads."""
        self.running = False
        if getattr(self, "power_monitor", None) is not None:
            self.power_monitor.stop()
        if hasattr(self, "control_server"):
            try:
                self.control_server.stop()
            except Exception:
                pass
        self.audio.stop_playback()
        self.audio.stop_capture()
        self.led.stop()
        print("AlsaGeist terminated cleanly.")
