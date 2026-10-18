"""Shared helpers for building a real AlsaGeistAgent with mocked hardware subsystems."""

import argparse
import threading
import time
from unittest.mock import MagicMock

from core.led_controller import ConsoleLEDController
from services.agent import AlsaGeistAgent
from services.network_monitor import NetworkMonitor


def make_args(**overrides) -> argparse.Namespace:
    """Return a parsed-arguments namespace with test-safe defaults."""
    values = dict(
        engine="intent",
        personality="grimjaw",
        input_device=None,
        output_device=None,
        volume=0.5,
        mic_gain=1.0,
        silence_timeout=0.6,
        threshold=0.003,
        min_speech_sec=0.2,
        tempo=None,
        noise=None,
        model_path=None,
        no_greeting=True,
        llm_fallback=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def make_bare_agent(**overrides) -> AlsaGeistAgent:
    """Create a real AlsaGeistAgent without running its constructor.

    Subsystems are mocks (audio, vad, stt, tts, logger, basic_info) except the LED,
    which is the silent console controller. Keyword overrides replace any attribute.

    Args:
        **overrides: Attribute values to set on the agent after the defaults.

    Returns:
        An AlsaGeistAgent instance whose public API runs against mocked subsystems.
    """
    agent = AlsaGeistAgent.__new__(AlsaGeistAgent)
    agent.args = make_args()
    agent.running = False
    agent.is_disabled = False
    agent.start_time = time.time()
    agent.profile = None
    agent.llm_fallback_enabled = False
    agent.llm_fallback_service = None
    agent._state_lock = threading.RLock()
    agent._audio_lock = threading.RLock()
    agent._switch_lock = threading.Lock()
    agent._playback_generation = 0
    agent._mic_lock = threading.Lock()
    agent._user_muted = False
    agent._stopped = False
    agent._consecutive_filtered_turns = 0
    agent.network = NetworkMonitor(probe=lambda: agent._check_internet())
    agent.power_monitor = None
    agent.control_server = MagicMock()
    agent._last_activity_time = time.time()
    agent._idle_reset_done = False

    agent.audio = MagicMock()
    agent.audio.volume = 0.5
    agent.audio.sample_rate = 16000
    agent.audio.is_muted = False
    agent.audio.mute_capture.side_effect = lambda: setattr(agent.audio, "is_muted", True)
    agent.audio.unmute_capture.side_effect = lambda: setattr(agent.audio, "is_muted", False)
    agent.vad = MagicMock()
    agent.stt = MagicMock()
    agent.stt.last_cloud_failed = False
    agent.tts = MagicMock()
    agent.tts.synthesize.return_value = (b"\x00" * 3200, 16000)
    agent.logger = MagicMock()
    agent.basic_info = MagicMock()
    agent.basic_info.match_intent.return_value = None
    agent.basic_info.respond.return_value = None
    agent.dialogue_engine = MagicMock()
    agent.led = ConsoleLEDController(verbose=False)

    for name, value in overrides.items():
        setattr(agent, name, value)
    return agent
