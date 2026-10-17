"""Voice Activity Detection (VAD) module for real-time speech boundary detection.

Processes in-memory audio frames to identify when a user starts speaking,
tracks ongoing speech, and signals when speech has finished after a silence duration.
"""

from collections import deque
import math
import threading
from typing import Deque, List, Optional, Tuple
import numpy as np


class VoiceActivityDetector:
    """Real-time energy-based voice activity detector with noise floor tracking."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold_multiplier: float = 1.6,
        min_speech_frames: int = 2,
        silence_timeout_sec: float = 0.9,
        pre_speech_padding_sec: float = 0.35,
        min_energy_threshold: float = 0.0020,
        min_active_speech_sec: float = 0.15,
        max_utterance_sec: float = 8.0,
    ) -> None:
        """Initialize the Voice Activity Detector.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000).
            frame_duration_ms: Duration of each audio frame in milliseconds (default: 30).
            energy_threshold_multiplier: Multiplier above background noise floor for speech trigger.
            min_speech_frames: Consecutive speech frames required to trigger active speech onset.
            silence_timeout_sec: Seconds of silence after speech before triggering end-of-speech.
            pre_speech_padding_sec: Seconds of audio before speech onset to retain in buffer.
            min_energy_threshold: Minimum absolute RMS threshold to prevent false ambient triggers.
            min_active_speech_sec: Minimum total duration of active voice frames required to accept utterance.
            max_utterance_sec: Maximum duration in seconds of a single utterance before forcing closure.
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
        self.energy_threshold_multiplier = energy_threshold_multiplier
        self.min_speech_frames = min_speech_frames
        self.silence_timeout_frames = int((silence_timeout_sec * 1000.0) / frame_duration_ms)
        self.pre_speech_padding_frames = int((pre_speech_padding_sec * 1000.0) / frame_duration_ms)
        self.min_energy_threshold = min_energy_threshold
        self.min_active_speech_frames = int((min_active_speech_sec * 1000.0) / frame_duration_ms)
        self.max_utterance_sec = max_utterance_sec

        # Frames arrive on the main loop while button, control and announcement threads reset
        self._lock = threading.RLock()
        self.noise_floor: float = 0.002
        self.is_speaking: bool = False
        self._consecutive_speech_frames: int = 0
        self._consecutive_silence_frames: int = 0
        self._active_speech_frames_count: int = 0
        self._peak_speech_rms: float = 0.0
        self._pre_speech_buffer: Deque[np.ndarray] = deque(maxlen=self.pre_speech_padding_frames)
        self._speech_buffer: List[np.ndarray] = []

    def calculate_rms(self, frame: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) energy of an audio frame.

        Args:
            frame: Numpy array of audio samples (float32 normalized between -1.0 and 1.0, or int16).

        Returns:
            RMS energy float value.
        """
        if frame is None or len(frame) == 0:
            return 0.0

        if frame.dtype == np.int16:
            normalized = frame.astype(np.float32) / 32768.0
        else:
            normalized = frame.astype(np.float32)

        mean_square = np.mean(normalized ** 2)
        if not np.isfinite(mean_square) or mean_square <= 0.0:
            return 0.0
        return float(math.sqrt(mean_square))

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, bool, Optional[np.ndarray]]:
        """Process a single audio frame and update speech state.

        Args:
            frame: Audio frame samples (1D numpy array).

        Returns:
            Tuple of (is_speech_frame, utterance_completed, utterance_audio_or_none).
        """
        with self._lock:
            return self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> Tuple[bool, bool, Optional[np.ndarray]]:
        """Update speech state for one frame; the caller holds the lock."""
        rms = self.calculate_rms(frame)
        threshold = max(self.noise_floor * self.energy_threshold_multiplier, self.min_energy_threshold)
        is_speech = rms > threshold

        if not self.is_speaking:
            # Update adaptive noise floor only during true ambient silence (clamped to prevent creeping)
            if not is_speech:
                max_adapt_rms = max(0.04, self.min_energy_threshold * 4)
                if rms < max_adapt_rms:
                    self.noise_floor = 0.98 * self.noise_floor + 0.02 * max(rms, 0.0005)
                self._pre_speech_buffer.append(frame)
                self._consecutive_speech_frames = 0
            else:
                self._consecutive_speech_frames += 1
                self._pre_speech_buffer.append(frame)
                if self._consecutive_speech_frames >= self.min_speech_frames:
                    self.is_speaking = True
                    self._consecutive_silence_frames = 0
                    self._active_speech_frames_count = self._consecutive_speech_frames
                    self._peak_speech_rms = rms
                    self._speech_buffer = list(self._pre_speech_buffer)
            return is_speech, False, None

        # Ongoing speech tracking
        self._speech_buffer.append(frame)

        if is_speech:
            self._consecutive_silence_frames = 0
            self._active_speech_frames_count += 1
            if rms > self._peak_speech_rms:
                self._peak_speech_rms = rms
        else:
            self._consecutive_silence_frames += 1

        # The maximum length applies to speech frames as well, so continuous loud sound cannot
        # grow the buffer without bound.
        utterance_dur = (len(self._speech_buffer) * self.frame_duration_ms) / 1000.0
        hit_max_duration = utterance_dur >= self.max_utterance_sec
        if self._consecutive_silence_frames >= self.silence_timeout_frames or hit_max_duration:
            if hit_max_duration and self._consecutive_silence_frames < self.silence_timeout_frames:
                self.noise_floor = 0.8 * self.noise_floor + 0.2 * min(0.05, rms)
            completed, audio = self._finish_utterance()
            return is_speech, completed, audio

        return is_speech, False, None

    def _finish_utterance(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Close the current utterance, reset tracking state, and validate the audio.

        Returns:
            Tuple of (utterance_accepted, audio). Audio is None when the utterance is rejected.
        """
        active_frames = self._active_speech_frames_count
        peak_rms = self._peak_speech_rms

        self.is_speaking = False
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._active_speech_frames_count = 0
        self._peak_speech_rms = 0.0

        completed_audio = np.concatenate(self._speech_buffer) if self._speech_buffer else np.array([], dtype=np.float32)
        self._speech_buffer.clear()
        self._pre_speech_buffer.clear()

        # Reject if not enough active speech frames or peak energy is too weak
        if (
            len(completed_audio) < int(self.sample_rate * 0.25)
            or active_frames < self.min_active_speech_frames
            or peak_rms < (self.min_energy_threshold * 0.8)
        ):
            return False, None

        return True, completed_audio

    def reset(self) -> None:
        """Reset internal buffers and state."""
        with self._lock:
            self.is_speaking = False
            self._consecutive_speech_frames = 0
            self._consecutive_silence_frames = 0
            self._active_speech_frames_count = 0
            self._peak_speech_rms = 0.0
            self._pre_speech_buffer.clear()
            self._speech_buffer.clear()
