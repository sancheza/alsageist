"""Audio Input/Output manager for in-memory streaming capture and playback.

Provides cross-platform low-latency audio capture and playback using sounddevice,
streaming directly in RAM without writing temporary files to disk.
"""

import queue
import threading
from typing import Callable, Generator, Optional
import numpy as np
import sounddevice as sd


class AudioIO:
    """Manages audio capture and playback streams in memory."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 480,  # 30ms at 16kHz
        device_input: Optional[str] = None,
        device_output: Optional[str] = None,
    ) -> None:
        """Initialize the Audio I/O manager.

        Args:
            sample_rate: Audio sampling frequency in Hz (default: 16000).
            channels: Number of audio channels (1 for Mono).
            block_size: Number of samples per buffer frame (default: 480 for 30ms).
            device_input: Name or index of input hardware device.
            device_output: Name or index of output hardware device.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device_input = self._resolve_device(device_input, kind="input")
        self.device_output = self._resolve_device(device_output, kind="output")

        self._input_queue: queue.Queue = queue.Queue()
        self._input_stream: Optional[sd.InputStream] = None
        self._is_capturing: bool = False
        self._lock = threading.Lock()

    def _resolve_device(self, name_or_index: Optional[str], kind: str) -> Optional[int]:
        """Resolve a device name substring to a sounddevice device ID.

        Args:
            name_or_index: Partial device name or integer index string.
            kind: 'input' or 'output'.

        Returns:
            Resolved integer device ID or None for default.
        """
        if name_or_index is None:
            return None

        try:
            return int(name_or_index)
        except ValueError:
            pass

        devices = sd.query_devices()
        target = name_or_index.lower()
        for idx, dev in enumerate(devices):
            channels = dev["max_input_channels"] if kind == "input" else dev["max_output_channels"]
            if channels > 0 and target in dev["name"].lower():
                return idx
        return None

    def _input_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for sounddevice input stream."""
        if self._is_capturing:
            self._input_queue.put(indata.copy())

    def start_capture(self) -> None:
        """Start non-blocking continuous microphone capture."""
        with self._lock:
            if self._input_stream is not None:
                return

            self._is_capturing = True
            while not self._input_queue.empty():
                self._input_queue.get_nowait()

            self._input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.block_size,
                device=self.device_input,
                callback=self._input_callback,
            )
            self._input_stream.start()

    def stop_capture(self) -> None:
        """Stop microphone capture and release stream."""
        with self._lock:
            self._is_capturing = False
            if self._input_stream is not None:
                try:
                    self._input_stream.stop()
                    self._input_stream.close()
                finally:
                    self._input_stream = None

    def drain_input_queue(self) -> None:
        """Flush and discard all buffered input audio frames."""
        with self._lock:
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    break

    def read_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Read a single audio frame from the capture queue.

        Args:
            timeout: Max seconds to wait for a frame.

        Returns:
            1D numpy array of float32 samples, or None on timeout.
        """
        try:
            frame_2d = self._input_queue.get(timeout=timeout)
            return frame_2d.flatten()
        except queue.Empty:
            return None

    def play_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None,
        blocking: bool = True,
    ) -> None:
        """Play in-memory audio array to the output speaker.

        Args:
            audio_data: 1D or 2D numpy array containing audio samples (float32 or int16).
            sample_rate: Sample rate of the audio (defaults to instance sample_rate).
            blocking: If True, waits until playback finishes before returning.
        """
        sr = sample_rate or self.sample_rate
        if audio_data.dtype == np.int16:
            play_data = audio_data.astype(np.float32) / 32768.0
        else:
            play_data = audio_data.astype(np.float32)

        sd.play(play_data, samplerate=sr, device=self.device_output, blocking=blocking)
        if blocking:
            sd.wait()

    def stop_playback(self) -> None:
        """Instantly stop any active audio playback."""
        sd.stop()
