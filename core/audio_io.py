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
        volume: float = 1.0,
        mic_gain: float = 1.0,
    ) -> None:
        """Initialize the Audio I/O manager.

        Args:
            sample_rate: Audio sampling frequency in Hz (default: 16000).
            channels: Number of audio channels (1 for Mono).
            block_size: Number of samples per buffer frame (default: 480 for 30ms).
            device_input: Name or index of input hardware device.
            device_output: Name or index of output hardware device.
            volume: Software output volume scaling (0.0 to 1.0, default: 1.0).
            mic_gain: Software input gain multiplier (default: 1.0).
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device_input = self._resolve_device(device_input, kind="input")
        self.device_output = self._resolve_device(device_output, kind="output")
        self.volume = min(1.0, max(0.0, float(volume)))
        self.mic_gain = max(0.1, float(mic_gain))

        self._input_queue: queue.Queue = queue.Queue()
        self._input_stream: Optional[sd.InputStream] = None
        self._is_capturing: bool = False
        self._is_muted: bool = False
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
        if self._is_capturing and not self._is_muted:
            if indata.ndim > 1 and indata.shape[1] > 1:
                # Downmix multi-channel / stereo audio to clean mono in RAM
                mono = np.mean(indata, axis=1, dtype=np.float32)
                if self.mic_gain != 1.0:
                    mono = np.clip(mono * self.mic_gain, -1.0, 1.0)
                self._input_queue.put(mono)
            else:
                flat = indata.flatten().astype(np.float32)
                if self.mic_gain != 1.0:
                    flat = np.clip(flat * self.mic_gain, -1.0, 1.0)
                self._input_queue.put(flat)

    def start_capture(self) -> None:
        """Start non-blocking continuous microphone capture."""
        with self._lock:
            if self._input_stream is not None:
                return

            self._is_capturing = True
            while not self._input_queue.empty():
                self._input_queue.get_nowait()

            try:
                self._input_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    blocksize=self.block_size,
                    device=self.device_input,
                    callback=self._input_callback,
                )
                self._input_stream.start()
            except Exception:
                # If hardware rejects mono (e.g. Google AIY Voice HAT dual Knowles MEMS),
                # retry with 2 channels stereo and downmix to mono in RAM
                if self.channels == 1:
                    self.channels = 2
                    self._input_stream = sd.InputStream(
                        samplerate=self.sample_rate,
                        channels=2,
                        dtype="float32",
                        blocksize=self.block_size,
                        device=self.device_input,
                        callback=self._input_callback,
                    )
                    self._input_stream.start()
                else:
                    raise

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

    def mute_capture(self) -> None:
        """Temporarily mute incoming microphone frames during speech playback."""
        with self._lock:
            self._is_muted = True
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    break

    def unmute_capture(self) -> None:
        """Unmute and flush queue to resume capturing clean speech."""
        with self._lock:
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    break
            self._is_muted = False

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
            frame = self._input_queue.get(timeout=timeout)
            return frame.flatten() if frame.ndim > 1 else frame
        except queue.Empty:
            return None

    @staticmethod
    def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio array in memory using linear interpolation.

        Args:
            audio_data: 1D float32 audio array.
            orig_sr: Source sample rate in Hz.
            target_sr: Target sample rate in Hz.

        Returns:
            Resampled 1D float32 audio array.
        """
        if orig_sr == target_sr or len(audio_data) == 0:
            return audio_data
        num_samples = int(len(audio_data) * (target_sr / orig_sr))
        orig_indices = np.linspace(0, len(audio_data) - 1, len(audio_data))
        target_indices = np.linspace(0, len(audio_data) - 1, num_samples)
        return np.interp(target_indices, orig_indices, audio_data).astype(np.float32)

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

        if self.volume != 1.0:
            play_data = np.clip(play_data * self.volume, -1.0, 1.0)

        try:
            sd.play(play_data, samplerate=sr, device=self.device_output, blocking=blocking)
            if blocking:
                sd.wait()
        except Exception:
            # If ALSA hardware rejects 22050 Hz (e.g. AIY Voice HAT MAX98357A I2S DAC),
            # resample in memory to 44100 Hz
            if sr == 22050:
                resampled = self.resample_audio(play_data, 22050, 44100)
                sd.play(resampled, samplerate=44100, device=self.device_output, blocking=blocking)
                if blocking:
                    sd.wait()
            else:
                raise

    def stop_playback(self) -> None:
        """Instantly stop any active audio playback."""
        sd.stop()
