"""In-memory Piper Text-to-Speech (TTS) synthesis service.

Synthesizes character speech directly into memory as raw PCM numpy arrays
without writing temporary audio files to disk.
"""

from collections import OrderedDict
import hashlib
import os
import re
import subprocess
import threading
import time
from typing import Optional, Tuple
import numpy as np

from core import system_utils


DEFAULT_MEMORY_CACHE_ENTRIES = 128
DEFAULT_DISK_CACHE_MB = 1024.0


class PiperTTSService:
    """Local neural TTS synthesizer using Piper ONNX models in RAM."""

    def __init__(
        self,
        model_path: str = "models/piper/en_GB-alan-medium.onnx",
        sample_rate: int = 22050,
        length_scale: float = 1.15,
        noise_scale: float = 0.85,
        pause_duration_sec: float = 0.55,
        cache_dir: Optional[str] = None,
        max_memory_entries: int = DEFAULT_MEMORY_CACHE_ENTRIES,
        max_disk_cache_mb: Optional[float] = None,
        lazy_load: bool = False,
    ) -> None:
        """Initialize the Piper TTS service.

        Args:
            model_path: Path to the ONNX model file, or a voice ID such as 'en_GB-alan-medium'.
            sample_rate: Model output sample rate in Hz (default: 22050).
            length_scale: Phoneme length / tempo scale (higher = slower).
            noise_scale: Noise level for character inflection.
            pause_duration_sec: Duration in seconds of silence inserted at ellipsis (...) breaks.
            cache_dir: Directory for the persistent audio cache (default: $ALSAGEIST_CACHE_DIR or models/cache).
            max_memory_entries: Most recently used phrases kept in RAM.
            max_disk_cache_mb: Disk cache size limit in MB; least recently used files are deleted
                beyond it (default: $ALSAGEIST_CACHE_MAX_MB or 1024).
            lazy_load: If True, load the voice model on the first phrase that is not already
                cached instead of at construction. Loading takes tens of seconds on a
                Raspberry Pi, so short-lived tools use this; a long-running service keeps
                the default so its first uncached reply is not delayed.
        """
        self.model_path = system_utils.resolve_piper_model_path(model_path=model_path)
        self.sample_rate = sample_rate
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.pause_duration_sec = pause_duration_sec

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Piper model not found at '{self.model_path}'. "
                "Please download the ONNX model and config."
            )

        self._voice = None
        self._load_error: Optional[Exception] = None
        self._syn_config = None
        self._voice_loaded = False
        self._load_lock = threading.Lock()
        self.voice_load_sec = 0.0  # time spent loading the voice model so far
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.max_memory_entries = max(1, int(max_memory_entries))
        if max_disk_cache_mb is None:
            max_disk_cache_mb = float(os.environ.get("ALSAGEIST_CACHE_MAX_MB", DEFAULT_DISK_CACHE_MB))
        self.max_disk_cache_bytes = int(max_disk_cache_mb * 1024 * 1024)
        default_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "cache")
        self.cache_dir = cache_dir or os.environ.get("ALSAGEIST_CACHE_DIR", default_cache)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError:
            pass
        self._prune_disk_cache()

        if not lazy_load:
            self._ensure_voice()

    @property
    def voice_loaded(self) -> bool:
        """Return True once loading the in-process voice has been attempted."""
        return self._voice_loaded

    def _ensure_voice(self) -> None:
        """Load the Piper voice in-process once; on failure fall back to the piper command."""
        if self._voice_loaded:
            return
        with self._load_lock:
            if self._voice_loaded:
                return
            started = time.perf_counter()
            try:
                import piper
                self._voice = piper.PiperVoice.load(self.model_path)
                self._syn_config = piper.SynthesisConfig(
                    length_scale=self.length_scale,
                    noise_scale=self.noise_scale,
                )
            except Exception as exc:
                self._voice = None
                self._load_error = exc
                print(
                    f"Warning: Piper could not load '{self.model_path}' in-process ({exc}); "
                    "falling back to the 'piper' command."
                )
            finally:
                self.voice_load_sec += time.perf_counter() - started
                self._voice_loaded = True

    def _synthesize_chunk(self, chunk_text: str) -> np.ndarray:
        """Synthesize a single text chunk into raw PCM numpy array.

        Args:
            chunk_text: Single sentence or phrase to speak.

        Returns:
            1D float32 numpy audio array.
        """
        # Apply phonetic pronunciation corrections for Piper TTS
        normalized_text = re.sub(r"\bKrueger\b", "Kruger", chunk_text, flags=re.IGNORECASE)
        normalized_text = re.sub(r"\bTannh[aä]user\b", "Tahnhoyzer", normalized_text, flags=re.IGNORECASE)
        normalized_text = re.sub(r"\bMwa+ha+ha+!*\b", "Ha ha ha ha ha!", normalized_text, flags=re.IGNORECASE)
        normalized_text = re.sub(r"\bHasta\b", "Ahsta", normalized_text, flags=re.IGNORECASE)

        self._ensure_voice()
        if self._voice is not None:
            chunks = list(self._voice.synthesize(normalized_text, syn_config=self._syn_config))
            if chunks:
                return np.concatenate([c.audio_float_array for c in chunks])
            return np.zeros(0, dtype=np.float32)

        # Fallback to subprocess invocation if in-memory voice failed to load
        cmd = [
            "piper",
            "-m", self.model_path,
            "--output-raw",
            "--length-scale", str(self.length_scale),
            "--noise-scale", str(self.noise_scale),
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Piper synthesis unavailable: in-process load failed ({self._load_error}) "
                f"and the 'piper' command could not be run ({exc})."
            ) from exc

        raw_pcm, stderr = process.communicate(input=normalized_text.encode("utf-8"))

        if process.returncode != 0 or not raw_pcm:
            detail = stderr.decode("utf-8", errors="replace").strip()[-300:]
            raise RuntimeError(f"Piper synthesis failed (exit code {process.returncode}): {detail}")

        int16_samples = np.frombuffer(raw_pcm, dtype=np.int16)
        return int16_samples.astype(np.float32) / 32768.0

    def _get_cache_path(self, text: str) -> str:
        """Return the filesystem path for cached audio array.

        The key covers the text, voice model, tempo and noise, so changing any of
        them never serves audio synthesized with the old settings.

        Args:
            text: Dialogue text.

        Returns:
            Absolute path to .npy cache file.
        """
        voice = os.path.basename(self.model_path)
        key = f"{voice}|{self.length_scale}|{self.noise_scale}|{text}"
        return os.path.join(self.cache_dir, f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.npy")

    def _remember(self, text: str, audio: np.ndarray) -> None:
        """Store audio in the RAM cache, evicting the least recently used phrase when full."""
        self._cache[text] = audio
        self._cache.move_to_end(text)
        while len(self._cache) > self.max_memory_entries:
            self._cache.popitem(last=False)

    def _prune_disk_cache(self) -> None:
        """Delete least recently used cache files until the cache fits its size limit."""
        try:
            entries = []
            for entry in os.scandir(self.cache_dir):
                if entry.name.endswith(".npy") and entry.is_file():
                    stat = entry.stat()
                    entries.append((stat.st_mtime, stat.st_size, entry.path))
        except OSError:
            return

        total = sum(size for _, size, _ in entries)
        for _, size, path in sorted(entries):
            if total <= self.max_disk_cache_bytes:
                break
            try:
                os.unlink(path)
                total -= size
            except OSError:
                pass

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize text into in-memory audio array, inserting pauses at ellipsis breaks.

        Args:
            text: Text to speak.

        Returns:
            Tuple of (1D float32 numpy audio array, sample_rate).

        Raises:
            RuntimeError: If synthesis fails.
        """
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32), self.sample_rate

        # 1. Check in-memory RAM cache
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text], self.sample_rate

        # 2. Check persistent disk cache
        cache_path = self._get_cache_path(text)
        if os.path.isfile(cache_path):
            try:
                audio = np.load(cache_path)
                os.utime(cache_path)  # mark as recently used
                self._remember(text, audio)
                return audio, self.sample_rate
            except (OSError, ValueError, EOFError):
                pass

        # Split on explicit pauses (ellipsis '...' or multiple newlines)
        chunks = [c.strip() for c in re.split(r"\.{2,}|\n+", text) if c.strip()]

        if len(chunks) <= 1:
            audio = self._synthesize_chunk(text)
        else:
            audio_parts = []
            pause_samples = np.zeros(int(self.sample_rate * self.pause_duration_sec), dtype=np.float32)

            for i, chunk in enumerate(chunks):
                chunk_audio = self._synthesize_chunk(chunk)
                audio_parts.append(chunk_audio)
                if i < len(chunks) - 1:
                    audio_parts.append(pause_samples)

            audio = np.concatenate(audio_parts)

        # Store in RAM and disk cache
        self._remember(text, audio)
        try:
            tmp_path = f"{cache_path[:-4]}.tmp.npy"
            np.save(tmp_path, audio)
            os.replace(tmp_path, cache_path)
        except OSError:
            return audio, self.sample_rate
        self._prune_disk_cache()

        return audio, self.sample_rate
