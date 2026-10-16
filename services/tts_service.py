"""In-memory Piper Text-to-Speech (TTS) synthesis service.

Synthesizes character speech directly into memory as raw PCM numpy arrays
without writing temporary audio files to disk.
"""

import hashlib
import os
import re
import subprocess
from typing import Optional, Tuple
import numpy as np


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
    ) -> None:
        """Initialize the Piper TTS service.

        Args:
            model_path: Path to the ONNX model file.
            sample_rate: Model output sample rate in Hz (default: 22050).
            length_scale: Phoneme length / tempo scale (higher = slower).
            noise_scale: Noise level for character inflection.
            pause_duration_sec: Duration in seconds of silence inserted at ellipsis (...) breaks.
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.pause_duration_sec = pause_duration_sec

        if not os.path.isabs(self.model_path) and not os.path.exists(self.model_path):
            repo_candidate = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", self.model_path)
            )
            if os.path.exists(repo_candidate):
                self.model_path = repo_candidate

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Piper model not found at '{self.model_path}'. "
                "Please download the ONNX model and config."
            )

        self._voice = None
        self._syn_config = None
        self._cache: dict = {}
        default_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "cache")
        self.cache_dir = cache_dir or os.environ.get("ALSAGEIST_CACHE_DIR", default_cache)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception:
            pass

        try:
            import piper
            self._voice = piper.PiperVoice.load(self.model_path)
            self._syn_config = piper.SynthesisConfig(
                length_scale=self.length_scale,
                noise_scale=self.noise_scale,
            )
        except Exception:
            self._voice = None

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

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        raw_pcm, _ = process.communicate(input=normalized_text.encode("utf-8"))

        if process.returncode != 0 or not raw_pcm:
            raise RuntimeError(f"Piper synthesis failed (exit code {process.returncode})")

        int16_samples = np.frombuffer(raw_pcm, dtype=np.int16)
        return int16_samples.astype(np.float32) / 32768.0

    def _get_cache_path(self, text: str) -> str:
        """Return the filesystem path for cached audio array.

        Args:
            text: Dialogue text.

        Returns:
            Absolute path to .npy cache file.
        """
        cache_key = hashlib.md5(f"{text}_{self.length_scale}_{self.noise_scale}".encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_key}.npy")

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
            return self._cache[text], self.sample_rate

        # 2. Check persistent disk cache
        cache_path = self._get_cache_path(text)
        if os.path.isfile(cache_path):
            try:
                audio = np.load(cache_path)
                self._cache[text] = audio
                return audio, self.sample_rate
            except Exception:
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
        self._cache[text] = audio
        try:
            np.save(cache_path, audio)
        except Exception:
            pass

        return audio, self.sample_rate
