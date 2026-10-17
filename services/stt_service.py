"""Speech-to-Text (STT) service supporting Groq Whisper and Gemini Flash audio transcription.

Processes in-memory audio buffers and transcribes spoken speech into text with zero disk I/O.
"""

import base64
import io
import json
import os
from typing import Optional
import wave
import numpy as np

from core import cloud_api

GEMINI_TIMEOUT_SEC = 4.0
GROQ_TIMEOUT_SEC = 10.0
MAX_TRANSCRIPT_TOKENS = 256  # an 8 s utterance is about 30 words; leave headroom


class SpeechToTextService:
    """In-memory Speech-to-Text transcriber."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        vosk_model_path: Optional[str] = None,
        sample_rate: int = 16000,
        groq_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ) -> None:
        """Initialize STT service with available API keys and offline Vosk model.

        Args:
            groq_api_key: Optional Groq API key for Whisper.
            gemini_api_key: Optional Gemini API key for Gemini Flash audio transcription.
            vosk_model_path: Optional path to local Vosk offline model directory.
            sample_rate: Input audio sample rate (default: 16000 Hz).
            groq_model: Optional Groq STT model name (default: $GROQ_STT_MODEL or whisper-large-v3).
            gemini_model: Optional Gemini model name override (default: $GEMINI_STT_MODEL).
        """
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.gemini_api_key = (
            gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self.groq_model = groq_model or os.environ.get("GROQ_STT_MODEL", "whisper-large-v3")
        self.gemini_model = gemini_model or os.environ.get("GEMINI_STT_MODEL")
        self.sample_rate = sample_rate
        # True when the last transcribe() call tried at least one cloud backend and every one failed
        self.last_cloud_failed = False

        # Initialize offline Vosk model if available
        self._vosk_model = None
        candidates = [
            vosk_model_path,
            os.environ.get("VOSK_MODEL_PATH"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "vosk-model-small-en-us-0.15"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "vosk-model-small-en-us"),
        ]
        chosen_path = next((p for p in candidates if p and os.path.isdir(p)), None)
        if chosen_path:
            try:
                from vosk import Model, SetLogLevel
                SetLogLevel(-1)
                self._vosk_model = Model(chosen_path)
            except Exception as exc:
                print(f"Warning: Failed to load offline Vosk model from '{chosen_path}': {exc}")

        if not self.groq_api_key and not self.gemini_api_key and not self._vosk_model:
            raise ValueError(
                "Neither GROQ_API_KEY nor GEMINI_API_KEY nor offline Vosk model is configured for Speech-to-Text."
            )

    def _audio_to_wav_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert numpy audio array to in-memory WAV bytes with gain normalization.

        Args:
            audio_array: 1D float32 or int16 numpy array.

        Returns:
            Raw WAV file bytes.
        """
        if audio_array.dtype == np.int16:
            samples = audio_array.astype(np.float32) / 32768.0
        else:
            samples = audio_array.astype(np.float32)

        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        if peak > 0.0005:
            # Normalize to 0.85 peak for high-clarity speech recognition
            samples = samples * (0.85 / peak)

        clipped = np.clip(samples, -1.0, 1.0)
        int16_samples = (clipped * 32767.0).astype(np.int16)

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(int16_samples.tobytes())

        return wav_io.getvalue()

    def transcribe_offline(self, audio_array: np.ndarray) -> str:
        """Transcribe in-memory audio array using local offline Vosk engine.

        Args:
            audio_array: 1D float32 or int16 numpy array.

        Returns:
            Transcribed text string.
        """
        if self._vosk_model is None:
            raise RuntimeError("Offline Vosk model is not loaded.")

        from vosk import KaldiRecognizer

        if audio_array.dtype == np.int16:
            int16_bytes = audio_array.tobytes()
        else:
            clipped = np.clip(audio_array, -1.0, 1.0)
            int16_bytes = (clipped * 32767.0).astype(np.int16).tobytes()

        rec = KaldiRecognizer(self._vosk_model, float(self.sample_rate))
        rec.AcceptWaveform(int16_bytes)
        res = json.loads(rec.FinalResult())
        return res.get("text", "").strip()

    def transcribe(self, audio_array: np.ndarray, offline: bool = False) -> str:
        """Transcribe in-memory audio array into text.

        Args:
            audio_array: 1D audio sample array.
            offline: If True, force using the offline local model.

        Returns:
            Transcribed text string.

        Raises:
            RuntimeError: If transcription fails.

        After the call, last_cloud_failed tells whether the cloud backends were tried and
        all failed, so a caller can check whether the network is down.
        """
        self.last_cloud_failed = False
        if offline:
            return self.transcribe_offline(audio_array)

        wav_bytes = self._audio_to_wav_bytes(audio_array)

        backends = []
        if self.groq_api_key:
            backends.append(("Groq Whisper", lambda: self._transcribe_groq(wav_bytes)))
        if self.gemini_api_key:
            backends.append(("Gemini", lambda: self._transcribe_gemini(wav_bytes)))
        if self._vosk_model:
            backends.append(("Vosk", lambda: self.transcribe_offline(audio_array)))
        if not backends:
            raise RuntimeError("No transcription backend available.")

        errors = []
        cloud_attempts = 0
        for name, run in backends:
            is_cloud = name != "Vosk"
            cloud_attempts += is_cloud
            try:
                text = run()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                print(f"  (STT backend {name} failed: {exc})")
                continue
            # Reaching Vosk with cloud backends configured means each of them failed first
            self.last_cloud_failed = (not is_cloud) and cloud_attempts > 0
            return text
        self.last_cloud_failed = cloud_attempts > 0
        raise RuntimeError("All transcription backends failed: " + "; ".join(errors))

    def _transcribe_gemini(self, wav_bytes: bytes) -> str:
        """Transcribe using fast Google Gemini multimodal audio transcription."""
        b64_data = base64.b64encode(wav_bytes).decode("utf-8")
        candidates_models = [
            self.gemini_model,
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.5-transcribe",
        ]
        models_to_try = [m for m in candidates_models if m]

        prompt_text = (
            "Listen carefully to the audio and transcribe the spoken words verbatim. "
            "If the audio is silent, muddled, or contains no intelligible words, return nothing. "
            "Return ONLY the plain transcript text with no formatting, markdown, quotes, or commentary."
        )

        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": "audio/wav", "data": b64_data}},
                    {"text": prompt_text},
                ]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": MAX_TRANSCRIPT_TOKENS,
            },
        }

        answered = False
        last_error: Optional[Exception] = None
        for model in models_to_try:
            try:
                cand_parts = cloud_api.gemini_generate_parts(
                    self.gemini_api_key, model, payload, timeout=GEMINI_TIMEOUT_SEC
                )
                answered = True
                for p in cand_parts:
                    if isinstance(p, dict):
                        if "audioTranscription" in p:
                            transcribed = p["audioTranscription"].get("text", "").strip()
                            if transcribed:
                                return transcribed
                        if "text" in p:
                            raw_t = p.get("text", "").strip()
                            if raw_t and "cannot fulfill" not in raw_t.lower():
                                return raw_t
            except Exception as exc:
                last_error = exc

        # An empty transcript is valid only when a model answered; if every model failed,
        # raise so the caller can fall back to another backend.
        if not answered and last_error is not None:
            raise RuntimeError(f"all Gemini models failed: {last_error}") from last_error
        return ""

    def _transcribe_groq(self, wav_bytes: bytes) -> str:
        """Transcribe using Groq Whisper multipart audio endpoint."""
        return cloud_api.groq_transcribe(
            self.groq_api_key, self.groq_model, wav_bytes, timeout=GROQ_TIMEOUT_SEC
        )
