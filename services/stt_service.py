"""Speech-to-Text (STT) service supporting Groq Whisper and Gemini Flash audio transcription.

Processes in-memory audio buffers and transcribes spoken speech into text with zero disk I/O.
"""

import base64
import io
import json
import os
from typing import Optional
import urllib.error
import urllib.request
import wave
import numpy as np


class SpeechToTextService:
    """In-memory Speech-to-Text transcriber."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize STT service with available API keys.

        Args:
            groq_api_key: Optional Groq API key for Whisper Large v3.
            gemini_api_key: Optional Gemini API key for Gemini Flash audio transcription.
            sample_rate: Input audio sample rate (default: 16000 Hz).
        """
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.gemini_api_key = (
            gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self.sample_rate = sample_rate

        if not self.groq_api_key and not self.gemini_api_key:
            raise ValueError(
                "Neither GROQ_API_KEY nor GEMINI_API_KEY is configured for Speech-to-Text."
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

    def transcribe(self, audio_array: np.ndarray) -> str:
        """Transcribe in-memory audio array into text.

        Args:
            audio_array: 1D audio sample array.

        Returns:
            Transcribed text string.

        Raises:
            RuntimeError: If transcription fails.
        """
        wav_bytes = self._audio_to_wav_bytes(audio_array)

        if self.groq_api_key:
            try:
                return self._transcribe_groq(wav_bytes)
            except Exception as exc:
                if self.gemini_api_key:
                    return self._transcribe_gemini(wav_bytes)
                raise RuntimeError(f"Groq Whisper transcription failed: {exc}") from exc

        return self._transcribe_gemini(wav_bytes)

    def _transcribe_gemini(self, wav_bytes: bytes) -> str:
        """Transcribe using fast Google Gemini multimodal audio transcription."""
        b64_data = base64.b64encode(wav_bytes).decode("utf-8")
        models_to_try = [
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.5-transcribe",
        ]

        prompt_text = (
            "Listen carefully to the audio and transcribe the spoken words verbatim. "
            "If the audio is silent, muddled, or contains no intelligible words, return nothing. "
            "Return ONLY the plain transcript text with no formatting, markdown, quotes, or commentary."
        )

        for model in models_to_try:
            parts = [
                {
                    "inlineData": {
                        "mimeType": "audio/wav",
                        "data": b64_data,
                    }
                },
                {"text": prompt_text},
            ]

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 60,
                },
            }

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={self.gemini_api_key}"
            )
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=4) as response:
                    result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if not candidates:
                    continue
                cand_parts = candidates[0].get("content", {}).get("parts", [])
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
            except Exception:
                continue

        return ""

    def _transcribe_groq(self, wav_bytes: bytes) -> str:
        """Transcribe using Groq Whisper multipart audio endpoint."""
        import uuid
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = io.BytesIO()

        # Part 1: model
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(b'Content-Disposition: form-data; name="model"\r\n\r\n')
        body.write(b"whisper-large-v3\r\n")

        # Part 2: file
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
        body.write(b"Content-Type: audio/wav\r\n\r\n")
        body.write(wav_bytes)
        body.write(b"\r\n")

        body.write(f"--{boundary}--\r\n".encode("utf-8"))

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            data=body.getvalue(),
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result.get("text", "").strip()
