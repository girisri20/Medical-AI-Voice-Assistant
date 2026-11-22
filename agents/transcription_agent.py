"""Agent 2: Speech-to-Text Agent.

Responsibilities:
- Use faster-whisper to convert audio into text.
- Support English initially.
- Return the complete transcript.
- Preserve the original spoken meaning.
- Display transcription progress.

This agent depends on the faster-whisper package, which downloads a Whisper
model automatically on first use (model stored locally under the user's
cache directory). No paid API or external network call is made after the
model download.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result produced by the Speech-to-Text Agent."""

    success: bool = False
    message: str = ""
    transcript: str = ""
    language: str = "en"
    model_name: str = "base"
    segments: list = field(default_factory=list)
    duration_seconds: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class SpeechToTextAgent:
    """Converts processed audio into text using faster-whisper."""

    # Whisper model sizes: tiny, base, small, medium, large-v2, large-v3
    AVAILABLE_MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3")

    def __init__(self, model_name: str = "base", device: str = "cpu") -> None:
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Choose from {self.AVAILABLE_MODELS}"
            )
        self.model_name = model_name
        self.device = device  # "cpu" or "cuda"
        self._model: Any = None
        self.status = "idle"  # idle | loading | transcribing | completed | error
        self.last_result: Optional[TranscriptionResult] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> Any:
        """Lazy-load the faster-whisper model. Downloads on first use."""
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed in this Python environment. "
                "Run 'install.bat' once, then 'run.bat' to start the app. "
                "Or manually: pip install faster-whisper --no-deps"
            ) from exc

        self.status = "loading"
        logger.info("Loading faster-whisper model '%s' on device '%s' ...", self.model_name, self.device)
        self._model = WhisperModel(self.model_name, device=self.device, compute_type="int8")
        self.status = "idle"
        logger.info("faster-whisper model '%s' loaded.", self.model_name)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio_path: Optional[str] = None,
        audio_samples: Optional[Any] = None,
        sample_rate: Optional[int] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio_path: Path to a WAV/MP3/etc. audio file.
            audio_samples: Numpy array of audio samples (float32). When
                provided, this is passed directly to faster-whisper,
                bypassing the ffmpeg/av decode step.
            sample_rate: Sample rate of audio_samples (required if samples
                are provided).

        Returns:
            TranscriptionResult containing the full transcript and segments.
        """
        self.status = "transcribing"
        result = TranscriptionResult(model_name=self.model_name)

        try:
            model = self._load_model()

            # faster-whisper accepts either a file path or a numpy array.
            # Passing samples directly avoids the av/ffmpeg decode path.
            transcribe_kwargs = dict(
                language="en",
                vad_filter=False,  # onnxruntime unavailable for Python 3.15
                word_timestamps=False,
            )
            if audio_samples is not None:
                transcribe_kwargs["audio"] = audio_samples
            elif audio_path and os.path.exists(audio_path):
                transcribe_kwargs["audio"] = audio_path
            else:
                self.status = "error"
                result.success = False
                result.message = "No audio file or samples provided for transcription."
                self.last_result = result
                return result

            # faster-whisper returns (segments, info)
            segments_iter, info = model.transcribe(**transcribe_kwargs)

            segments = []
            full_text_parts = []

            for segment in segments_iter:
                segments.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                    }
                )
                full_text_parts.append(segment.text)

            transcript = " ".join(p.strip() for p in full_text_parts if p.strip())

            result.success = True
            result.message = "Transcription completed successfully."
            result.transcript = transcript
            result.language = getattr(info, "language", "en") or "en"
            result.segments = segments
            result.duration_seconds = getattr(info, "duration", None)
            result.metadata = {
                "model": self.model_name,
                "device": self.device,
                "language": result.language,
                "num_segments": len(segments),
                "audio_path": audio_path,
            }
            self.status = "completed"
        except Exception as exc:  # pragma: no cover - runtime safety
            self.status = "error"
            result.success = False
            result.message = f"Transcription failed: {exc}"
            logger.exception("Transcription error")

        self.last_result = result
        return result

    def transcribe_from_agent_result(
        self, audio_result: Any
    ) -> TranscriptionResult:
        """Convenience: transcribe using the result of AudioProcessingAgent."""
        audio_path = getattr(audio_result, "audio_path", None)
        samples = getattr(audio_result, "samples", None)
        sample_rate = getattr(audio_result, "sample_rate", None)
        if audio_path is None and samples is None:
            result = TranscriptionResult()
            result.success = False
            result.message = "No audio path or samples available from Audio Processing Agent."
            self.last_result = result
            return result
        return self.transcribe(
            audio_path=audio_path,
            audio_samples=samples,
            sample_rate=sample_rate,
        )
