"""Agent 1: Audio Processing Agent.

Responsibilities:
- Accept audio from microphone or uploaded audio file.
- Validate audio format.
- Prepare audio for transcription.
- Pass processed audio to the transcription agent.

This agent does NOT perform speech recognition. It only handles audio I/O
and validation so the downstream transcription agent receives clean audio.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.audio_utils import (
    SUPPORTED_AUDIO_EXTENSIONS,
    audio_bytes_info,
    bytes_to_wav_file,
    read_audio,
    validate_audio_file,
)


@dataclass
class AudioProcessingResult:
    """Result produced by the Audio Processing Agent."""

    success: bool = False
    message: str = ""
    audio_path: Optional[str] = None
    samples: Optional[Any] = None
    sample_rate: Optional[int] = None
    duration_seconds: Optional[float] = None
    source: str = ""  # "microphone" or "upload"
    metadata: dict = field(default_factory=dict)


class AudioProcessingAgent:
    """Processes raw audio input and prepares it for transcription."""

    def __init__(self) -> None:
        self.status = "idle"  # idle | processing | completed | error
        self.last_result: Optional[AudioProcessingResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_uploaded_file(self, file_path: str) -> AudioProcessingResult:
        """Validate and load an uploaded audio file.

        Args:
            file_path: Path to the uploaded audio file on disk.

        Returns:
            AudioProcessingResult with the validated audio ready for the
            transcription agent.
        """
        self.status = "processing"
        result = AudioProcessingResult(source="upload")

        try:
            is_valid, message = validate_audio_file(file_path)
            if not is_valid:
                self.status = "error"
                result.success = False
                result.message = message
                self.last_result = result
                return result

            loaded = read_audio(file_path)
            if loaded is None:
                self.status = "error"
                result.success = False
                result.message = "Failed to decode audio file."
                self.last_result = result
                return result

            samples, sample_rate = loaded

            from utils.audio_utils import get_audio_duration

            duration = get_audio_duration(file_path)

            result.success = True
            result.message = "Audio file validated and loaded successfully."
            result.audio_path = file_path
            result.samples = samples
            result.sample_rate = sample_rate
            result.duration_seconds = duration
            result.metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "sample_rate": sample_rate,
                "duration_seconds": duration,
            }
            self.status = "completed"
        except Exception as exc:  # pragma: no cover - runtime safety
            self.status = "error"
            result.success = False
            result.message = f"Unexpected error processing audio file: {exc}"

        self.last_result = result
        return result

    def process_microphone_bytes(
        self, audio_bytes: bytes, save_to_disk: bool = True
    ) -> AudioProcessingResult:
        """Process raw microphone audio bytes.

        Args:
            audio_bytes: Raw WAV bytes captured from the microphone.
            save_to_disk: Whether to persist bytes to a temporary WAV file
                so faster-whisper can read it directly.

        Returns:
            AudioProcessingResult ready for the transcription agent.
        """
        self.status = "processing"
        result = AudioProcessingResult(source="microphone")

        try:
            is_valid, message = audio_bytes_info(audio_bytes)
            if not is_valid:
                self.status = "error"
                result.success = False
                result.message = message
                self.last_result = result
                return result

            audio_path = None
            if save_to_disk:
                tmp_dir = tempfile.gettempdir()
                audio_path = os.path.join(tmp_dir, "medical_demo_mic.wav")
                if not bytes_to_wav_file(audio_bytes, audio_path):
                    self.status = "error"
                    result.success = False
                    result.message = "Failed to write microphone audio to temp file."
                    self.last_result = result
                    return result

            loaded = read_audio(audio_path) if audio_path else None
            if loaded is None:
                # Try reading from an in-memory buffer instead
                import io

                try:
                    buffer = io.BytesIO(audio_bytes)
                    samples, sample_rate = sf_read(buffer)
                except Exception:
                    self.status = "error"
                    result.success = False
                    result.message = "Failed to decode microphone audio."
                    self.last_result = result
                    return result
            else:
                samples, sample_rate = loaded

            from utils.audio_utils import get_audio_duration

            duration = get_audio_duration(audio_path) if audio_path else None

            result.success = True
            result.message = "Microphone audio received and prepared successfully."
            result.audio_path = audio_path
            result.samples = samples
            result.sample_rate = sample_rate
            result.duration_seconds = duration
            result.metadata = {
                "source": "microphone",
                "sample_rate": sample_rate,
                "duration_seconds": duration,
                "bytes": len(audio_bytes),
            }
            self.status = "completed"
        except Exception as exc:  # pragma: no cover - runtime safety
            self.status = "error"
            result.success = False
            result.message = f"Unexpected error processing microphone audio: {exc}"

        self.last_result = result
        return result


# Imported lazily inside process_microphone_bytes to keep top imports clean.
import soundfile as sf  # noqa: E402


def sf_read(buffer):  # pragma: no cover - thin wrapper
    return sf.read(buffer, dtype="float32")
