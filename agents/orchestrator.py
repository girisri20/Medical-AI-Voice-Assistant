"""Agent 6: Agent Orchestrator.

Responsibilities:
- Coordinate all agents.
- Execute agents in the correct sequence.
- Pass output from one agent to another.
- Maintain agent execution status.
- Display each agent's output and processing status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.audio_agent import AudioProcessingAgent, AudioProcessingResult
from agents.clinical_extraction_agent import (
    ClinicalExtractionAgent,
    ClinicalExtractionResult,
)
from agents.formatting_agent import FormattingAgent, FormattingResult
from agents.medical_terminology_agent import (
    MedicalTerminologyAgent,
    TerminologyResult,
)
from agents.transcription_agent import SpeechToTextAgent, TranscriptionResult


@dataclass
class AgentStep:
    """Represents a single step in the orchestration pipeline."""

    name: str
    status: str = "pending"  # pending | running | completed | error
    message: str = ""
    output: Any = None


@dataclass
class OrchestrationResult:
    """Aggregate result from running all agents."""

    success: bool = False
    message: str = ""
    steps: List[AgentStep] = field(default_factory=list)

    # Individual agent outputs
    audio_result: Optional[AudioProcessingResult] = None
    transcription_result: Optional[TranscriptionResult] = None
    terminology_result: Optional[TerminologyResult] = None
    clinical_result: Optional[ClinicalExtractionResult] = None
    formatting_result: Optional[FormattingResult] = None

    @property
    def final_report(self) -> str:
        if self.formatting_result:
            return self.formatting_result.formatted_report
        return ""

    @property
    def json_report(self) -> dict:
        if self.formatting_result:
            return self.formatting_result.json_report
        return {}

    @property
    def original_transcript(self) -> str:
        if self.transcription_result:
            return self.transcription_result.transcript
        return ""

    @property
    def all_completed(self) -> bool:
        return all(step.status == "completed" for step in self.steps)


class AgentOrchestrator:
    """Coordinates the six-agent pipeline for the medical speech-to-text demo."""

    AGENT_ORDER = (
        "Audio Processing",
        "Speech-to-Text",
        "Medical Terminology",
        "Clinical Information Extraction",
        "Transcript Formatting",
    )

    def __init__(
        self,
        whisper_model: str = "base",
        device: str = "cpu",
    ) -> None:
        self.audio_agent = AudioProcessingAgent()
        self.transcription_agent = SpeechToTextAgent(
            model_name=whisper_model, device=device
        )
        self.terminology_agent = MedicalTerminologyAgent()
        self.clinical_agent = ClinicalExtractionAgent()
        self.formatting_agent = FormattingAgent()

        self.status = "idle"  # idle | running | completed | error
        self.last_result: Optional[OrchestrationResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_uploaded_audio(self, file_path: str) -> OrchestrationResult:
        """Run the full pipeline on an uploaded audio file."""
        return self._run_pipeline(
            lambda: self.audio_agent.process_uploaded_file(file_path)
        )

    def process_microphone_audio(self, audio_bytes: bytes) -> OrchestrationResult:
        """Run the full pipeline on microphone audio bytes."""
        return self._run_pipeline(
            lambda: self.audio_agent.process_microphone_bytes(audio_bytes)
        )

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------
    def _run_pipeline(self, audio_callable) -> OrchestrationResult:
        self.status = "running"
        result = OrchestrationResult()

        steps = [
            AgentStep("Audio Processing"),
            AgentStep("Speech-to-Text"),
            AgentStep("Medical Terminology"),
            AgentStep("Clinical Information Extraction"),
            AgentStep("Transcript Formatting"),
        ]
        result.steps = steps

        try:
            # ---- Step 1: Audio Processing ----
            step = steps[0]
            step.status = "running"
            audio_result = audio_callable()
            result.audio_result = audio_result
            step.status = "completed" if audio_result.success else "error"
            step.message = audio_result.message
            step.output = audio_result

            if not audio_result.success:
                self._finalize(result, steps, success=False)
                return result

            # ---- Step 2: Speech-to-Text ----
            step = steps[1]
            step.status = "running"
            transcription_result = self.transcription_agent.transcribe_from_agent_result(
                audio_result
            )
            result.transcription_result = transcription_result
            step.status = "completed" if transcription_result.success else "error"
            step.message = transcription_result.message
            step.output = transcription_result

            if not transcription_result.success:
                self._finalize(result, steps, success=False)
                return result

            transcript = transcription_result.transcript

            # ---- Step 3: Medical Terminology ----
            step = steps[2]
            step.status = "running"
            terminology_result = self.terminology_agent.analyze(transcript)
            result.terminology_result = terminology_result
            step.status = "completed" if terminology_result.success else "error"
            step.message = terminology_result.message
            step.output = terminology_result

            if not terminology_result.success:
                self._finalize(result, steps, success=False)
                return result

            # ---- Step 4: Clinical Information Extraction ----
            step = steps[3]
            step.status = "running"
            clinical_result = self.clinical_agent.extract(transcript)
            result.clinical_result = clinical_result
            step.status = "completed" if clinical_result.success else "error"
            step.message = clinical_result.message
            step.output = clinical_result

            if not clinical_result.success:
                self._finalize(result, steps, success=False)
                return result

            # ---- Step 5: Transcript Formatting ----
            step = steps[4]
            step.status = "running"
            formatting_result = self.formatting_agent.format_report(
                transcript, terminology_result, clinical_result
            )
            result.formatting_result = formatting_result
            step.status = "completed" if formatting_result.success else "error"
            step.message = formatting_result.message
            step.output = formatting_result

            self._finalize(result, steps, success=formatting_result.success)
        except Exception as exc:  # pragma: no cover - runtime safety
            self.status = "error"
            result.success = False
            result.message = f"Orchestration failed: {exc}"
            for step in steps:
                if step.status == "pending":
                    step.status = "error"
                    step.message = "Pipeline halted due to error."

        self.last_result = result
        return result

    def _finalize(
        self,
        result: OrchestrationResult,
        steps: List[AgentStep],
        success: bool,
    ) -> None:
        result.success = success and all(s.status == "completed" for s in steps)
        if result.success:
            result.message = "All agents completed successfully."
            self.status = "completed"
        else:
            result.message = "One or more agents failed."
            self.status = "error"
