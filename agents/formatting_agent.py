"""Agent 5: Transcript Formatting Agent.

Responsibilities:
- Organize the extracted information into sections.
- Generate a clean consultation transcript.
- Maintain the original meaning.
- Clearly mark missing or unclear information.

This agent formats data produced by the Medical Terminology Agent and the
Clinical Information Extraction Agent into a human-readable report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FormattingResult:
    """Result produced by the Transcript Formatting Agent."""

    success: bool = False
    message: str = ""
    formatted_report: str = ""
    json_report: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class FormattingAgent:
    """Builds a clean medical consultation report from agent outputs."""

    def __init__(self) -> None:
        self.status = "idle"  # idle | processing | completed | error
        self.last_result: Optional[FormattingResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def format_report(
        self,
        original_transcript: str,
        terminology_result: Any,
        clinical_result: Any,
    ) -> FormattingResult:
        """Format extracted data into a consultation report.

        Args:
            original_transcript: Raw speech-to-text transcript.
            terminology_result: Result from MedicalTerminologyAgent.
            clinical_result: Result from ClinicalExtractionAgent.

        Returns:
            FormattingResult with plain-text and JSON versions of the report.
        """
        self.status = "processing"
        result = FormattingResult()

        try:
            sections: List[str] = []

            # Header
            sections.append("=" * 60)
            sections.append("MEDICAL CONSULTATION REPORT")
            sections.append("AI Voice Assistant Demo - Transcription & Organization Tool")
            sections.append("=" * 60)
            sections.append("")
            sections.append("DISCLAIMER: This is a transcription/organization tool only.")
            sections.append("It does not diagnose, prescribe, or replace clinical judgment.")
            sections.append("All medical information must be verified by a qualified doctor.")
            sections.append("")

            # 1. Original transcript
            sections.append("1. ORIGINAL TRANSCRIPT (verbatim from speech recognition)")
            sections.append("-" * 60)
            sections.append(original_transcript or "(No transcript available)")
            sections.append("")

            # 2. Medical terminology
            sections.append("2. MEDICAL TERMINOLOGY IDENTIFIED")
            sections.append("-" * 60)
            terms = getattr(terminology_result, "identified_terms", []) or []
            uncertain = getattr(terminology_result, "uncertain_terms", []) or []
            suggestions = getattr(terminology_result, "suggestions", {}) or {}

            if terms:
                for t in terms:
                    sections.append(
                        f"  - {t.get('term', '')} [{t.get('category', 'medical')}]"
                    )
            else:
                sections.append("  (No medical terms identified from the dictionary.)")

            if uncertain:
                sections.append("")
                sections.append("  TERMS FLAGGED FOR DOCTOR VERIFICATION:")
                for u in uncertain:
                    sugg = ", ".join(u.get("suggestions", [])) or "none"
                    sections.append(
                        f"  - {u.get('term', '')} ({u.get('reason', 'verify')}) | "
                        f"Suggestions: {sugg}"
                    )
            else:
                sections.append("")
                sections.append("  (No terms flagged for verification.)")
            sections.append("")

            # 3. Clinical information
            sections.append("3. CLINICAL INFORMATION EXTRACTION")
            sections.append("-" * 60)

            def _list_block(label: str, items: list) -> None:
                sections.append(f"  {label}:")
                if items:
                    for it in items:
                        sections.append(f"    - {it}")
                else:
                    sections.append("    (not mentioned)")

            _list_block("Symptoms", getattr(clinical_result, "symptoms", []) or [])
            _list_block("Symptom Durations", getattr(clinical_result, "symptom_durations", []) or [])
            _list_block("Conditions / Diseases", getattr(clinical_result, "conditions", []) or [])
            _list_block("Medicines", getattr(clinical_result, "medicines", []) or [])
            _list_block("Dosage / Frequency", getattr(clinical_result, "dosages", []) or [])
            _list_block("Tests / Investigations", getattr(clinical_result, "tests", []) or [])
            _list_block("Doctor's Recommendations", getattr(clinical_result, "recommendations", []) or [])
            _list_block("Follow-up Instructions", getattr(clinical_result, "follow_up", []) or [])
            sections.append("")

            missing = getattr(clinical_result, "missing_fields", []) or []
            if missing:
                sections.append("  INFORMATION NOT MENTIONED (flagged for review):")
                for m in missing:
                    sections.append(f"    - {m}")
                sections.append("")

            # JSON report structure
            json_report = {
                "original_transcript": original_transcript,
                "medical_terminology": {
                    "identified_terms": terms,
                    "uncertain_terms": uncertain,
                    "suggestions": suggestions,
                },
                "clinical_information": {
                    "symptoms": getattr(clinical_result, "symptoms", []) or [],
                    "symptom_durations": getattr(clinical_result, "symptom_durations", []) or [],
                    "conditions": getattr(clinical_result, "conditions", []) or [],
                    "medicines": getattr(clinical_result, "medicines", []) or [],
                    "dosages": getattr(clinical_result, "dosages", []) or [],
                    "tests": getattr(clinical_result, "tests", []) or [],
                    "recommendations": getattr(clinical_result, "recommendations", []) or [],
                    "follow_up": getattr(clinical_result, "follow_up", []) or [],
                    "missing_fields": missing,
                },
            }

            report_text = "\n".join(sections)

            result.success = True
            result.message = "Consultation report formatted successfully."
            result.formatted_report = report_text
            result.json_report = json_report
            result.metadata = {
                "terms_found": len(terms),
                "uncertain_terms": len(uncertain),
                "missing_fields_count": len(missing),
            }
            self.status = "completed"
        except Exception as exc:  # pragma: no cover - runtime safety
            self.status = "error"
            result.success = False
            result.message = f"Formatting failed: {exc}"

        self.last_result = result
        return result
