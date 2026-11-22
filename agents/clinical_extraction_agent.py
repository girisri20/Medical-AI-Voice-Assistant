"""Agent 4: Clinical Information Extraction Agent.

Responsibilities:
- Extract relevant information from the transcript.
- Identify:
  - Patient symptoms
  - Duration of symptoms
  - Diseases or conditions mentioned
  - Medicines mentioned
  - Dosage and frequency, if spoken
  - Tests and investigations
  - Doctor's recommendations
  - Follow-up instructions

Do NOT invent information that the doctor did not mention.

NOTE: This is a RULE-BASED demonstration agent. It uses keyword matching and
simple regex patterns. It is NOT an autonomous LLM agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClinicalExtractionResult:
    """Result produced by the Clinical Information Extraction Agent."""

    success: bool = False
    message: str = ""
    symptoms: list = field(default_factory=list)
    symptom_durations: list = field(default_factory=list)
    conditions: list = field(default_factory=list)
    medicines: list = field(default_factory=list)
    dosages: list = field(default_factory=list)
    tests: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    follow_up: list = field(default_factory=list)
    missing_fields: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ClinicalExtractionAgent:
    """Extracts structured clinical information from a medical transcript."""

    # Symptom keywords
    SYMPTOM_KEYWORDS = [
        "fever", "headache", "cough", "fatigue", "nausea", "vomiting",
        "dizziness", "chest pain", "shortness of breath", "abdominal pain",
        "sore throat", "rash", "pain", "soreness", "swelling",
    ]

    # Condition / disease keywords
    CONDITION_KEYWORDS = [
        "diabetes", "hypertension", "asthma", "pneumonia", "bronchitis",
        "covid", "influenza", "migraine", "arthritis", "depression", "anxiety",
    ]

    # Medicine keywords
    MEDICINE_KEYWORDS = [
        "paracetamol", "ibuprofen", "aspirin", "amoxicillin", "azithromycin",
        "metformin", "atorvastatin", "omeprazole",
    ]

    # Test / investigation keywords
    TEST_KEYWORDS = [
        "blood test", "x-ray", "mri", "ct scan", "ecg", "ekg", "biopsy",
    ]

    # Recommendation cue phrases
    RECOMMENDATION_CUES = [
        "recommend", "suggest", "advise", "should", "take", "prescribe",
        "avoid", "rest", "drink", "follow up", "return",
    ]

    # Follow-up cue phrases
    FOLLOW_UP_CUES = [
        "follow up", "follow-up", "return", "revisit", "next visit",
        "in a week", "in 2 weeks", "in one month",
    ]

    # Duration patterns (e.g., "for 3 days", "since yesterday", "2 weeks")
    DURATION_PATTERNS = [
        r"for\s+(\d+\s*(?:second|minute|hour|day|week|month|year)s?)",
        r"since\s+(yesterday|today|last week|last month|\d+\s*(?:day|week|month|year)s?)",
        r"(\d+\s*(?:day|week|month|year)s?)\s+(ago|old)",
    ]

    # Dosage patterns (e.g., "500 mg", "twice daily", "once a day")
    DOSAGE_PATTERNS = [
        r"(\d+\s*(?:mg|g|ml|mcg|microgram|tablet|pill|capsule)s?)",
        r"(once|twice|thrice)\s+(?:a|per)\s+(day|daily)",
        r"(daily|bid|tid|qid|every\s+\d+\s+hours?)",
    ]

    def __init__(self) -> None:
        self.status = "idle"  # idle | processing | completed | error
        self.last_result: Optional[ClinicalExtractionResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, transcript: str) -> ClinicalExtractionResult:
        """Extract clinical information from a transcript.

        Args:
            transcript: Medical consultation transcript.

        Returns:
            ClinicalExtractionResult with categorized findings.
        """
        self.status = "processing"
        result = ClinicalExtractionResult()

        try:
            if not transcript or not transcript.strip():
                self.status = "error"
                result.success = False
                result.message = "Empty transcript provided."
                self.last_result = result
                return result

            text = transcript
            lower = text.lower()

            # --- Symptoms ---
            symptoms = self._find_keywords(lower, self.SYMPTOM_KEYWORDS)
            result.symptoms = symptoms

            # --- Durations ---
            durations = self._find_durations(text)
            result.symptom_durations = durations

            # --- Conditions ---
            conditions = self._find_keywords(lower, self.CONDITION_KEYWORDS)
            result.conditions = conditions

            # --- Medicines ---
            medicines = self._find_keywords(lower, self.MEDICINE_KEYWORDS)
            result.medicines = medicines

            # --- Dosages ---
            dosages = self._find_dosages(text)
            result.dosages = dosages

            # --- Tests ---
            tests = self._find_keywords(lower, self.TEST_KEYWORDS)
            result.tests = tests

            # --- Recommendations ---
            recommendations = self._find_cue_sentences(text, self.RECOMMENDATION_CUES)
            result.recommendations = recommendations

            # --- Follow-up ---
            follow_up = self._find_cue_sentences(text, self.FOLLOW_UP_CUES)
            result.follow_up = follow_up

            # --- Missing fields (for the formatting agent to flag) ---
            missing = []
            if not symptoms:
                missing.append("symptoms")
            if not conditions:
                missing.append("conditions/diseases")
            if not medicines:
                missing.append("medicines")
            if not tests:
                missing.append("tests/investigations")
            if not recommendations:
                missing.append("doctor's recommendations")
            if not follow_up:
                missing.append("follow-up instructions")
            result.missing_fields = missing

            result.success = True
            result.message = (
                f"Extraction complete: {len(symptoms)} symptom(s), "
                f"{len(conditions)} condition(s), {len(medicines)} medicine(s), "
                f"{len(tests)} test(s)."
            )
            result.metadata = {
                "symptom_count": len(symptoms),
                "condition_count": len(conditions),
                "medicine_count": len(medicines),
                "dosage_count": len(dosages),
                "test_count": len(tests),
                "recommendation_count": len(recommendations),
                "follow_up_count": len(follow_up),
                "missing_count": len(missing),
            }
            self.status = "completed"
        except Exception as exc:  # pragma: no cover - runtime safety
            self.status = "error"
            result.success = False
            result.message = f"Clinical extraction failed: {exc}"

        self.last_result = result
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_keywords(lower_text: str, keywords: List[str]) -> List[str]:
        found = []
        for kw in keywords:
            if kw in lower_text:
                found.append(kw)
        return found

    @staticmethod
    def _find_durations(text: str) -> List[str]:
        found = []
        for pattern in ClinicalExtractionAgent.DURATION_PATTERNS:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                found.append(m.group(0).strip())
        return found

    @staticmethod
    def _find_dosages(text: str) -> List[str]:
        found = []
        for pattern in ClinicalExtractionAgent.DOSAGE_PATTERNS:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                found.append(m.group(0).strip())
        return found

    @staticmethod
    def _find_cue_sentences(text: str, cues: List[str]) -> List[str]:
        """Return sentences containing any of the cue phrases."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        found = []
        lower_sentences = [s.lower() for s in sentences]
        for sentence, lower in zip(sentences, lower_sentences):
            if any(cue in lower for cue in cues):
                found.append(sentence.strip())
        return found
