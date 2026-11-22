"""Agent 3: Medical Terminology Agent.

Responsibilities:
- Identify medical terms, diseases, symptoms, medicines, and procedures.
- Highlight potentially misrecognized medical words.
- Provide possible terminology suggestions.
- Never silently replace uncertain medical terminology.
- Mark uncertain words for doctor verification.

NOTE: This is a RULE-BASED demonstration agent. It uses a curated keyword
dictionary and simple heuristics. It is NOT an autonomous LLM agent and does
not perform clinical diagnosis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TerminologyResult:
    """Result produced by the Medical Terminology Agent."""

    success: bool = False
    message: str = ""
    original_transcript: str = ""
    highlighted_transcript: str = ""
    identified_terms: list = field(default_factory=list)
    uncertain_terms: list = field(default_factory=list)
    suggestions: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class MedicalTerminologyAgent:
    """Flags medical terminology in a transcript using rule-based keyword matching."""

    # Curated medical term dictionary for the demo. Each entry maps a known
    # term to a category and a list of acceptable alternatives/suggestions.
    MEDICAL_TERMS: Dict[str, Dict[str, Any]] = {
        # Symptoms
        "fever": {"category": "symptom", "suggestions": ["pyrexia"]},
        "headache": {"category": "symptom", "suggestions": ["cephalalgia"]},
        "cough": {"category": "symptom", "suggestions": ["tussis"]},
        "fatigue": {"category": "symptom", "suggestions": ["lethargy", "malaise"]},
        "nausea": {"category": "symptom", "suggestions": ["emesis"]},
        "vomiting": {"category": "symptom", "suggestions": ["emesis"]},
        "dizziness": {"category": "symptom", "suggestions": ["vertigo"]},
        "chest pain": {"category": "symptom", "suggestions": ["angina"]},
        "shortness of breath": {"category": "symptom", "suggestions": ["dyspnea"]},
        "abdominal pain": {"category": "symptom", "suggestions": ["gastralgia"]},
        "sore throat": {"category": "symptom", "suggestions": ["pharyngitis"]},
        "rash": {"category": "symptom", "suggestions": ["exanthem"]},
        # Diseases / conditions
        "diabetes": {"category": "condition", "suggestions": ["diabetes mellitus"]},
        "hypertension": {"category": "condition", "suggestions": ["high blood pressure"]},
        "asthma": {"category": "condition", "suggestions": ["bronchial asthma"]},
        "pneumonia": {"category": "condition", "suggestions": []},
        "bronchitis": {"category": "condition", "suggestions": []},
        "covid": {"category": "condition", "suggestions": ["covid-19", "sars-cov-2"]},
        "influenza": {"category": "condition", "suggestions": ["flu"]},
        "migraine": {"category": "condition", "suggestions": []},
        "arthritis": {"category": "condition", "suggestions": []},
        "depression": {"category": "condition", "suggestions": []},
        "anxiety": {"category": "condition", "suggestions": []},
        # Medicines
        "paracetamol": {"category": "medicine", "suggestions": ["acetaminophen"]},
        "ibuprofen": {"category": "medicine", "suggestions": ["advil", "motrin"]},
        "aspirin": {"category": "medicine", "suggestions": []},
        "amoxicillin": {"category": "medicine", "suggestions": []},
        "azithromycin": {"category": "medicine", "suggestions": ["zithromax"]},
        "metformin": {"category": "medicine", "suggestions": []},
        "atorvastatin": {"category": "medicine", "suggestions": ["lipitor"]},
        "omeprazole": {"category": "medicine", "suggestions": ["prilosec"]},
        # Procedures / tests
        "blood test": {"category": "test", "suggestions": ["cbc", "complete blood count"]},
        "x-ray": {"category": "test", "suggestions": ["radiograph"]},
        "mri": {"category": "test", "suggestions": ["magnetic resonance imaging"]},
        "ct scan": {"category": "test", "suggestions": ["computed tomography"]},
        "ecg": {"category": "test", "suggestions": ["ekg", "electrocardiogram"]},
        "biopsy": {"category": "procedure", "suggestions": []},
    }

    def __init__(self) -> None:
        self.status = "idle"  # idle | processing | completed | error
        self.last_result: Optional[TerminologyResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, transcript: str) -> TerminologyResult:
        """Analyze a transcript for medical terminology.

        Args:
            transcript: Raw speech-to-text transcript.

        Returns:
            TerminologyResult with highlighted text and flagged terms.
        """
        self.status = "processing"
        result = TerminologyResult(original_transcript=transcript)

        try:
            if not transcript or not transcript.strip():
                self.status = "error"
                result.success = False
                result.message = "Empty transcript provided."
                self.last_result = result
                return result

            text = transcript
            lower_text = text.lower()

            identified: List[Dict[str, Any]] = []
            uncertain: List[Dict[str, Any]] = []
            suggestions: Dict[str, List[str]] = {}

            # Sort terms by length descending so multi-word terms match first.
            sorted_terms = sorted(self.MEDICAL_TERMS.keys(), key=len, reverse=True)

            highlighted = text
            # Track positions already highlighted to avoid double-wrapping.
            highlighted_spans: List[tuple] = []

            for term in sorted_terms:
                pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
                for match in pattern.finditer(text):
                    start, end = match.span()
                    # Skip if this span overlaps an already highlighted span.
                    if any(start < hs_end and end > hs_start for hs_start, hs_end in highlighted_spans):
                        continue

                    entry = self.MEDICAL_TERMS[term]
                    category = entry.get("category", "medical")
                    term_suggestions = entry.get("suggestions", [])

                    identified.append(
                        {
                            "term": match.group(0),
                            "normalized": term,
                            "category": category,
                            "start": start,
                            "end": end,
                        }
                    )

                    if term_suggestions:
                        suggestions[term] = term_suggestions

                    # Mark terms that sound commonly misrecognized by ASR.
                    # Heuristic: shorter single-word terms or terms with
                    # homophones are flagged for review.
                    is_uncertain = len(term.split()) == 1 and len(term) <= 7
                    if is_uncertain:
                        uncertain.append(
                            {
                                "term": match.group(0),
                                "normalized": term,
                                "category": category,
                                "reason": "Short or commonly misrecognized term - verify with doctor",
                                "suggestions": term_suggestions,
                            }
                        )

                    # Wrap in a highlight marker (kept as plain text markers).
                    highlighted_spans.append((start, end))

            # Build highlighted transcript by inserting markers from right to left.
            highlighted_chars = list(text)
            for start, end in sorted(highlighted_spans, key=lambda s: s[0], reverse=True):
                highlighted_chars.insert(end, "</term>")
                highlighted_chars.insert(start, "<term>")
            result.highlighted_transcript = "".join(highlighted_chars)

            result.success = True
            result.message = (
                f"Identified {len(identified)} medical term(s); "
                f"{len(uncertain)} flagged for verification."
            )
            result.identified_terms = identified
            result.uncertain_terms = uncertain
            result.suggestions = suggestions
            result.metadata = {
                "total_terms_found": len(identified),
                "uncertain_count": len(uncertain),
                "dictionary_size": len(self.MEDICAL_TERMS),
            }
            self.status = "completed"
        except Exception as exc:  # pragma: no cover - runtime safety
            self.status = "error"
            result.success = False
            result.message = f"Terminology analysis failed: {exc}"

        self.last_result = result
        return result

    def strip_markers(self, highlighted_text: str) -> str:
        """Remove <term>/</term> markers, returning plain text."""
        return highlighted_text.replace("<term>", "").replace("</term>", "")
