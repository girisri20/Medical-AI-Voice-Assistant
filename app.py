"""app.py — Medical AI Voice Assistant (Streamlit dashboard).

Agent-Based Medical Speech Recognition System — demonstration prototype.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from typing import Optional

import streamlit as st

# Allow running both as `streamlit run app.py` from the project folder and
# as `python app.py` by ensuring the project root is on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agents.orchestrator import AgentOrchestrator  # noqa: E402


# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical AI Voice Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: healthcare dashboard, white background, blue & teal accents.
st.markdown(
    """
    <style>
    :root {
        --primary: #0b6ba4;      /* deep blue */
        --teal: #1b9aaa;         /* teal accent */
        --bg: #ffffff;
        --card: #f4f8fb;
        --muted: #6b7a8d;
        --border: #dbe7ee;
    }
    .block-container { padding-top: 1.2rem; }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.3px;
    }
    .subtitle {
        font-size: 1.05rem;
        color: var(--muted);
        margin: 0 0 1.5rem 0;
    }
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        color: var(--primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        margin-left: auto;
    }
    .badge-idle { background: #e9ecef; color: #495057; }
    .badge-running { background: #fff3cd; color: #856404; }
    .badge-completed { background: #d4edda; color: #155724; }
    .badge-error { background: #f8d7da; color: #721c24; }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--teal);
        margin: 1.5rem 0 0.75rem 0;
    }
    .disclaimer {
        background: #fffdf3;
        border: 1px dashed #e2c86b;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #7a6a14;
        margin-top: 1rem;
    }
    .term-chip {
        display: inline-block;
        background: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 12px;
        padding: 0.2rem 0.6rem;
        font-size: 0.8rem;
        color: #0d47a1;
        margin: 0.2rem 0.4rem 0.2rem 0;
    }
    .uncertain-chip {
        background: #fff3e0;
        border: 1px solid #ffb74d;
        color: #e65100;
    }
    .transcript-box {
        background: #ffffff;
        color: #111827;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.9rem;
        line-height: 1.55;
        white-space: pre-wrap;
        max-height: 420px;
        overflow-y: auto;
    }
    .stButton>button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: var(--teal);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "orchestrator": None,
        "result": None,
        "audio_bytes": None,
        "uploaded_file_path": None,
        "transcript_text": "",
        "report_text": "",
        "json_report": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🩺 Medical AI Voice Assistant")
        st.markdown("Agent-Based Speech Recognition System")
        st.divider()

        st.markdown("### Project info")
        st.markdown(
            "- **Frontend:** Streamlit\n"
            "- **Speech recognition:** faster-whisper (OpenAI Whisper)\n"
            "- **Audio:** SoundFile / FFmpeg\n"
            "- **Agents:** 6 modular Python agents\n"
            "- **Storage:** In-memory only\n"
            "- **Language:** English"
        )
        st.divider()

        st.markdown("### Medical safety")
        st.warning(
            "Demonstration prototype only. "
            "Not for clinical diagnosis, prescribing, or treatment decisions. "
            "Verify all output with a qualified doctor."
        )
        st.divider()

        st.markdown("### Agent pipeline")
        for name in [
            "1. Audio Processing",
            "2. Speech-to-Text",
            "3. Medical Terminology",
            "4. Clinical Extraction",
            "5. Transcript Formatting",
        ]:
            st.markdown(f"- {name}")


# ---------------------------------------------------------------------------
# Status badge
# ---------------------------------------------------------------------------
def _status_badge(status: str) -> str:
    mapping = {
        "pending": "badge badge-idle",
        "running": "badge badge-running",
        "completed": "badge badge-completed",
        "error": "badge badge-error",
    }
    cls = mapping.get(status, "badge badge-idle")
    return f'<span class="{cls}">{status.upper()}</span>'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _render_sidebar()

    st.markdown(
        '<h1 class="main-title">🩺 Medical AI Voice Assistant</h1>'
        '<p class="subtitle">Agent-Based Medical Speech Recognition System</p>',
        unsafe_allow_html=True,
    )

    # Orchestrator (create once per session)
    if st.session_state.orchestrator is None:
        st.session_state.orchestrator = AgentOrchestrator(
            whisper_model="base", device="cpu"
        )

    orch: AgentOrchestrator = st.session_state.orchestrator

    # ------------------------------------------------------------------
    # Audio input section
    # ------------------------------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card h3">🎙️ Audio Input</h3>', unsafe_allow_html=True)

    col_mic, col_file = st.columns(2)

    with col_mic:
        st.markdown("**Record from microphone**")
        mic_audio = st.audio_input("Record your voice", key="mic_input")
        if mic_audio is not None:
            # st.audio_input returns a file-like object; read bytes.
            try:
                st.session_state.audio_bytes = mic_audio.read()
                st.session_state.uploaded_file_path = None
                st.success("Microphone audio captured.")
            except Exception as exc:
                st.error(f"Failed to read microphone audio: {exc}")

    with col_file:
        st.markdown("**Or upload an audio file**")
        uploaded = st.file_uploader(
            "Upload WAV / MP3 / M4A / FLAC / OGG",
            type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
            key="file_upload",
        )
        if uploaded is not None:
            # Save uploaded file to a temp path for faster-whisper.
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                f"medical_demo_upload_{uploaded.name}",
            )
            try:
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.session_state.uploaded_file_path = tmp_path
                st.session_state.audio_bytes = None
                st.success(f"Uploaded: {uploaded.name}")
            except Exception as exc:
                st.error(f"Failed to save upload: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Process button
    # ------------------------------------------------------------------
    process_disabled = (
        st.session_state.audio_bytes is None
        and st.session_state.uploaded_file_path is None
    )

    if st.button("🔊 Process Audio", disabled=process_disabled, type="primary"):
        # Reset prior results
        st.session_state.result = None
        st.session_state.transcript_text = ""
        st.session_state.report_text = ""
        st.session_state.json_report = {}

        progress = st.progress(0, text="Starting pipeline...")
        status_text = st.empty()

        try:
            if st.session_state.audio_bytes is not None:
                result = orch.process_microphone_audio(
                    st.session_state.audio_bytes
                )
            else:
                result = orch.process_uploaded_audio(
                    st.session_state.uploaded_file_path
                )

            st.session_state.result = result

            # Reflect step statuses in UI via progress bar
            steps = result.steps
            for i, step in enumerate(steps):
                progress.progress((i + 1) / len(steps), text=step.name)
                status_text.info(f"{step.name}: {step.message}")

            if result.success:
                st.session_state.transcript_text = result.original_transcript
                st.session_state.report_text = result.final_report
                st.session_state.json_report = result.json_report
                st.success("Pipeline completed successfully.")
            else:
                st.error("Pipeline encountered errors. See agent status below.")

        except Exception as exc:
            st.error(f"Runtime error: {exc}")

        progress.empty()
        status_text.empty()

    # ------------------------------------------------------------------
    # Agent execution dashboard
    # ------------------------------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card h3">🤖 Agent Execution Dashboard</h3>', unsafe_allow_html=True)

    if st.session_state.result is None:
        st.info("No processing run yet. Provide audio and click **Process Audio**.")
    else:
        result: object = st.session_state.result
        steps = getattr(result, "steps", [])
        # Render status cards in a row
        cols = st.columns(len(steps) if steps else 1)
        for idx, step in enumerate(steps):
            with cols[idx]:
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<div class="badge {step.status}">{step.status.upper()}</div>'
                    f'<div style="font-size:0.8rem; margin-top:0.25rem;">{step.name}</div>'
                    f'<div style="font-size:0.75rem; color:#6b7a8d;">{step.message}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Results sections
    # ------------------------------------------------------------------
    if st.session_state.result is not None:
        result = st.session_state.result

        # Original transcript
        st.markdown('<div class="section-title">📝 Original Speech Transcript</div>', unsafe_allow_html=True)
        transcript = st.session_state.transcript_text or getattr(result, "original_transcript", "") or ""
        st.markdown(
            f'<div class="transcript-box">{transcript}</div>',
            unsafe_allow_html=True,
        )

        # Medical terminology
        st.markdown('<div class="section-title">🏷️ Medical Terminology Extraction</div>', unsafe_allow_html=True)
        term_result = getattr(result, "terminology_result", None)
        if term_result and term_result.success:
            terms = getattr(term_result, "identified_terms", []) or []
            uncertain = getattr(term_result, "uncertain_terms", []) or []
            suggestions = getattr(term_result, "suggestions", {}) or {}

            c1, c2, c3 = st.columns(3)
            c1.metric("Terms identified", len(terms))
            c2.metric("Flagged for review", len(uncertain))
            c3.metric("Suggestions available", len(suggestions))

            if terms:
                st.markdown("**Identified terms:**")
                chips = "".join(
                    f'<span class="term-chip">{t.get("term","")} '
                    f'<small>[{t.get("category","medical")}]</small></span>'
                    for t in terms
                )
                st.markdown(chips, unsafe_allow_html=True)

            if uncertain:
                st.markdown("**⚠️ Terms flagged for doctor verification:**")
                for u in uncertain:
                    sugg = ", ".join(u.get("suggestions", [])) or "none"
                    st.markdown(
                        f'<span class="term-chip uncertain-chip">{u.get("term","")}</span> '
                        f'<small>{u.get("reason","")} — suggestions: {sugg}</small>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Terminology results unavailable.")

        # Clinical information
        st.markdown('<div class="section-title">🔬 Clinical Information Extraction</div>', unsafe_allow_html=True)
        clin_result = getattr(result, "clinical_result", None)
        if clin_result and clin_result.success:
            def _kv(label: str, items: list) -> None:
                items = items or []
                st.markdown(f"**{label}**")
                if items:
                    for it in items:
                        st.markdown(f"- {it}")
                else:
                    st.markdown("*Not mentioned*")

            col_a, col_b = st.columns(2)
            with col_a:
                _kv("Symptoms", getattr(clin_result, "symptoms", []) or [])
                _kv("Symptom durations", getattr(clin_result, "symptom_durations", []) or [])
                _kv("Conditions / diseases", getattr(clin_result, "conditions", []) or [])
                _kv("Medicines", getattr(clin_result, "medicines", []) or [])
            with col_b:
                _kv("Dosage / frequency", getattr(clin_result, "dosages", []) or [])
                _kv("Tests / investigations", getattr(clin_result, "tests", []) or [])
                _kv("Doctor's recommendations", getattr(clin_result, "recommendations", []) or [])
                _kv("Follow-up instructions", getattr(clin_result, "follow_up", []) or [])

            missing = getattr(clin_result, "missing_fields", []) or []
            if missing:
                st.markdown("**⚠️ Information not mentioned (flagged for review):**")
                for m in missing:
                    st.markdown(f"- {m}")
        else:
            st.info("Clinical extraction results unavailable.")

        # Final formatted report
        st.markdown('<div class="section-title">📄 Final Formatted Consultation Report</div>', unsafe_allow_html=True)
        report = st.session_state.report_text or getattr(result, "final_report", "") or ""
        st.markdown(
            f'<div class="transcript-box">{report}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="disclaimer">'
                    'This report is generated by a demonstration AI transcription system. '
                    'It does not constitute medical advice, diagnosis, or prescription. '
                    'All content must be reviewed and verified by a qualified doctor.'
                    '</div>', unsafe_allow_html=True)

        # Downloads
        st.markdown("### ⬇️ Download transcript")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download TXT",
                data=report,
                file_name=f"medical_consultation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
        with col_dl2:
            json_data = json.dumps(
                getattr(result, "json_report", {}) or {},
                indent=2,
                ensure_ascii=False,
            )
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name=f"medical_consultation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
