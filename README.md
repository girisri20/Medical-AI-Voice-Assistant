# Medical AI Voice Assistant — Agent-Based Medical Speech-to-Text Demo

> **IMPORTANT:** This is a demonstration prototype only. It is NOT a
> production-ready medical application. It does not diagnose, prescribe, or
> replace clinical judgment. All transcribed medical information must be
> verified by a qualified doctor.

## Project Structure

```
medical_speech_demo/
├── app.py                      # Streamlit dashboard
├── install.bat                 # One-time dependency setup (run once)
├── run.bat                     # Launch the app (run every time)
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # This file
├── .streamlit/
│   └── config.toml             # Healthcare theme (white bg, blue/teal accents)
├── agents/
│   ├── __init__.py
│   ├── audio_agent.py          # Agent 1: Audio Processing
│   ├── transcription_agent.py  # Agent 2: Speech-to-Text (faster-whisper)
│   ├── medical_terminology_agent.py  # Agent 3: Medical Terminology (rule-based)
│   ├── clinical_extraction_agent.py  # Agent 4: Clinical Information Extraction (rule-based)
│   ├── formatting_agent.py     # Agent 5: Transcript Formatting
│   └── orchestrator.py         # Agent 6: Agent Orchestrator
└── utils/
    ├── __init__.py
    └── audio_utils.py          # Audio validation / I/O helpers
```

## Quick Start (Windows CMD)

```cmd
REM 1. Open the project folder
cd "D:\DATA SCIENCE COURSE\OpenCode\medical_speech_demo"

REM 2. Install dependencies (run ONCE)
install.bat

REM 3. Start the app (run every time)
run.bat
```

The app opens at **http://localhost:8501**.

## Why install.bat / run.bat?

Previously the app seemed to "need reinstalling every time" because Streamlit was being launched from a different Python environment than the one where packages were installed. This kit fixes that:

- `install.bat` installs `streamlit`, `soundfile`, `numpy`, `ctranslate2`, `faster-whisper` (and friends) into the default Python environment once.
- `run.bat` launches `python -m streamlit run app.py` from that same environment, so `faster-whisper` is always found.

No virtual environment juggling, no manual reinstalls on code changes.

> **Python 3.15 note:** `onnxruntime` (a `faster-whisper` dependency) has no wheel for Python 3.15. The app disables the VAD filter, so `onnxruntime` is not needed at runtime. `install.bat` therefore installs `faster-whisper --no-deps`. On Python 3.10–3.12 a normal `pip install -r requirements.txt` works too.

## How to test transcription

1. Click **Process Audio** after providing input.
2. **Microphone:** allow browser microphone permission, record your voice, then click Process Audio.
3. **Upload:** upload a WAV/MP3/M4A/FLAC/OGG file containing a doctor speaking about a case, then click Process Audio.
4. Watch the **Agent Execution Dashboard** — each agent runs in sequence and its status updates.
5. Review the **Original Transcript**, **Medical Terminology**, **Clinical Information**, and the **Final Formatted Report**.
6. Download the report as **TXT** or **JSON**.

## Agent communication

The **Agent Orchestrator** runs agents sequentially, passing each agent's output to the next:

```
Doctor's Voice
    ↓
Agent 1 Audio Processing Agent        → validated audio file / samples
    ↓
Agent 2 Speech-to-Text Agent          → transcript (faster-whisper)
    ↓
Agent 3 Medical Terminology Agent     → flagged terms + suggestions
    ↓
Agent 4 Clinical Information Extraction Agent → symptoms, medicines, tests, etc.
    ↓
Agent 5 Transcript Formatting Agent   → formatted consultation report
    ↓
Final Medical Consultation Transcript
```

Each agent returns a result object consumed by the next agent. The orchestrator maintains the status of every step and surfaces it in the dashboard.

## Medical safety

- Transcription and organization tool only — no diagnosis.
- No medicines prescribed, no dosages changed.
- No symptoms or medical information fabricated.
- Original transcript preserved verbatim.
- Uncertain medical terms flagged for doctor review.
- All information must be verified by a qualified doctor.

## Notes

- The faster-whisper model is downloaded automatically on first run and cached locally. No paid API or cloud service is required.
- Terminology and extraction agents are rule-based (keyword / regex) for this demo. They are not autonomous LLM agents.
- Code changes to `app.py` or agents take effect on refresh — Streamlit hot-reloads automatically; you do **not** need to reinstall.
