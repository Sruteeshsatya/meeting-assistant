"""Meeting assistant UI. Run: streamlit run app.py"""

from __future__ import annotations

import html
import os
import sys

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx(suppress_warning=True) is None:  # not run via streamlit
    print("\nThis is a Streamlit app. Start it with:\n\n    streamlit run app.py\n")
    sys.exit(1)

from meeting_assistant.config import RUNS_DIR, SUPPORTED_EXTENSIONS, Settings
from meeting_assistant.errors import LLMError
from meeting_assistant.pipeline import STAGES, PipelineResult, run_pipeline
from meeting_assistant.render import build_zip, transcript_to_text
from meeting_assistant.schema import UNSPECIFIED, format_time

st.set_page_config(page_title="Meeting Assistant", page_icon="🎙️", layout="wide")

STATUS_ICONS = {"pending": "⚪", "running": "⏳", "done": "✅", "failed": "❌"}


# Cached models
@st.cache_resource(show_spinner=False)
def get_transcriber(model_name: str):
    from meeting_assistant.transcribe import WhisperTranscriber

    return WhisperTranscriber(model_name)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    from meeting_assistant.llm import GeminiClient

    return GeminiClient(api_key)


def key_from_secrets() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


# Sidebar
settings = Settings()
with st.sidebar:
    st.header("Settings")
    api_key = settings.gemini_api_key or key_from_secrets()
    if api_key:
        st.success("Gemini API key loaded", icon="🔑")
    else:
        api_key = st.text_input("Gemini API key", type="password", help="Get one free at aistudio.google.com/apikey")
    settings.gemini_api_key = api_key

    st.subheader("Models")
    whisper_options = ["auto", "large-v3", "large-v3-turbo", "medium.en", "small.en", "base.en"]
    default_whisper = settings.whisper_model if settings.whisper_model in whisper_options else "auto"
    settings.whisper_model = st.selectbox(
        "Stage 1 · Speech-to-text (faster-whisper)", whisper_options, index=whisper_options.index(default_whisper),
        help="'auto' uses large-v3 on a GPU and small.en on CPU.",
    )
    settings.refine_model = st.text_input("Stage 2 · Refinement LLM (Gemini)", settings.refine_model)
    settings.docs_model = st.text_input("Stage 3 · Documentation LLM (Gemini)", settings.docs_model)

    st.subheader("Meeting context (optional)")
    context = st.text_area(
        "Domain terms, product names or people's names",
        placeholder="e.g. Kubernetes, PyTorch, CI/CD, Priya, Project Atlas",
        help="Helps both speech-to-text and refinement spell domain terms correctly. "
             "It is never used as a source of facts for the minutes.",
        height=110,
    )
    settings.use_glossary_hint = st.checkbox("Also give these terms to speech-to-text", value=settings.use_glossary_hint)

# Upload
st.title("🎙️ AI Meeting Assistant")
st.caption(
    "Upload a meeting recording → **speech-to-text** → **domain-aware transcript refinement** → "
    "**minutes, decisions and action items**. Every result is generated live from your audio."
)

uploaded_file = st.file_uploader(
    "Meeting recording (English)",
    help=f"Supported formats: {', '.join(sorted(e.lstrip('.') for e in SUPPORTED_EXTENSIONS))}",
)
process_clicked = st.button("▶ Process meeting", type="primary", disabled=uploaded_file is None)


# Processing
def process_upload(file) -> PipelineResult:
    stage_states = {stage: "pending" for stage in STAGES}
    with st.status("Processing meeting…", expanded=True) as status_box:
        stage_list = st.empty()
        progress_bar = st.progress(0.0, text="Starting")

        def draw_stage_list():
            stage_list.markdown(
                "\n".join(f"{STATUS_ICONS[stage_states[stage]]} {label}" for stage, label in STAGES.items())
            )

        def on_stage(stage: str, state: str):
            stage_states[stage] = state
            draw_stage_list()
            if state == "running":
                progress_bar.progress(0.0, text=STAGES[stage])
                if stage == "transcribe":
                    progress_bar.progress(0.0, text="Loading speech-to-text model (first run downloads it)…")

        def on_progress(stage: str, fraction: float, message: str):
            progress_bar.progress(min(max(fraction, 0.0), 1.0), text=f"{STAGES[stage]} — {message}")

        def make_llm():
            if not settings.gemini_api_key:
                raise LLMError("No Gemini API key configured. Enter it in the sidebar or set GEMINI_API_KEY.")
            return get_llm(settings.gemini_api_key)

        draw_stage_list()
        outcome = run_pipeline(
            file.getvalue(), file.name,
            stt=lambda: get_transcriber(settings.whisper_model),
            llm=make_llm,
            settings=settings,
            context=context,
            on_stage=on_stage,
            on_progress=on_progress,
            save_dir=RUNS_DIR,
        )
        if outcome.ok:
            progress_bar.progress(1.0, text="Done")
            total_seconds = sum(outcome.timings.values())
            status_box.update(label=f"Finished in {total_seconds:.0f}s", state="complete", expanded=False)
        else:
            status_box.update(label=f"Failed during: {STAGES[outcome.failed_stage]}", state="error", expanded=True)
    return outcome


if process_clicked and uploaded_file is not None:
    st.session_state["result"] = process_upload(uploaded_file)
    st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1

result: PipelineResult | None = st.session_state.get("result")
if result is None:
    st.info("Upload an audio file and press **Process meeting** to start.")
    st.stop()

# Errors
if result.error:
    st.error(f"**Could not finish processing `{result.filename}`.**\n\n"
             f"Stage: *{STAGES.get(result.failed_stage, result.failed_stage)}*\n\n{result.error.user_message}")
    if result.error.detail:
        with st.expander("Technical details", expanded=True):
            st.code(result.error.detail)
    if result.raw is None:
        st.stop()
    st.warning("Showing the results produced before the failure.")

run_id = st.session_state.get("run_id", 0)  # unique button keys
record = result.record

# Results
if record:
    st.subheader(record.title)
    duration_col, decisions_col, actions_col, fixes_col = st.columns(4)
    duration_col.metric("Duration", record.metadata.duration)
    decisions_col.metric("Decisions", len(record.decisions))
    actions_col.metric("Action items", len(record.action_items))
    fixes_col.metric("Transcript fixes", sum(c.applied for c in result.corrections))
if result.raw is not None and "fallback" in result.raw.model:
    st.warning(
        "The GPU failed, so speech-to-text ran on the CPU with a smaller, less accurate model "
        f"(`{result.raw.model}`). Restart the Colab runtime to try the GPU again.", icon="⚠️",
    )

tab_names = (["📝 Summary & minutes", "✅ Decisions", "📌 Action items"] if record else []) + ["🗣️ Transcripts", "⬇️ Downloads"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

if record:
    with tabs["📝 Summary & minutes"]:
        st.markdown("#### Summary")
        st.write(record.summary)
        if record.participants_mentioned:
            st.caption("Names mentioned: " + ", ".join(record.participants_mentioned))
        st.markdown("#### Minutes")
        if not record.minutes:
            st.write("_No minutes generated._")
        for topic in record.minutes:
            st.markdown(f"**{topic.topic}**")
            st.markdown("\n".join(f"- {point}" for point in topic.points))
        if record.open_items:
            st.markdown("#### Open items (raised but not agreed)")
            st.markdown("\n".join(
                f"- {item.item}" + (f" `[{item.timestamp}]`" if item.timestamp else "") for item in record.open_items
            ))

    with tabs["✅ Decisions"]:
        if not record.decisions:
            st.info("No decisions were reached in this meeting.")
        for decision in record.decisions:
            with st.container(border=True):
                st.markdown(f"**{decision.id}. {decision.decision}**")
                time_label = f" — `{decision.timestamp}`" if decision.timestamp else ""
                st.caption(f"“{decision.evidence}”{time_label}")
                if not decision.evidence_verified:
                    st.warning("Supporting quote could not be matched to the transcript — please verify.", icon="⚠️")

    with tabs["📌 Action items"]:
        if not record.action_items:
            st.info("No action items were agreed in this meeting.")
        else:
            st.dataframe(
                [
                    {
                        "#": item.id, "Task": item.task, "Owner": item.owner, "Deadline": item.deadline,
                        "Evidence": f"“{item.evidence}”", "Time": item.timestamp or "",
                    }
                    for item in record.action_items
                ],
                hide_index=True, width="stretch",
            )
            st.caption(f"“{UNSPECIFIED}” means the recording did not state an owner or deadline — nothing is guessed.")
            for item in record.action_items:
                for note in item.notes:
                    st.caption(f"{item.id}: {note}")

with tabs["🗣️ Transcripts"]:
    changed_ids = {c.segment_id for c in result.corrections if c.applied}
    left, right = st.columns(2)
    left.markdown("#### Raw transcript")
    left.caption(result.raw.model)
    with left.container(height=500):
        st.text(result.raw.as_timestamped_text())
    right.markdown("#### Refined transcript")
    right.caption("Highlighted lines were changed by the refinement model.")
    if result.refined:
        with right.container(height=500):
            rows = []
            for segment in result.refined.segments:
                line = f"[{format_time(segment.start)}] {html.escape(segment.text)}"
                if segment.id in changed_ids:
                    line = f"<mark>{line}</mark>"
                rows.append(f"<div style='font-family:monospace;font-size:0.85rem;margin-bottom:2px'>{line}</div>")
            st.markdown("".join(rows), unsafe_allow_html=True)
    else:
        right.info("Refinement did not complete.")

    if result.corrections:
        st.markdown("#### Changes proposed by the refinement model")
        st.dataframe(
            [
                {
                    "Time": c.timestamp, "Original": c.original, "Corrected": c.corrected, "Reason": c.reason,
                    "Status": "Applied" if c.applied else f"Rejected: {c.rejected_because}",
                }
                for c in result.corrections
            ],
            hide_index=True, width="stretch",
        )
    elif result.refined:
        st.caption("The refinement model found no recognition errors to correct.")

with tabs["⬇️ Downloads"]:
    files = result.outputs
    if not files:
        files = {"raw_transcript.txt": transcript_to_text(result.raw, "RAW TRANSCRIPT")}
    st.download_button(
        "⬇️ Download everything (.zip)", build_zip(files), file_name="meeting_outputs.zip",
        mime="application/zip", type="primary", key=f"zip-{run_id}",
    )
    button_labels = {
        "raw_transcript.txt": "Raw transcript (.txt)",
        "refined_transcript.txt": "Refined transcript (.txt)",
        "meeting_record.md": "Meeting record — human-readable (.md)",
        "meeting_record.json": "Meeting record — machine-readable (.json)",
        "refinement_changes.json": "Refinement change log (.json)",
    }
    file_types = {".txt": "text/plain", ".md": "text/markdown", ".json": "application/json"}
    columns = st.columns(2)
    for index, (name, content) in enumerate(files.items()):
        columns[index % 2].download_button(
            button_labels.get(name, name), content, file_name=name,
            mime=file_types.get(os.path.splitext(name)[1], "text/plain"), key=f"{name}-{run_id}",
        )
    if "meeting_record.json" in files:
        with st.expander("Preview meeting_record.json"):
            st.code(files["meeting_record.json"], language="json")
