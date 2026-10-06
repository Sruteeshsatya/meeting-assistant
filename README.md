# 🎙️ AI Meeting Assistant

Turns a recorded meeting into an accurate transcript and a usable written record through a
three-stage pipeline:

1. **Speech-to-text:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper `large-v3` on GPU) produces the **raw transcript**.
2. **Transcript refinement (LLM #1):** Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) corrects misrecognised domain terms and produces the **refined transcript**.
3. **Meeting documentation (LLM #2):** Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`, a separate call with its own prompt) writes the **summary, minutes, key decisions and action items**.

Everything is generated live from the uploaded audio. Nothing is prewritten. Owners and deadlines
that were not stated in the recording are shown as **Unspecified**, never guessed.

See [`TECHNICAL.md`](TECHNICAL.md) for the model roles and how data moves between stages.

---

## Quick start: Google Colab (recommended, free GPU)

1. Get a free Gemini API key at <https://aistudio.google.com/apikey>.
2. Open `notebooks/colab_launch.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
3. In Colab's 🔑 **Secrets** panel, add `GEMINI_API_KEY` and enable notebook access.
4. Run the cells in order. The last setup cell prints a public `trycloudflare.com` link to the app.

## Run locally

Requirements: Python 3.10+ and internet access the first time (to download the Whisper model).
`ffmpeg` on your PATH is used if present; otherwise audio is decoded with the bundled PyAV library.

**Windows (PowerShell or Command Prompt)**

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
streamlit run app.py
```

In Notepad, replace `your-key-from-aistudio.google.com` with your real key and save.

**macOS / Linux**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                   # then put your GEMINI_API_KEY in .env
streamlit run app.py
```

> Start the app with `streamlit run app.py`, not `python app.py`.

On a CPU-only machine, `WHISPER_MODEL=auto` picks `small.en`, which is fast but less accurate. Set
`WHISPER_MODEL=medium.en` or use Colab for better accuracy.

### Command line

```bash
python cli.py path/to/meeting.mp3 --context "Kubernetes, PyTorch, Priya, Arjun"
```

All outputs are saved under `runs/<timestamp>_<file>/`.

## Using the app

1. *(Optional)* In the sidebar, enter **meeting context**: domain terms, product names and people's
   names. This helps the models spell them correctly. It is never used as a source of facts.
2. Upload a recording (`wav, mp3, m4a, aac, ogg, opus, flac, webm, mp4`, up to 500 MB).
3. Click **Process meeting** and watch each stage's status.
4. Inspect the tabs:
   - **Summary & minutes**
   - **Decisions**
   - **Action items**
   - **Transcripts:** raw and refined side by side, with changed lines highlighted and a table of every proposed correction, including rejected ones
   - **Downloads**

### Outputs (per recording)

| File | Contents |
|---|---|
| `raw_transcript.txt` | Speech-to-text result before any language-model refinement (timestamped) |
| `refined_transcript.txt` | Transcript after domain-aware terminology correction |
| `meeting_record.md` | **Human-readable** record: summary, minutes, decisions, action items, open items |
| `meeting_record.json` | **Machine-readable** record with the same decisions and tasks |
| `refinement_changes.json` | Audit log of every correction (applied or rejected, with reasons) |

The Markdown file is rendered *from* the JSON record, so the two formats always contain identical
decisions and tasks.

### Error handling

The app shows a clear message, naming the stage that failed, for:
- unsupported file types
- empty files
- corrupt or unreadable audio
- recordings that are too short, silent, or contain no speech
- Whisper model download failures
- a missing or invalid API key
- an unknown model name
- Gemini rate limits (after automatic retries)

Results produced before a failure, such as the raw transcript, remain visible.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | Required. Gemini API key |
| `REFINE_MODEL` | `gemini-3.5-flash-lite` | Stage 2 model |
| `DOCS_MODEL` | `gemini-3.5-flash-lite` | Stage 3 model |
| `WHISPER_MODEL` | `auto` | `auto` → `large-v3` (GPU) / `small.en` (CPU); any faster-whisper model name |
| `STT_USE_GLOSSARY` | `1` | Pass the meeting-context terms to Whisper as a vocabulary hint |

All three models can also be changed in the sidebar. If a Gemini model name stops working,
`gemini-flash-latest` is a safe alternative.

## Tests

```bash
python -m pytest -q
```

The 33 tests run offline using fake models. They cover:
- audio validation errors
- the number and negation guardrails
- refinement and grounding logic
- Gemini retry and error handling
- speech-to-text GPU-to-CPU fallback
- an end-to-end pipeline run
- consistency between the Markdown and JSON exports

## Project layout

```
app.py                      Streamlit interface
cli.py                      Command-line runner
meeting_assistant/
  audio.py                  Stage 1a: file validation and decoding (ffmpeg)
  transcribe.py             Stage 1b: faster-whisper speech-to-text
  refine.py                 Stage 2: refinement (LLM #1) and safe application of edits
  document.py               Stage 3: documentation (LLM #2) and grounding checks
  guardrails.py             Number/negation checks, evidence matching, owner/deadline verification
  llm.py                    Gemini client (structured JSON output, retries)
  pipeline.py               Orchestrator: runs the stages in order, reports status
  render.py                 Markdown/JSON/TXT exports and zip bundle
  schema.py                 Pydantic data models
prompts/
  refine_system.md          System prompt for LLM #1
  document_system.md        System prompt for LLM #2
notebooks/colab_launch.ipynb
samples/meeting_script.md   Script for the shareable sample recording
tests/
```

## Troubleshooting

- **Colab crashes with a `libcudnn` error during transcription:** run
  `!pip install -q nvidia-cudnn-cu12==9.*`, then restart the runtime. Alternatively, set
  `WHISPER_MODEL=small.en` to use the CPU.
- **"Gemini rate limit or quota exceeded":** the free tier has per-minute limits. Wait a minute, or
  switch both stages to `gemini-flash-latest` in the sidebar.
- **Model download fails:** the first run downloads Whisper weights (about 3 GB for `large-v3`) from Hugging Face.
