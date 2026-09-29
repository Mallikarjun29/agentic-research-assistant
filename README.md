# Jan-v1 Deep Research Assistant

A local Streamlit demo that plans research queries, asks `janhq/Jan-v1-4B` to draft scout briefs, and synthesizes a sectioned report. Scout briefs are **model-generated**, not retrieved from the web or verified sources. The app does not browse, cite live evidence, or validate the facts it generates; check claims independently before relying on them.

![Streamlit research assistant](https://github.com/user-attachments/assets/5b1b2305-a7bc-4d8e-9a14-73a03753d1db)

## Architecture

```text
Topic + depth / focus / timeframe / format (Streamlit app.py)
  -> deterministic query planning (generate_search_queries)
  -> Jan-v1 scout briefs for each query (build_internal_sources)
  -> Jan-v1 report synthesis from up to 20 briefs (synthesize_research)
  -> <final> extraction and section cleanup (utils.py)
  -> report display + TXT/JSON downloads (app.py)
```

`jan_assistant/assistant.py` loads the Hugging Face tokenizer and causal model, generates template-based queries (5, 8, 15, or 25 depending on depth), prompts the model for a title/summary per query, then feeds those notes into a format-specific report prompt. There is **no external search or retrieval call**. `sources` in the UI/JSON are internal model notes with empty URLs, not citations. The Streamlit app keeps the loaded model and latest run in session state; it does not save a persistent run history. Timeframe is passed to the scout prompt, not enforced by a date-filtered search.

## Setup

- Python 3.10+ and enough RAM/VRAM for the model. The default `ResearchConfig` uses `device_map="auto"` and float16; an NVIDIA GPU with a compatible PyTorch build is the practical route. CPU operation is not tested here.
- Network access for the initial Hugging Face model download. Later use can rely on cached weights, but the app does not explicitly set an offline mode.

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install a PyTorch build appropriate for your machine first (see pytorch.org).
pip install -r requirements.txt
streamlit run app.py
```

Set `HF_TOKEN` in your shell if Hugging Face authentication or rate limits require it, for example `export HF_TOKEN=...`. No `.env.example` is checked in, and the code reads `os.getenv("HF_TOKEN")` directly: creating a `.env` file alone will not load it. Do not commit tokens. `jan_assistant/config.py` contains the model ID, generation settings, dtype and device map; edit those defaults if needed. The first launch downloads `janhq/Jan-v1-4B` into the Hugging Face cache.

Enter a topic, choose depth/focus/timeframe/report format, and click **Start Deep Research**. The app displays generated notes under "Sources" and offers TXT (report) and JSON (report, query list, notes and run metadata) downloads. Those notes are not evidence-backed sources. Review the generated report before sharing it.

## Repository layout

- `app.py`: Streamlit UI, progress, session state and exports.
- `jan_assistant/assistant.py`: model loading, query planning, scout prompts and synthesis.
- `jan_assistant/config.py`: `ResearchConfig` defaults and `HF_TOKEN` environment lookup.
- `jan_assistant/utils.py`: section templates and final-block extraction.
- `requirements.txt`: Python dependencies; install a hardware-compatible PyTorch build separately if required.

If model loading fails, inspect the console error, confirm the model ID/token and check `python -c "import torch; print(torch.cuda.is_available())"`. If memory is tight, reduce `max_tokens` in `ResearchConfig` or use a machine with more memory.
