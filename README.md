# AI Video Assistant

Turn any YouTube video or recording into a transcript, a summary, action items, key decisions and open questions — then chat with it using RAG.

Based on the Sheryians AI School tutorial [AI Video Assistant With RAG | Full Project in Python](https://www.youtube.com/watch?v=xlYJhtL0qbQ), extended with a redesigned UI, user accounts, a personal library and faster transcription.

## Features

- **Fast transcripts** — YouTube captions when available (seconds), otherwise Groq cloud Whisper, with local faster-whisper as an offline fallback. Hindi / Hinglish goes to Sarvam AI.
- **Insights** — title, map-reduce summary, action items (with owner and deadline), key decisions and open questions, via Mistral + LangChain.
- **Chat with your video** — RAG over the transcript with ChromaDB and HuggingFace embeddings.
- **Accounts and library** — sign up / sign in (bcrypt), "keep me signed in" sessions, and a saved library of past analyses.
- **Export** — PDF and TXT reports.

## Tech stack

Python · Streamlit · LangChain · Mistral AI · ChromaDB · HuggingFace embeddings · faster-whisper · Groq · Sarvam AI · yt-dlp · SQLite

## Setup

Requirements: Python 3.12, [FFmpeg](https://ffmpeg.org/download.html) on your PATH, and Node.js (yt-dlp uses it to download from YouTube).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env            # then add your API keys
```

## Run

```bash
streamlit run app.py            # web app
python main.py                  # command-line version
```

## Project structure

```
app.py                  Streamlit app (auth, home, processing, results)
main.py                 CLI pipeline
core/
  transcript_source.py  Picks captions → Groq → local Whisper / Sarvam
  transcriber.py        Groq, faster-whisper and Sarvam transcription
  summarizer.py         Title + map-reduce summary
  extractor.py          Action items, decisions, open questions
  vector_store.py       ChromaDB + embeddings
  rag_engine.py         RAG chat chain
  llm.py                Shared Mistral client with rate limiting
  auth.py               Accounts and login sessions (SQLite + bcrypt)
  history.py            Per-user library of analyses
utils/
  audio_processor.py    YouTube download, conversion, chunking
  captions.py           YouTube caption fetching
  exporter.py           PDF / TXT export
ui/                     Styles, icons, auth page, cookie sessions, effects
```
