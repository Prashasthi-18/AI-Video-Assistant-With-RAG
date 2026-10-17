"""
Gets a transcript the fastest way available:
  1. YouTube link with captions  → use the captions (seconds, no audio download)
  2. Otherwise download/convert the audio, then transcribe it:
       hinglish → Sarvam AI · english → Groq cloud Whisper → local faster-whisper fallback
"""
import wave

from utils.audio_processor import process_input
from utils.captions import fetch_youtube_captions
from core.transcriber import transcribe


def is_youtube(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def audio_seconds(chunks: list) -> float:
    total = 0.0
    for c in chunks:
        try:
            with wave.open(c) as w:
                total += w.getnframes() / w.getframerate()
        except Exception:
            pass
    return total


def prepare_source(source: str, language: str = "english") -> dict:
    """Step 1: captions if available, otherwise audio chunks ready for speech-to-text."""
    if is_youtube(source):
        print("Checking for YouTube captions...")
        captions = fetch_youtube_captions(source, language)
        if captions:
            print(f"Using {captions['engine']} — skipping audio transcription.")
            return {"captions": captions}
        print("No usable captions — transcribing the audio instead.")
    chunks = process_input(source)
    return {"chunks": chunks, "duration": audio_seconds(chunks)}


def get_transcript(prepared: dict, language: str = "english") -> dict:
    """Step 2: returns {"text", "engine", "duration"}."""
    if "captions" in prepared:
        c = prepared["captions"]
        return {"text": c["text"], "engine": c["engine"], "duration": c["duration"]}
    text, engine = transcribe(prepared["chunks"], language)
    return {"text": text, "engine": engine, "duration": prepared["duration"]}
