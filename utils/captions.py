"""
Fast path for YouTube: use the captions the video already has instead of transcribing audio.
Fetching captions takes a few seconds, versus minutes of speech-to-text on a CPU.
"""
import json
import re
import shutil
import yt_dlp

JS_RUNTIMES = {name: {} for name in ("deno", "node", "bun") if shutil.which(name)}


def _pick_track(info: dict, language: str):
    """
    Choose the best English caption track, or None if there's no trustworthy one.
      1. Creator-uploaded English subtitles (most accurate).
      2. YouTube's auto-captions in the *spoken* language when that's English ("en-orig").
      3. For Hindi/Hinglish videos only: YouTube's auto-translation to English.
    """
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    for key in sorted(manual, key=lambda k: (k != "en", k)):
        if key == "en" or key.startswith("en-"):
            return manual[key], "YouTube subtitles"

    if "en-orig" in auto:
        return auto["en-orig"], "YouTube auto-captions"

    spoken = next((k[:-5] for k in auto if k.endswith("-orig")), None)
    if language == "hinglish" and spoken in ("hi", "ur", "pa", "mr", "bn") and "en" in auto:
        return auto["en"], "YouTube auto-translated captions"

    return None, None


def _json3_to_text(raw: str) -> str:
    data = json.loads(raw)
    words = []
    for event in data.get("events", []):
        for seg in event.get("segs") or []:
            words.append(seg.get("utf8", ""))
    text = "".join(words).replace("\n", " ")
    text = re.sub(r"\[(music|applause|laughter)\]", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def fetch_youtube_captions(url: str, language: str = "english") -> dict | None:
    """
    Returns {"text", "engine", "duration", "title"} if usable captions exist, else None.
    Never raises for "no captions" — the caller just falls back to audio transcription.
    """
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True,
            "js_runtimes": JS_RUNTIMES, "no_color": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            formats, engine = _pick_track(info, language)
            if not formats:
                return None
            fmt = next((f for f in formats if f.get("ext") == "json3"), None)
            if not fmt:
                return None
            text = _json3_to_text(ydl.urlopen(fmt["url"]).read().decode("utf-8"))
    except Exception as e:
        print(f"Couldn't fetch captions ({e}) — falling back to audio transcription.")
        return None

    # Guard against empty / near-empty tracks (e.g. music videos with only "[Music]").
    if len(text.split()) < 20:
        return None
    return {"text": text, "engine": engine, "duration": info.get("duration") or 0,
            "title": info.get("title")}
