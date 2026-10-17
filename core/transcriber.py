import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # Windows can't symlink; caching still works

from faster_whisper import WhisperModel
import numpy as np
import gc
import time
import requests
from pydub import AudioSegment

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

# faster-whisper runs the same Whisper models through CTranslate2 with 8-bit weights:
# ~3-4x faster on CPU and about half the RAM of openai-whisper, with near-identical accuracy.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))   # 1 = greedy (fastest); 5 = slightly more accurate
WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", "0"))       # 0 = let CTranslate2 pick

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

WHISPER_FALLBACKS = ["base", "tiny"]   # used only if the configured model runs out of memory

_model = None
_model_name = None


def load_model(name: str = WHISPER_MODEL):

    global _model, _model_name

    if _model is None or _model_name != name:
        _model = None
        gc.collect()
        print(f"Loading Whisper model: {name} ({WHISPER_COMPUTE_TYPE}) ...")
        # First use downloads the model from Hugging Face (small ≈ 460 MB) and caches it.
        _model = WhisperModel(name, device="cpu", compute_type=WHISPER_COMPUTE_TYPE,
                              cpu_threads=WHISPER_THREADS)
        _model_name = name
        print("Whisper model loaded.")
    return _model


def _load_wav(path: str) -> np.ndarray:
    """
    Chunks are already 16 kHz mono WAV (see utils/audio_processor.py), so read the samples
    directly. This skips faster-whisper's PyAV decoder, which breaks on newer PyAV releases.
    """
    audio = AudioSegment.from_wav(path).set_channels(1).set_frame_rate(16000).set_sample_width(2)
    return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0


def _is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return isinstance(e, MemoryError) or "not enough memory" in msg or "out of memory" in msg \
        or "memory allocation" in msg or "errno 22" in msg or "bad_alloc" in msg


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """
    Transcribe with the configured model. On low-RAM machines, if it runs out of memory,
    fall back to smaller models (small → base → tiny) instead of crashing.
    """
    global _model

    order = [WHISPER_MODEL] + [m for m in WHISPER_FALLBACKS if m != WHISPER_MODEL]
    # If an earlier chunk already fell back, keep using that smaller model.
    names = order[order.index(_model_name):] if _model_name in order else order

    for name in names:
        try:
            model = load_model(name)
            # task="translate" always outputs English: English audio is transcribed as-is, and
            # code-switched speech (e.g. Hinglish) doesn't come out as a mix of random scripts.
            # vad_filter skips silences, so quiet stretches cost no compute.
            segments, _ = model.transcribe(_load_wav(chunk_path), task="translate",
                                           beam_size=WHISPER_BEAM_SIZE, vad_filter=True)
            # segments is a lazy generator — the actual transcription happens while joining.
            return " ".join(s.text.strip() for s in segments)
        except (RuntimeError, MemoryError, OSError) as e:
            if not _is_oom(e) or name == names[-1]:
                raise
            print(f"⚠️ Not enough memory for Whisper '{name}' — falling back to a smaller model.")
            _model = None
            gc.collect()


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk_groq(chunk_path: str) -> str:
    """
    Cloud Whisper large-v3-turbo on Groq: ~200x realtime and more accurate than local "small".
    The WAV chunk is re-encoded to 64 kbps MP3 (10 min ≈ 5 MB) to stay under Groq's 25 MB limit.
    """
    mp3_path = f"{chunk_path}_groq.mp3"
    AudioSegment.from_wav(chunk_path).export(mp3_path, format="mp3", bitrate="64k")
    try:
        for attempt in range(3):
            with open(mp3_path, "rb") as f:
                response = requests.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    files={"file": (os.path.basename(mp3_path), f, "audio/mpeg")},
                    data={"model": GROQ_MODEL, "language": "en", "response_format": "json", "temperature": "0"},
                    timeout=300,
                )
            if response.status_code == 429:
                # Free tier has per-hour audio limits. Short waits are worth it; long ones aren't.
                wait = float(response.headers.get("retry-after", "60"))
                if wait > 30 or attempt == 2:
                    raise RuntimeError(f"Groq rate limit (retry after {wait:.0f}s)")
                print(f"  Groq rate limited — retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            if not response.ok:
                raise RuntimeError(f"Groq returned {response.status_code}: {response.text[:200]}")
            return response.json().get("text", "").strip()
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


def transcribe(chunks: list, language: str = "english") -> tuple[str, str]:
    """
    Transcribe all chunks and return (transcript, engine label).
      hinglish → Sarvam AI (translates to English while transcribing)
      english  → Groq cloud Whisper if GROQ_API_KEY is set, else local faster-whisper.
                 If Groq fails on a chunk (rate limit / network), that chunk falls back to local.
    """
    hinglish = language.lower() == "hinglish"
    use_groq = not hinglish and bool(GROQ_API_KEY)
    engines = set()

    print(f"Using {'Sarvam AI' if hinglish else 'Groq Whisper' if use_groq else 'local Whisper'} for transcription.")
    parts = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        if hinglish:
            parts.append(transcribe_chunk_sarvam(chunk)); engines.add("Sarvam AI")
            continue
        if use_groq:
            try:
                parts.append(transcribe_chunk_groq(chunk)); engines.add("Groq Whisper")
                continue
            except Exception as e:
                print(f"⚠️ Groq failed ({e}) — using local Whisper for this chunk.")
        parts.append(transcribe_chunk_whisper(chunk)); engines.add("Whisper (local)")

    print("Transcription complete.")
    order = ["Sarvam AI", "Groq Whisper", "Whisper (local)"]
    return " ".join(parts).strip(), " + ".join(e for e in order if e in engines)


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Kept for main.py / test.py compatibility — returns just the transcript text."""
    return transcribe(chunks, language)[0]
