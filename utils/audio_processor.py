import yt_dlp
from pydub import AudioSegment
import os
import re
import shutil

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok = True)

# YouTube now protects its video URLs with a JavaScript challenge. yt-dlp solves it with
# the `yt-dlp-ejs` package + a JS runtime (deno is the default; we also allow Node.js).
JS_RUNTIMES = {name: {} for name in ("deno", "node", "bun") if shutil.which(name)}

# If the default client gets HTTP 403, retry with other YouTube clients.
FALLBACK_CLIENTS = [None, ["tv", "web_safari"], ["android", "ios"]]


def _clean_error(e: Exception) -> str:
    """Strip terminal colour codes like \x1b[0;31m from yt-dlp errors."""
    return re.sub(r"\x1b\[[0-9;]*m", "", str(e)).replace("ERROR: ", "").strip()


def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s [%(id)s].%(ext)s") # %(title)s - placeholder understood by yt-dlp-Replace this with the actual YouTube title.
    ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": output_path,
    "restrictfilenames": True,   # ASCII-only filenames — emoji / '｜' in titles break ffmpeg on Windows
    "noplaylist": True,
    "js_runtimes": JS_RUNTIMES,
    "retries": 5,
    "fragment_retries": 5,
    "no_color": True,

    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }
    ],

    "quiet": False,
}

    last_error = None
    for clients in FALLBACK_CLIENTS:
        opts = dict(ydl_opts)
        if clients:
            print(f"Retrying with YouTube clients: {clients}")
            opts["extractor_args"] = {"youtube": {"player_client": clients}}
        try:
            # with ensures resources are cleaned up automatically when downloading finishes or if an error occurs.
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                base = os.path.splitext(ydl.prepare_filename(info))[0]
                return base + ".wav"
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            if "403" not in str(e) and "Sign in" not in str(e):
                break

    hint = "" if JS_RUNTIMES else " (Install Node.js or Deno — YouTube needs a JS runtime.)"
    raise RuntimeError(f"YouTube download failed: {_clean_error(last_error)}{hint}")

# print(download_youtube_audio(
#     "https://www.youtube.com/watch?v=tiplN2Qc_ws"
# ))

# data = download_youtube_audio("https://www.youtube.com/watch?v=tiplN2Qc_ws")

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)  #Pydub loads the file into memory.
    #Most speech recognition models like Whisper are trained on 16 kHz mono audio, which is sufficient for human speech while reducing computation and storage.
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path

# data_final = convert_to_wav(data)

# print(data_final)

def chunk_audio(wav_path : str, chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

# print(chunk_audio(data_final))

#This is the controller function. It decides which processing path to take based on the input.
def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = convert_to_wav(download_youtube_audio(source))
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks