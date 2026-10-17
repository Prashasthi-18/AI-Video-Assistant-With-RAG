import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # OpenBLAS pre-allocates big buffers per thread — fails on low-RAM PCs

from dotenv import load_dotenv
load_dotenv()   # MUST be before any core/ imports (transcriber reads env vars at import time)

import re
import html
import time
from datetime import datetime, timezone
import streamlit as st

from ui.icons import icon, LOGO_SVG
from ui.auth_page import render_auth_page
from ui.session import restore_login, logout, flush_cookie_ops
from ui.effects import inject_effects

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="AI Video Assistant", page_icon=os.path.join(BASE_DIR, "assets", "logo.png"),
                   layout="wide", initial_sidebar_state="collapsed")

with open(os.path.join(BASE_DIR, "ui", "style.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
inject_effects()


# ── Auth gate ─────────────────────────────────────────────────────────────────
flush_cookie_ops()
restore_login()

if not st.session_state.get("user"):
    render_auth_page()
    st.stop()

user = st.session_state.user

# Light imports only — the heavy AI stack (torch, Whisper, Chroma) loads lazily when first needed,
# so the home screen appears instantly after sign-in.
from utils.audio_processor import DOWNLOAD_DIR
from utils.exporter import export_pdf, export_txt
from core.history import save_analysis, list_analyses, get_analysis, delete_analysis, count_analyses, youtube_id

for key, default in {"result": None, "chat_history": [], "error": None, "job": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

STEPS = [
    ("audio", "audio", "Source"),
    ("transcript", "file-text", "Transcribe"),
    ("title", "tag", "Title"),
    ("summary", "summary", "Summary"),
    ("extract", "sparkles", "Insights"),
    ("rag", "database", "RAG Index"),
]

SUGGESTED_QUESTIONS = [
    "What is the main topic of this video?",
    "What are the 3 most important takeaways?",
    "Were any tools, resources or links mentioned?",
    "Explain the hardest concept discussed in simple terms.",
]

EXAMPLES = [
    ("Steve Jobs · Stanford", "https://www.youtube.com/watch?v=UF8uR6Z6KLc", "english"),
    ("Tim Urban · TED", "https://www.youtube.com/watch?v=arj7oStGLkU", "english"),
    ("Obama · Speech", "https://www.youtube.com/watch?v=PGUdWfB8nLg", "english"),
    ("AI/ML roadmap · Hinglish", "https://www.youtube.com/watch?v=tiplN2Qc_ws", "hinglish"),
]

YOUTUBE, UPLOAD = ":material/smart_display: YouTube link", ":material/upload_file: Upload file"
LANG_LABELS = {"english": ":material/translate: English", "hinglish": ":material/record_voice_over: Hindi / Hinglish"}
GROQ_ON = bool(os.getenv("GROQ_API_KEY"))
ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds or 0), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def short_duration(seconds: float) -> str:
    m, s = divmod(int(seconds or 0), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def time_ago(created_at: str) -> str:
    try:
        then = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return ""
    secs = int((datetime.now(timezone.utc) - then).total_seconds())
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= size:
            return f"{secs // size}{unit} ago"
    return "just now"


def greeting() -> str:
    h = datetime.now().hour
    return "Good morning" if h < 12 else "Good afternoon" if h < 17 else "Good evening"


def engine_short(engine: str | None) -> str:
    e = (engine or "").lower()
    return ("Captions" if "caption" in e or "subtitle" in e else "Groq" if "groq" in e
            else "Sarvam" if "sarvam" in e else "Whisper")


def plain_preview(md: str | None) -> str:
    text = re.sub(r"[*#>`_]|^\s*[-•]\s*", " ", md or "", flags=re.M)
    return html.escape(re.sub(r"\s+", " ", text).strip())


def card(key: str, ic: str, title: str, body_md: str):
    with st.container(key=f"card_{key}"):
        st.markdown(f'<div class="card-title"><div class="ic">{icon(ic, 17)}</div><span>{title}</span></div>',
                    unsafe_allow_html=True)
        st.markdown(body_md)


def metric(ic: str, label: str, value: str) -> str:
    return f'<div class="metric"><div class="k">{icon(ic, 14)}{label}</div><div class="v">{value}</div></div>'


def friendly_error(e: Exception) -> tuple[str, str]:
    low = ANSI.sub("", str(e)).lower()
    if "403" in low or "youtube download failed" in low:
        return "Couldn't download this YouTube video", (
            "YouTube blocked the request. Make sure the video is public, then try again — "
            "or download it yourself and use <b>Upload file</b>.")
    if "429" in low or "rate limit" in low:
        return "Rate limit reached", ("An AI provider's request limit was hit. Wait a minute and try again, or set "
                                      "<code>MISTRAL_MODEL</code> in <code>.env</code> to a model your plan allows.")
    if "mistral" in low or "401" in low or "unauthorized" in low:
        return "LLM request failed", "Check that <code>MISTRAL_API_KEY</code> in your <code>.env</code> is valid."
    if "sarvam" in low:
        return "Sarvam AI transcription failed", "Check <code>SARVAM_API_KEY</code> in your <code>.env</code>."
    if "not enough memory" in low or "out of memory" in low or "errno 22" in low or "bad_alloc" in low:
        return "Your computer ran out of memory", (
            "Close other apps and try again — or add a <code>GROQ_API_KEY</code> so transcription runs in the cloud.")
    if "ffmpeg" in low or "ffprobe" in low:
        return "FFmpeg not found", "Install FFmpeg and make sure it's on your PATH."
    if "no speech" in low:
        return "No speech found", "This audio doesn't seem to contain any speech."
    return "Something went wrong", "See the technical details below."


# ── Navigation / state actions ────────────────────────────────────────────────
def go_home():
    st.session_state.result, st.session_state.chat_history, st.session_state.job = None, [], None


def start_job(source: str, language: str, label: str):
    st.session_state.error = None
    st.session_state.job = {"source": source, "language": language, "label": label}


def start_example(url: str, language: str, label: str):
    st.session_state.yt_url = url
    start_job(url, language, label)


def open_analysis(analysis_id: int):
    row = get_analysis(user["id"], analysis_id)
    if not row:
        st.session_state.error = ("Analysis not found", "It may have been deleted.", "")
        return
    source = row.get("source") or ""
    st.session_state.result = {
        **{k: row.get(k) for k in ("id", "title", "transcript", "summary", "action_items", "key_decisions",
                                   "open_questions", "duration", "elapsed", "language", "engine")},
        "youtube_url": source if source.startswith("http") else None,
        "rag_chain": None,   # rebuilt lazily on the first chat question
    }
    st.session_state.chat_history = []


@st.dialog("Welcome aboard", width="medium")
def welcome_dialog():
    first = html.escape(user["name"].split()[0])
    st.markdown(
        f"""<div class="welcome">{LOGO_SVG}<h2>Hi {first}, you're all set!</h2>
<p>Here's how to get the most out of AI Video Assistant:</p>
<ol><li><b>Paste a YouTube link</b> — videos with captions are analysed in seconds.</li>
<li><b>Or upload a recording</b> — meetings, lectures, podcasts.</li>
<li><b>Chat with it</b> — every answer is grounded in what was actually said.</li></ol></div>""",
        unsafe_allow_html=True,
    )
    if st.button("Let's go", type="primary", icon=":material/rocket_launch:", use_container_width=True):
        st.rerun()


def _clear_delete():
    st.session_state.pop("confirm_delete", None)


@st.dialog("Delete this analysis?", on_dismiss=_clear_delete)
def delete_dialog(analysis_id: int, title: str):
    st.markdown(f"**{html.escape(title)}** will be removed from your library. This can't be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", use_container_width=True):
        _clear_delete()
        st.rerun()
    if c2.button("Delete", type="primary", icon=":material/delete:", use_container_width=True):
        delete_analysis(user["id"], analysis_id)
        if (st.session_state.result or {}).get("id") == analysis_id:
            go_home()
        _clear_delete()
        st.session_state.flash = ("Deleted from your library", ":material/delete:")
        st.rerun()


# ── Top navigation ────────────────────────────────────────────────────────────
def render_topnav():
    with st.container(key="topnav"):
        c1, c2, c3 = st.columns([5, 4, 2.4], vertical_alignment="center")
        c1.markdown(
            f'<div class="brand">{LOGO_SVG}<div><div class="brand-name">AI Video Assistant</div>'
            '<div class="brand-tag">Transcribe · Summarise · Chat</div></div></div>',
            unsafe_allow_html=True,
        )
        with c2:
            if st.session_state.result:
                st.button("New analysis", icon=":material/add:", type="tertiary", on_click=go_home, key="nav_new")
            else:
                mode = "Fast mode · Groq + captions" if GROQ_ON else "Captions + local Whisper"
                st.markdown(f'<div class="nav-meta"><span class="live-dot"></span>{mode}</div>',
                            unsafe_allow_html=True)
        with c3:
            first = user["name"].split()[0]
            with st.popover(first, icon=":material/account_circle:", key="usermenu", use_container_width=True):
                initials = "".join(p[0] for p in user["name"].split()[:2]).upper() or "U"
                st.markdown(
                    f'<div class="um-card"><div class="avatar">{html.escape(initials)}</div><div>'
                    f'<b>{html.escape(user["name"])}</b><span>{html.escape(user["email"])}</span></div></div>'
                    f'<div class="um-stats"><div><b>{count_analyses(user["id"])}</b><span>videos analysed</span></div>'
                    f'<div><b>{"Groq" if GROQ_ON else "Local"}</b><span>transcription</span></div></div>',
                    unsafe_allow_html=True,
                )
                st.button("My library", icon=":material/video_library:", on_click=go_home,
                          use_container_width=True, key="um_lib")
                if st.button("Sign out", icon=":material/logout:", use_container_width=True, key="um_out"):
                    logout()
                    st.rerun()


def render_error():
    if not st.session_state.error:
        return
    head, hint, detail = st.session_state.error
    st.markdown(f'<div class="err"><div class="ic">{icon("alert", 22)}</div><div><b>{head}</b><p>{hint}</p></div></div>',
                unsafe_allow_html=True)
    if detail:
        _, mid, _ = st.columns([1, 6, 1])
        with mid.expander("Technical details"):
            st.code(detail, language=None)


# ── Home ──────────────────────────────────────────────────────────────────────
def render_home():
    if st.session_state.pop("show_welcome", False):
        welcome_dialog()

    first = html.escape(user["name"].split()[0])
    st.markdown(
        f"""<div class="home-hero">
  <span class="greet fade-up">{icon("sparkles", 14)}{greeting()}, {first}</span>
  <h1 class="fade-up d1">What will you <span class="grad-anim">learn</span> today?</h1>
  <p class="sub fade-up d2">Paste a video or drop a recording — get the summary, the decisions,
  and a chat that knows every word.</p>
</div>""",
        unsafe_allow_html=True,
    )

    render_error()

    with st.container(key="composer"):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            mode = st.segmented_control("Source", [YOUTUBE, UPLOAD], default=YOUTUBE, key="src_mode",
                                        label_visibility="collapsed") or YOUTUBE
            language = st.segmented_control("Language", list(LANG_LABELS), default="english", key="lang",
                                            format_func=LANG_LABELS.get, label_visibility="collapsed") or "english"

        if mode == YOUTUBE:
            c1, c2 = st.columns([5, 1.3], vertical_alignment="center")
            url = c1.text_input("YouTube URL", key="yt_url", label_visibility="collapsed", icon=":material/link:",
                                placeholder="Paste a YouTube link — lectures, podcasts, talks, meetings…")
            go = c2.button("Analyse", type="primary", icon=":material/auto_awesome:", use_container_width=True, key="go")
            if go:
                if re.match(r"^https?://", url.strip()):
                    start_job(url.strip(), language, "YouTube video")
                    st.rerun()
                st.session_state.error = ("Add a video first", "Paste a full YouTube link (starting with https://).", "")
                st.rerun()
            hint = (None if language == "english"
                    else "Hindi / Hinglish speech is transcribed and translated to English by Sarvam AI.")
        else:
            upload = st.file_uploader("Audio / video file", label_visibility="collapsed",
                                      type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "mov", "aac", "flac"])
            if st.button("Analyse", type="primary", icon=":material/auto_awesome:", use_container_width=True, key="go_up"):
                if not upload:
                    st.session_state.error = ("Add a file first", "Choose an audio or video file to upload.", "")
                    st.rerun()
                path = os.path.join(DOWNLOAD_DIR, os.path.basename(upload.name))
                with open(path, "wb") as f:
                    f.write(upload.getbuffer())
                start_job(path, language, upload.name)
                st.rerun()
            hint = ("Hindi / Hinglish audio is transcribed and translated by Sarvam AI." if language == "hinglish"
                    else "Audio is transcribed in the cloud with Groq Whisper." if GROQ_ON
                    else "Audio is transcribed locally — add a GROQ_API_KEY to .env for ~20× faster results.")
        if hint:
            st.markdown(f'<div class="composer-hint">{icon("zap", 13)}{hint}</div>', unsafe_allow_html=True)

    with st.container(key="examples", horizontal=True, gap="small"):
        st.markdown('<span class="chip-label">Try one:</span>', unsafe_allow_html=True)
        for i, (label, url, lang) in enumerate(EXAMPLES):
            st.button(label, key=f"ex_{i}", on_click=start_example, args=(url, lang, label))

    render_library()
    st.markdown('<div class="footer">Built with Python · Streamlit · LangChain · Mistral · ChromaDB</div>',
                unsafe_allow_html=True)


def render_library():
    items = list_analyses(user["id"])
    count = f"{len(items)} video{'s' if len(items) != 1 else ''}" if items else ""
    st.markdown(f'<div class="section-head"><h3>{icon("library", 20)}Your library</h3>'
                f'<span class="count">{count}</span></div>', unsafe_allow_html=True)

    if not items:
        steps = [("link", "Paste a link", "Any YouTube video or a recording from your computer."),
                 ("sparkles", "Get the essentials", "Summary, action items, decisions and open questions."),
                 ("message", "Chat with it", "Ask anything — answers cite only what was said.")]
        st.markdown('<div class="onboard">' + "".join(
            f'<div class="ob-step" style="animation-delay:{i * 0.08}s"><span class="num">{i + 1}</span>'
            f'<div class="ic">{icon(ic, 20)}</div><h4>{t}</h4><p>{d}</p></div>'
            for i, (ic, t, d) in enumerate(steps)) + "</div>", unsafe_allow_html=True)
        return

    cols = st.columns(3, gap="medium")
    for idx, item in enumerate(items):
        vid = youtube_id(item.get("source"))
        thumb = (f'<img src="https://i.ytimg.com/vi/{vid}/mqdefault.jpg" alt="" loading="lazy">' if vid
                 else icon("upload", 34, 1.5))
        with cols[idx % 3]:
            with st.container(key=f"lib_{item['id']}"):
                st.markdown(
                    f'<div class="thumb">{thumb}<span class="badge">{engine_short(item.get("engine"))}</span>'
                    f'<span class="dur">{short_duration(item.get("duration"))}</span></div>'
                    f'<div class="lib-title">{html.escape(item["title"])}</div>'
                    f'<div class="lib-meta">{icon("calendar", 12)}{time_ago(item.get("created_at"))}</div>'
                    f'<div class="lib-preview">{plain_preview(item.get("summary_preview"))}</div>',
                    unsafe_allow_html=True,
                )
                b1, b2 = st.columns([4, 1])
                b1.button("Open", key=f"open_{item['id']}", icon=":material/arrow_outward:",
                          on_click=open_analysis, args=(item["id"],), use_container_width=True)
                if b2.button("", key=f"del_{item['id']}", icon=":material/delete:", help="Delete",
                             use_container_width=True):
                    st.session_state.confirm_delete = (item["id"], item["title"])

    if st.session_state.get("confirm_delete"):
        delete_dialog(*st.session_state.confirm_delete)


# ── Processing ────────────────────────────────────────────────────────────────
def render_stepper(placeholder, states: dict, times: dict, started: float):
    items = []
    for key, ic, name in STEPS:
        state = states.get(key, "pending")
        glyph = icon("check", 20, 2.4) if state == "done" else icon("x", 20, 2.4) if state == "error" else icon(ic, 20)
        t = f"{times[key]:.1f}s" if key in times else ("running…" if state == "active" else "")
        items.append(f'<div class="step {state}"><div class="bubble">{glyph}</div>'
                     f'<div class="name">{name}</div><div class="time">{t}</div></div>')
    done = sum(1 for k, *_ in STEPS if states.get(k) == "done")
    pct = max(4, int(done / len(STEPS) * 100))
    active = next((n for k, _, n in STEPS if states.get(k) == "active"), None)
    caption = (f"Working on <b>{active}</b>…" if active else "Complete" if done == len(STEPS)
               else "Loading AI models…")
    placeholder.markdown(
        f'<div class="stepper">{"".join(items)}</div>'
        f'<div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>'
        f'<div class="progress-caption"><span>{caption}</span>'
        f'<span>{done}/{len(STEPS)} · {time.time() - started:.0f}s elapsed</span></div>',
        unsafe_allow_html=True,
    )


def render_processing():
    job = st.session_state.job
    # While this long run works, hide leftovers from the previous screen (Streamlit marks them stale).
    st.markdown('<style>[data-stale="true"], div[class*="st-key-lib_"], div.st-key-composer, div.st-key-examples '
                '{ display: none !important; }</style>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="proc-head"><div class="orb">{icon("wand", 26)}</div>'
        f'<h1>Analysing your video</h1><p>{html.escape(job["label"])}</p></div>',
        unsafe_allow_html=True,
    )
    stepper = st.empty()
    states, times, started = {}, {}, time.time()

    def run(key, fn, *args):
        states[key] = "active"
        render_stepper(stepper, states, times, started)
        t0 = time.time()
        try:
            out = fn(*args)
        except Exception:
            states[key] = "error"
            render_stepper(stepper, states, times, started)
            raise
        times[key], states[key] = time.time() - t0, "done"
        render_stepper(stepper, states, times, started)
        return out

    render_stepper(stepper, states, times, started)
    try:
        # Heavy AI stack loads here (first run only).
        from core.transcript_source import prepare_source, get_transcript
        from core.summarizer import summarize, generate_title
        from core.extractor import extract_all
        from core.rag_engine import build_rag_chain

        source, language = job["source"], job["language"]
        prepared = run("audio", prepare_source, source, language)
        tr = run("transcript", get_transcript, prepared, language)
        if not tr["text"]:
            raise RuntimeError("No speech was detected in this audio.")
        title = run("title", generate_title, tr["text"])
        summary = run("summary", summarize, tr["text"])
        extracted = run("extract", extract_all, tr["text"])
        rag_chain = run("rag", build_rag_chain, tr["text"])

        result = {
            "title": title, "transcript": tr["text"], "summary": summary, **extracted,
            "duration": tr["duration"], "elapsed": time.time() - started, "language": language,
            "engine": tr["engine"], "source": source,
            "youtube_url": source if source.startswith("http") else None,
        }
        result["id"] = save_analysis(user["id"], result)
        result["rag_chain"] = rag_chain
        st.session_state.result, st.session_state.chat_history = result, []
        st.session_state.flash = ("Analysis ready — saved to your library", ":material/check_circle:")
    except Exception as e:
        failed = next((n for k, _, n in STEPS if states.get(k) == "error"), None)
        head, hint = friendly_error(e)
        st.session_state.error = (head + (f" (at {failed})" if failed else ""), hint, ANSI.sub("", str(e)))
    st.session_state.job = None
    st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
def ensure_rag_chain(r: dict):
    if not r.get("rag_chain"):
        from core.rag_engine import build_rag_chain
        with st.spinner("Preparing chat — indexing the transcript…"):
            r["rag_chain"] = build_rag_chain(r["transcript"])
    return r["rag_chain"]


def render_results():
    r = st.session_state.result
    chat = st.session_state.chat_history

    left, right = st.columns([1, 1], vertical_alignment="center")
    left.button("Back to library", icon=":material/arrow_back:", type="tertiary", on_click=go_home, key="back")
    with right.container(horizontal=True, horizontal_alignment="right", gap="small"):
        with open(export_pdf(r, chat), "rb") as f:
            st.download_button("PDF", f.read(), file_name=f"{r['title']}.pdf", mime="application/pdf",
                               icon=":material/picture_as_pdf:")
        with open(export_txt(r, chat), "rb") as f:
            st.download_button("TXT", f.read(), file_name=f"{r['title']}.txt", mime="text/plain",
                               icon=":material/description:")
        if r.get("id") and st.button("", icon=":material/delete:", help="Delete from library", key="res_del"):
            st.session_state.confirm_delete = (r["id"], r["title"])
    if st.session_state.get("confirm_delete"):
        delete_dialog(*st.session_state.confirm_delete)

    words = len(r["transcript"].split())
    st.markdown(
        f'<div class="result-head"><div class="eyebrow">Analysis complete '
        f'<span class="engine-chip">{html.escape(r.get("engine") or "Whisper")}</span></div>'
        f'<h1>{html.escape(r["title"])}</h1></div>'
        '<div class="metrics">'
        + metric("clock", "Duration", fmt_duration(r.get("duration")))
        + metric("type", "Words", f"{words:,}")
        + metric("book", "Read time", f'{max(1, words // 230)}<small>min</small>')
        + metric("zap", "Processed in", f'{r.get("elapsed") or 0:.0f}<small>s</small>')
        + "</div>",
        unsafe_allow_html=True,
    )

    tab_overview, tab_chat, tab_transcript = st.tabs(
        [":material/dashboard: Overview", ":material/forum: Chat", ":material/description: Transcript"])

    with tab_overview:
        if r.get("youtube_url"):
            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                card("summary", "summary", "Summary", r["summary"])
            with c2:
                st.video(r["youtube_url"])
        else:
            card("summary", "summary", "Summary", r["summary"])
        st.write("")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            card("actions", "circle-check", "Action Items", r["action_items"])
        with c2:
            card("decisions", "flag", "Key Decisions", r["key_decisions"])
        with c3:
            card("questions", "help", "Open Questions", r["open_questions"])

    with tab_chat:
        render_chat(r)

    with tab_transcript:
        query = st.text_input("Search transcript", placeholder="Search the transcript…",
                              icon=":material/search:", label_visibility="collapsed")
        text = html.escape(r["transcript"])
        if query.strip():
            pattern = re.compile(re.escape(html.escape(query.strip())), re.IGNORECASE)
            hits = len(pattern.findall(text))
            text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
            st.caption(f"{hits} match{'es' if hits != 1 else ''}")
        st.markdown(f'<div class="transcript">{text}</div>', unsafe_allow_html=True)


def render_chat(r: dict):
    USER_AVATAR, BOT_AVATAR = ":material/person:", ":material/auto_awesome:"
    history = st.session_state.chat_history
    picked = None

    if not history:
        st.markdown(f'<div class="chat-empty"><div class="big">{icon("message", 30)}</div>'
                    '<h3>Ask your video anything</h3>'
                    '<p>Answers come only from the transcript — no hallucinated facts.</p></div>',
                    unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"suggest_{i}", icon=":material/arrow_outward:", use_container_width=True):
                picked = q

    for msg in history:
        with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about the video…") or picked
    if question:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            try:
                from core.rag_engine import ask_question
                chain = ensure_rag_chain(r)
                with st.spinner("Searching the transcript…"):
                    answer = ask_question(chain, question)
            except Exception as e:
                answer = f"**{friendly_error(e)[0]}:** {ANSI.sub('', str(e))}"
            st.markdown(answer)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        st.rerun()

    if history and st.button("Clear conversation", icon=":material/delete_sweep:", type="tertiary"):
        st.session_state.chat_history = []
        st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────
render_topnav()

if flash := st.session_state.pop("flash", None):
    st.toast(flash[0], icon=flash[1])

if st.session_state.job:
    render_processing()
elif st.session_state.result:
    render_results()
else:
    render_home()
