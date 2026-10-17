import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # OpenBLAS pre-allocates big buffers per thread — fails on low-RAM PCs

from dotenv import load_dotenv
load_dotenv()   # MUST be before any core/ imports (transcriber reads env vars at import time)

import sys

from utils.exporter import export_pdf, export_txt
from core.transcript_source import prepare_source, get_transcript
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question

# Windows consoles default to cp1252, which can't print emoji / Hindi text.
sys.stdout.reconfigure(encoding="utf-8")


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Video Assistant")

    tr = get_transcript(prepare_source(source, language), language)
    transcript = tr["text"]
    print(f"Transcript source: {tr['engine']}")
    if not transcript:
        raise RuntimeError("No speech detected in this audio.")
    print(f"Raw transcription (first 300 characters): {transcript[:300]}")

    title = generate_title(transcript)
    summary = summarize(transcript)
    extracted = extract_all(transcript)
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        **extracted,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip().strip('"')
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    print(f"\n📄 PDF report: {export_pdf(result)}")
    print(f"📄 TXT report: {export_txt(result)}")

    # Phase 2 — Chat with your video via RAG
    print("\n💬 Chat with your video (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")
