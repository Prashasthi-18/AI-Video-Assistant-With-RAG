from dotenv import load_dotenv
load_dotenv()   # MUST be before any core/ imports

import sys
from main import run_pipeline
from core.rag_engine import ask_question

# Quick end-to-end check on a local file or YouTube URL:
#   python test.py "downloads/some_audio.wav"
source = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=tiplN2Qc_ws"

result = run_pipeline(source)
for key in ["title", "summary", "action_items", "key_decisions", "open_questions"]:
    print(f"\n--- {key} ---\n{result[key]}")

ask_question(result["rag_chain"], "What is this video about?")
