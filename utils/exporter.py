import os
import re
from fpdf import FPDF

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

SECTIONS = [
    ("Summary", "summary"),
    ("Action Items", "action_items"),
    ("Key Decisions", "key_decisions"),
    ("Open Questions", "open_questions"),
]

# A TrueType font lets the PDF contain non-Latin characters (Hindi, curly quotes, ₹ …).
UNICODE_FONTS = [
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]


def _file_base(title: str) -> str:
    safe = re.sub(r"[^\w\- ]", "", title).strip()[:60] or "report"
    return os.path.join(EXPORT_DIR, safe)


def _plain(text: str) -> str:
    """Strip basic Markdown so it reads cleanly in PDF/TXT."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = re.sub(r"^(\s*)[-*]\s+", r"\1• ", text, flags=re.M)
    return text.strip()


def export_txt(result: dict, chat: list | None = None) -> str:
    lines = [result["title"], "=" * len(result["title"]), ""]
    for heading, key in SECTIONS:
        lines += [heading.upper(), "-" * len(heading), _plain(result.get(key, "")), ""]

    if chat:
        lines += ["Q&A", "---"]
        for msg in chat:
            who = "You" if msg["role"] == "user" else "Assistant"
            lines += [f"{who}: {msg['content']}", ""]

    lines += ["FULL TRANSCRIPT", "---------------", result["transcript"]]

    path = _file_base(result["title"]) + ".txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def export_pdf(result: dict, chat: list | None = None) -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font = "Helvetica"
    for regular, bold in UNICODE_FONTS:
        if os.path.exists(regular):
            pdf.add_font("Body", "", regular)
            pdf.add_font("Body", "B", bold if os.path.exists(bold) else regular)
            font = "Body"
            break

    def clean(text):
        # Built-in Helvetica only supports latin-1
        return text if font == "Body" else text.encode("latin-1", "replace").decode("latin-1")

    def heading(text, size=14):
        pdf.set_font(font, "B", size)
        pdf.multi_cell(0, 9, clean(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def body(text):
        pdf.set_font(font, "", 11)
        pdf.multi_cell(0, 6, clean(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    heading(result["title"], 18)
    for title, key in SECTIONS:
        heading(title)
        body(_plain(result.get(key, "")))

    if chat:
        heading("Q&A")
        for msg in chat:
            who = "You" if msg["role"] == "user" else "Assistant"
            body(f"{who}: {msg['content']}")

    pdf.add_page()
    heading("Full Transcript")
    body(result["transcript"])

    path = _file_base(result["title"]) + ".pdf"
    pdf.output(path)
    return path
