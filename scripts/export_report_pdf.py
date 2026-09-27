"""Render the written report to PDF.

`reports/final_report.md` is the deliverable. This converts it to a paginated
PDF so it can be read, printed and submitted without a Markdown renderer, and
so the tables stay tables rather than turning into walls of pipe characters.

Pipeline: Markdown -> HTML (python-markdown) -> PDF (headless Edge print).

Why Edge rather than a Python PDF library. `reportlab` is available, but
building a paginated document engine - column widths, table header repetition
across page breaks, widow/orphan control, code-block backgrounds - by hand is a
large piece of work that produces a worse document. A browser already does all
of that correctly, and headless Chrome/Edge can print to PDF with no GUI. The
print stylesheet below is where the typographic decisions live.

Run:
    python scripts/export_report_pdf.py
"""

from __future__ import annotations

import html as htmllib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import FIGURES_DIR, REPORTS_DIR  # noqa: E402

SOURCE = REPORTS_DIR / "final_report.md"
TARGET = ROOT / "NorthGate-AI-Report.pdf"

# Candidates in preference order. Edge is usually present on Windows; Chrome on
# Linux and macOS. Both accept the same --headless --print-to-pdf contract.
BROWSERS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
    Path("/usr/bin/chromium-browser"),
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
]

# Print styling. Sized for A4 with room for a binding margin, because this is a
# document someone reads rather than a slide someone glances at.
CSS = """
@page { size: A4; margin: 20mm 18mm 22mm 18mm; }

html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }

body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.2pt;
  line-height: 1.55;
  color: #1a1a1a;
  max-width: none;
  margin: 0;
  hyphens: auto;
}

/* ---- Title block ---- */
h1 {
  font-size: 22pt; font-weight: 700; letter-spacing: -0.01em;
  margin: 0 0 4pt 0; color: #111;
}
h1 + p { font-size: 10.5pt; color: #555; margin-top: 0; }

/* A rule under the H1, and a page break before each top-level section after the
   first so a chapter does not begin two lines from the bottom of a page. */
hr {
  border: 0; border-top: 1.5pt solid #1a1a1a;
  margin: 10pt 0 16pt 0;
}
h2 {
  font-size: 15pt; font-weight: 700; color: #0d1b2a;
  margin: 20pt 0 8pt 0; padding-bottom: 4pt;
  border-bottom: 1pt solid #c8d0da;
  page-break-after: avoid; break-after: avoid;
}
h3 {
  font-size: 12pt; font-weight: 650; color: #16324f;
  margin: 15pt 0 6pt 0;
  page-break-after: avoid; break-after: avoid;
}
h4 {
  font-size: 10.6pt; font-weight: 650; color: #333;
  margin: 12pt 0 4pt 0;
  page-break-after: avoid; break-after: avoid;
}
p { margin: 0 0 7pt 0; text-align: justify; hyphens: auto; }

ul, ol { margin: 0 0 8pt 0; padding-left: 16pt; }
li { margin-bottom: 3.5pt; }

/* ---- Tables ----
   The report's core evidence is tabular (the Section 9.5 leaderboard, the
   acceptance table). Wide tables are allowed to run to the page width and are
   kept on one page where possible, because a comparison table split across two
   pages is much harder to read than the same table in smaller type. */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 8pt 0 12pt 0;
  font-size: 8.3pt;
  page-break-inside: avoid;
  break-inside: avoid;
}
th, td {
  border: 0.5pt solid #b6bfca;
  padding: 3.2pt 5pt;
  text-align: left;
  vertical-align: top;
  hyphens: none;
  word-break: break-word;
}
th {
  background: #e8edf3;
  font-weight: 650;
  color: #0d1b2a;
}
/* Only genuinely numeric cells are right-aligned. Applying this to every
   non-first column, as an earlier version did, right-aligned text columns too -
   so a "Module" column of file paths sat flush right and read as if the paths
   were the values being compared. The `num` class is applied per-cell by
   `mark_numeric_cells`, which looks at the content rather than the position. */
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td code { word-break: break-all; }

code {
  font-family: "Cascadia Mono", Consolas, "SF Mono", monospace;
  font-size: 8.6pt;
  background: #f0f2f5;
  padding: 0.5pt 2.5pt;
  border-radius: 2pt;
  color: #1a3a5c;
}
pre {
  background: #f6f8fa;
  border: 0.5pt solid #d0d7de;
  border-left: 2.5pt solid #4a6b8a;
  border-radius: 3pt;
  padding: 7pt 9pt;
  margin: 8pt 0 11pt 0;
  font-size: 8pt;
  line-height: 1.4;
  white-space: pre-wrap;
  word-wrap: break-word;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8pt; color: #1a1a1a; }

blockquote {
  margin: 8pt 0;
  padding: 5pt 10pt;
  border-left: 2.5pt solid #9aa7b4;
  background: #f6f8fa;
  color: #333;
  page-break-inside: avoid;
}
strong { font-weight: 650; color: #0d1b2a; }
a { color: #1a4d8f; text-decoration: none; }

img { max-width: 100%; height: auto; page-break-inside: avoid; }

/* Keep a heading attached to the paragraph that follows it, and never strand a
   single line of a paragraph at the foot of a page. */
p, li { orphans: 2; widows: 2; }
"""


def find_browser() -> Path | None:
    for p in BROWSERS:
        if p.exists():
            return p
    return None


def mark_numeric_cells(html: str) -> str:
    """Tag table cells whose content is a number, so CSS can align just those.

    Alignment by column position cannot work on these tables: the layering table
    has a text "Module" column in the middle and numeric columns elsewhere in
    the same document, and the acceptance table mixes prose verdicts into
    numeric-looking columns. So each cell is classified on its own content.

    A cell counts as numeric if stripping markdown emphasis, code ticks,
    thousands separators, currency and trailing percent/symbol leaves something
    that parses as a number. Verdicts like "**MET**" therefore stay left-aligned,
    which is what a reader expects from a word.
    """
    cell_re = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
    tag_re = re.compile(r"^<t([dh])([^>]*)>", re.S)

    def looks_numeric(fragment: str) -> bool:
        text = re.sub(r"<[^>]+>", "", fragment)
        text = text.replace("**", "").replace("`", "").replace("*", "")
        text = text.replace("&minus;", "-").replace("&nbsp;", " ")
        text = text.strip().replace(",", "").replace(" ", "")
        text = text.rstrip("%$€£ ").lstrip("$€£")
        # Accept an optional sign, digits with optional decimals, and common
        # exponent forms. Reject anything with a letter left over.
        return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text))

    def fix(match: re.Match) -> str:
        whole = match.group(0)
        kind_and_attrs = tag_re.match(whole)
        if kind_and_attrs is None:
            return whole
        kind, attrs = kind_and_attrs.group(1), kind_and_attrs.group(2)
        if 'class="num"' in attrs:
            return whole
        if kind == "th" or looks_numeric(match.group(1)):
            return f'<t{kind} class="num"{attrs}>{match.group(1)}</t{kind}>'
        return whole

    return cell_re.sub(fix, html)


def build_html(md_text: str) -> str:
    """Markdown -> a complete, self-contained HTML document.

    The `tables` extension is not optional here: without it every comparison
    table in the report renders as a paragraph of pipe characters, which would
    make the PDF unusable for the one thing it exists to show.
    """
    import markdown

    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )
    body = mark_numeric_cells(body)
    title = "Northgate AI Stock Predictor — Technical Report"
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{htmllib.escape(title)}</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def main() -> int:
    if not SOURCE.exists():
        print(f"source report not found: {SOURCE}")
        print("Run `python scripts/fill_report.py` first.")
        return 1

    md_text = SOURCE.read_text(encoding="utf-8")

    # Strip the in-page table of contents if one was generated: it duplicates
    # the page numbers Edge is about to compute, and a stale one is worse than
    # none.
    md_text = re.sub(r"^\s*##\s*Table of contents.*?(?=^##\s)", "",
                     md_text, flags=re.S | re.M)

    browser = find_browser()
    if browser is None:
        print("No Chromium-based browser found for PDF rendering.")
        print("Looked for:")
        for p in BROWSERS:
            print(f"  {p}")
        print("Install Google Chrome or Microsoft Edge, or install a Python "
              "renderer (e.g. `pip install weasyprint`).")
        return 1

    html_doc = build_html(md_text)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        html_path = tmpdir / "report.html"
        html_path.write_text(html_doc, encoding="utf-8")
        out = tmpdir / "report.pdf"

        cmd = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={out}",
            html_path.as_uri(),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if not out.exists():
            print("PDF rendering failed.")
            print(f"  browser: {browser}")
            print(f"  exit   : {proc.returncode}")
            if proc.stderr.strip():
                print(f"  stderr : {proc.stderr.strip()[:500]}")
            return 1

        TARGET.write_bytes(out.read_bytes())

    size_kb = TARGET.stat().st_size / 1024
    # Cheap sanity check: a real PDF starts with %PDF and is not a few hundred
    # bytes, which is what a browser writes when it renders an empty page.
    if not TARGET.read_bytes()[:4] == b"%PDF" or size_kb < 40:
        print(f"Rendered file does not look like a valid PDF ({size_kb:.0f} KB).")
        return 1

    print(f"wrote {TARGET}")
    print(f"  source : {SOURCE.relative_to(ROOT)}")
    print(f"  size   : {size_kb:,.0f} KB")
    print(f"  words  : {len(md_text.split()):,}")
    if FIGURES_DIR.exists():
        n = len(list(FIGURES_DIR.glob("*.png")))
        print(f"  figures: {n} available under {FIGURES_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
