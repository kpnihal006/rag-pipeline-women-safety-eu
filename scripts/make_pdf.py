from __future__ import annotations

"""
scripts/make_pdf.py

Render the scientific report to PDF.

Markdown -> styled HTML -> headless Chrome -> PDF. Chrome is used rather than a
LaTeX toolchain because the report embeds PNG figures and wide comparison
tables, and a browser lays both out correctly without a 4 GB TeX install.

Figures are inlined as base64 data URIs so the intermediate HTML is a single
self-contained file; without that, Chrome's PDF pass silently drops images it
cannot resolve relative to a temp directory.

Usage:
    uv run python -m scripts.make_pdf
    uv run python -m scripts.make_pdf --input reports/SCIENTIFIC_REPORT.md \
                                      --output reports/SCIENTIFIC_REPORT.pdf
"""

import argparse
import base64
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome", "chromium", "chromium-browser",
]

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10.2pt; line-height: 1.55; color: #1d1d1f; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 21pt; margin: 0 0 4pt; letter-spacing: -0.4pt; }
h2 { font-size: 15pt; margin: 22pt 0 7pt; padding-bottom: 4pt;
     border-bottom: 1.5px solid #d2d2d7; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 15pt 0 5pt; color: #0b4f8a; page-break-after: avoid; }
h4 { font-size: 10.6pt; margin: 12pt 0 4pt; page-break-after: avoid; }
p  { margin: 0 0 8pt; }
ul, ol { margin: 0 0 9pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }
strong { font-weight: 600; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.8pt;
       background: #f5f5f7; padding: 1px 4px; border-radius: 3px; }
pre { background: #f5f5f7; border: 1px solid #e5e5ea; border-radius: 6px;
      padding: 9pt 11pt; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.45; }
table { border-collapse: collapse; width: 100%; margin: 9pt 0 13pt;
        font-size: 9pt; page-break-inside: avoid; }
th { background: #f5f5f7; text-align: left; font-weight: 600;
     border-bottom: 1.5px solid #c7c7cc; padding: 5pt 7pt; }
td { border-bottom: 1px solid #e5e5ea; padding: 5pt 7pt; vertical-align: top; }
tr:nth-child(even) td { background: #fafafa; }
blockquote { margin: 10pt 0; padding: 8pt 12pt; background: #fff8e6;
             border-left: 3px solid #e6a700; page-break-inside: avoid; }
blockquote p:last-child { margin-bottom: 0; }
img { max-width: 100%; height: auto; display: block; margin: 10pt auto 4pt;
      page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #d2d2d7; margin: 16pt 0; }
em { color: #4a4a4f; }
a { color: #0b4f8a; text-decoration: none; }
.title-block { border-bottom: 2.5px solid #1d1d1f; padding-bottom: 10pt;
               margin-bottom: 14pt; }
"""


def find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
        found = shutil.which(c)
        if found:
            return found
    return None


def inline_images(html: str, base: Path) -> tuple[str, int, int]:
    """Replace <img src="..."> with base64 data URIs."""
    embedded = missing = 0

    def repl(m: re.Match) -> str:
        nonlocal embedded, missing
        src = m.group(1)
        if src.startswith(("data:", "http://", "https://")):
            return m.group(0)
        path = (base / src).resolve()
        if not path.exists():
            missing += 1
            return m.group(0)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode()
        embedded += 1
        return m.group(0).replace(src, f"data:{mime};base64,{data}")

    return re.sub(r'<img[^>]+src="([^"]+)"', repl, html), embedded, missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the report to PDF")
    ap.add_argument("--input", default="reports/SCIENTIFIC_REPORT.md")
    ap.add_argument("--output", default="reports/SCIENTIFIC_REPORT.pdf")
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Report not found: {src}")

    try:
        import markdown
    except ImportError:
        sys.exit("Missing dependency. Install with: uv pip install markdown")

    chrome = find_chrome()
    if not chrome:
        sys.exit(
            "No Chrome/Chromium/Edge found. Install one, or convert manually:\n"
            "  pandoc reports/SCIENTIFIC_REPORT.md -o report.pdf"
        )

    text = src.read_text(encoding="utf-8")

    # Mermaid blocks are diagram source, not code. Chrome will not render them
    # without the mermaid runtime, so label them rather than dumping raw source.
    text = re.sub(
        r"```mermaid\n(.*?)```",
        lambda m: "**Diagram (Mermaid source):**\n\n```\n" + m.group(1) + "```",
        text, flags=re.DOTALL,
    )

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
    )
    body, embedded, missing = inline_images(body, src.parent)

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{src.stem}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "report.html"
        html_path.write_text(html, encoding="utf-8")
        if args.keep_html:
            kept = out.with_suffix(".html")
            kept.write_text(html, encoding="utf-8")
            print(f"  HTML kept → {kept}")

        # `--headless=new` is required on current Chrome: the legacy mode
        # hangs indefinitely on --print-to-pdf. --virtual-time-budget forces
        # the renderer to settle instead of waiting on idle callbacks.
        cmd = [
            chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
            "--disable-extensions", "--disable-background-networking",
            "--no-first-run", "--no-pdf-header-footer",
            "--virtual-time-budget=20000",
            f"--print-to-pdf={out}",
            f"--user-data-dir={tmp}/profile",
            f"file://{html_path}",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            proc = subprocess.CompletedProcess(cmd, 1, "", "chrome timed out")

    if not out.exists():
        print(proc.stderr[-1500:])
        sys.exit("Chrome did not produce a PDF.")

    size = out.stat().st_size
    print(f"Rendered  : {src}")
    print(f"Figures   : {embedded} embedded"
          + (f", {missing} MISSING" if missing else ""))
    print(f"PDF       : {out}  ({size/1024:.0f} KB)")
    if size < 20_000:
        print("WARNING: the PDF is suspiciously small — check it opens correctly.")


if __name__ == "__main__":
    main()
