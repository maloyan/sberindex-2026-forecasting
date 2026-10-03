#!/usr/bin/env bash
# Build report/report.pdf from report/report.md (pandoc + xelatex) and presentation/slides.pdf
# from presentation/index.html (headless Chrome). Needs pandoc, a TeX distribution and Chrome.
set -euo pipefail
cd "$(dirname "$0")/../report"
pandoc report.md -o report.pdf --pdf-engine=xelatex \
  -V mainfont="PT Serif" -V sansfont="PT Sans" -V monofont="Menlo" -V fontsize=10pt \
  -V geometry:margin=1.8cm -V colorlinks=true -V linkcolor=blue \
  -V tables=true --resource-path=. -H header.tex
echo "report/report.pdf"
cd ..
.venv/bin/python src/build_site.py     # presentation/index.html, slides.html, copies figures and report.pdf
cd presentation
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf=slides.pdf \
    "file://$(pwd)/slides.html" 2>/dev/null
  echo "presentation/slides.pdf"
fi
