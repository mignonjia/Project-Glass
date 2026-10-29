#!/usr/bin/env bash
# Build all PRISM figures: tectonic -> PDF, then PyMuPDF -> 300 dpi PNG.
# Usage: ./build.sh [PYTHON_WITH_PYMUPDF] [name ...]
set -euo pipefail
cd "$(dirname "$0")"
PY="${1:-python3}"; shift || true
OUT=../figures
mkdir -p "$OUT"
files=("$@")
[ ${#files[@]} -eq 0 ] && files=($(ls *.tex | grep -v prism_style))
for f in "${files[@]}"; do
  name="${f%.tex}"
  tectonic -X compile --outdir "$OUT" "$name.tex" >/dev/null 2>"$OUT/$name.log" || { cat "$OUT/$name.log"; exit 1; }
  rm -f "$OUT/$name.log"
  "$PY" -c "import pymupdf,sys; d=pymupdf.open(sys.argv[1]); d[0].get_pixmap(dpi=300).save(sys.argv[2])" "$OUT/$name.pdf" "$OUT/$name.png"
  echo "built $name"
done
