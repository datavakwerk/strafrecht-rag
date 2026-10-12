#!/usr/bin/env bash
# Record the README GIFs with asciinema + agg (brew install asciinema agg ffmpeg).
# The LLM stage is compressed in post (scripts/compress_cast.py); everything else is real time.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=docs/img
TMP=$(mktemp -d)
SIZE=${SIZE:-132x48}
THEME=${THEME:-monokai}
FONT=${FONT:-15}

rec() {  # name, rows, command...
  local name=$1 rows=$2; shift 2
  asciinema rec --window-size "${SIZE%x*}x${rows}" --overwrite -c "$*" "$TMP/$name.cast"
  uv run python scripts/compress_cast.py "$TMP/$name.cast" "$TMP/$name.fast.cast"
  agg --theme "$THEME" --font-size "$FONT" --last-frame-duration 4 "$TMP/$name.fast.cast" "$OUT/$name.gif"
  ffmpeg -y -loglevel error -i "$OUT/$name.gif" -vf reverse -frames:v 1 -update 1 "$OUT/$name.png"
  echo "→ $OUT/$name.gif  $OUT/$name.png"
}

ONLY=${1:-}
rec_if() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]] && rec "$@" || true; }

rec_if ask       48 uv run strafrecht-rag ask "'Welke uitspraken gaan over verduistering door een bestuurder van een stichting?'"
rec_if filtered  44 uv run strafrecht-rag ask "'Welke uitspraken gaan over handel in vuurwapens en munitie?'" --type Rechtbank --van 2024-01-01
rec_if no-source 30 uv run strafrecht-rag ask "'Wat zegt de rechter over de octrooiaanvraag voor een kernfusiereactor?'"
rec_if eval      40 uv run strafrecht-rag eval --no-llm
