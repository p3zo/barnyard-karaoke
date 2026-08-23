#!/bin/bash
#
# Download a YouTube video's audio and cut the excerpt used as a mosaicing target.
#
#   ./prepare_target.sh <name> <youtube_id> <start> <duration>
#
# Writes data/targets/<name>/audio.wav at 44.1 kHz, which is what analyze.py reads.
#
# The two targets used in the paper:
#
#   ./prepare_target.sh over_the_rainbow V1bFr2SWP1I 00:01:05 10   # 1:05-1:15
#   ./prepare_target.sh old_macdonald   _6HzoUcx3eo 00:00:15 20   # 0:15-0:35
#
# Requires yt-dlp, ffmpeg, and a JavaScript runtime (deno or node) for YouTube's
# challenge solver, without which most videos report themselves as unavailable.

set -euo pipefail

if [ "$#" -ne 4 ]; then
    cat <<'USAGE'
Usage: ./prepare_target.sh <name> <youtube_id> <start> <duration>

Writes data/targets/<name>/audio.wav at 44.1 kHz, which is what analyze.py reads.

The two targets used in the paper:
  ./prepare_target.sh over_the_rainbow V1bFr2SWP1I 00:01:05 10
  ./prepare_target.sh old_macdonald   _6HzoUcx3eo 00:00:15 20
USAGE
    exit 1
fi

NAME="$1"
YOUTUBE_ID="$2"
START="$3"
DURATION="$4"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGETS_DIR="$REPO_DIR/data/targets/$NAME"
FULL="$TARGETS_DIR/$YOUTUBE_ID.wav"
EXCERPT="$TARGETS_DIR/audio.wav"

mkdir -p "$TARGETS_DIR"

# YouTube requires solving a JS challenge; yt-dlp enables deno by default and needs to be
# told about any other runtime. Without one, videos wrongly report as unavailable.
if command -v deno >/dev/null 2>&1; then
    JS_RUNTIME=deno
elif command -v node >/dev/null 2>&1; then
    JS_RUNTIME=node
else
    echo "Need deno or node on PATH for YouTube's challenge solver." >&2
    exit 1
fi

# -- separates the ID from any leading-dash filter (e.g. _6HzoUcx3eo is fine, but IDs can
# start with a dash and would otherwise be read as an option).
yt-dlp \
    --js-runtimes "$JS_RUNTIME" \
    --remote-components ejs:github \
    --extract-audio \
    --audio-format wav \
    --output "$TARGETS_DIR/%(id)s.%(ext)s" \
    -- "$YOUTUBE_ID"

# -ss before -i seeks without decoding the skipped audio; -t after -i bounds the output.
ffmpeg -y -loglevel error -ss "$START" -i "$FULL" -t "$DURATION" -ar 44100 -ac 1 "$EXCERPT"
rm -f "$FULL"

echo "Wrote $EXCERPT ($DURATION s from $START of $YOUTUBE_ID)"
