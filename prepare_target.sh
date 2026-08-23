#!/bin/bash
#
# Download a YouTube video's audio and cut the excerpt used as a mosaicing target.
#
#   ./prepare_target.sh <youtube_id> <start> <duration>
#
# Writes src/targets/short_<youtube_id>.wav at 44.1 kHz, which is what notebook 2 reads.
#
# The two targets used in the paper:
#
#   ./prepare_target.sh V1bFr2SWP1I 00:01:05 10   # Somewhere Over the Rainbow, 1:05-1:15
#   ./prepare_target.sh _6HzoUcx3eo 00:00:15 20   # Old Macdonald Had A Farm, 0:15-0:35
#
# Requires yt-dlp and ffmpeg.

set -euo pipefail

if [ "$#" -ne 3 ]; then
    cat <<'USAGE'
Usage: ./prepare_target.sh <youtube_id> <start> <duration>

Writes src/targets/short_<youtube_id>.wav at 44.1 kHz, which is what notebook 2 reads.

The two targets used in the paper:
  ./prepare_target.sh V1bFr2SWP1I 00:01:05 10   # Somewhere Over the Rainbow, 1:05-1:15
  ./prepare_target.sh _6HzoUcx3eo 00:00:15 20   # Old Macdonald Had A Farm, 0:15-0:35
USAGE
    exit 1
fi

YOUTUBE_ID="$1"
START="$2"
DURATION="$3"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGETS_DIR="$REPO_DIR/src/targets"
FULL="$TARGETS_DIR/$YOUTUBE_ID.wav"
EXCERPT="$TARGETS_DIR/short_$YOUTUBE_ID.wav"

mkdir -p "$TARGETS_DIR"

# -- separates the ID from any leading-dash filter (e.g. _6HzoUcx3eo is fine, but IDs can
# start with a dash and would otherwise be read as an option).
yt-dlp \
    --extract-audio \
    --audio-format wav \
    --output "$TARGETS_DIR/%(id)s.%(ext)s" \
    -- "$YOUTUBE_ID"

# -ss before -i seeks without decoding the skipped audio; -t after -i bounds the output.
ffmpeg -y -loglevel error -ss "$START" -i "$FULL" -t "$DURATION" -ar 44100 -ac 1 "$EXCERPT"

echo "Wrote $EXCERPT ($DURATION s from $START of $YOUTUBE_ID)"
