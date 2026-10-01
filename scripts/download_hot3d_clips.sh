#!/usr/bin/env bash
# Download the HOT3D-Clips benchmark subset from the hf-mirror.com mirror of
# LafouCC/hot3d-full (a re-upload of Meta's HOT3D-Clips: per clip video.mp4,
# camera.json, hands_pose.json, clip_info.json - no intrinsics, UmeTrack-only
# hand poses, so the benchmark runs at wrist level).
#
# The mirror connection is flaky, so every file is fetched with curl --retry
# and resumed with -C - until complete, several files in flight. Files already
# fully present (size matches the server) are skipped, so the script is
# idempotent - just re-run it until verification reports nothing missing.
#
# Usage:
#   bash scripts/download_hot3d_clips.sh                 # download to data/hot3d_clips
#   bash scripts/download_hot3d_clips.sh DEST PARALLEL
set -uo pipefail

REPO="datasets/LafouCC/hot3d-full"
BASE="https://hf-mirror.com/${REPO}/resolve/main"
API="https://hf-mirror.com/api/${REPO}/tree/main"
DEST="${1:-data/hot3d_clips}"
PARALLEL="${2:-4}"
FILES_PER_CLIP=(clip_info.json camera.json hands_pose.json video.mp4)
# The whole public validation split plus training sequences from other
# participants for diversity (override with TRAIN_SEQS="...").
VALID_SEQ="valid/P0015_e7458eb3"
TRAIN_SEQS="${TRAIN_SEQS:-train/P0001_10a27bf7 train/P0002_2ea9af5b}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# --- worker mode: download_hot3d_clips.sh --fetch-one <clip> <file> ---------
# Must run before DEST is parsed from $1, which is "--fetch-one" here.
if [[ "${1:-}" == "--fetch-one" ]]; then
    DEST="${HOT3D_DEST:-data/hot3d_clips}"
    clip="$2" f="$3"
    out="$DEST/$clip/$f"
    url="$BASE/$clip/$f"
    expected() {
        curl -sIL --max-time 30 "$url" | tr -d '\r' \
            | awk 'tolower($1)=="content-length:" {print $2}' | tail -1
    }
    want=$(expected)
    if [[ -n "$want" && -f "$out" && "$(wc -c < "$out")" == "$want" ]]; then
        exit 0
    fi
    mkdir -p "$(dirname "$out")"
    for attempt in $(seq 1 15); do
        curl -sfL --retry 3 --retry-delay 2 --max-time 900 -C - "$url" -o "$out" || {
            sleep $((attempt < 5 ? 2 : 10)); continue; }
        want=$(expected)
        [[ -z "$want" || "$(wc -c < "$out")" == "$want" ]] && exit 0
    done
    log "FAILED: $clip/$f"
    exit 1
fi

# --- listing ---------------------------------------------------------------
api_list() { # api_list <path>
    local path="$1" attempt
    for attempt in $(seq 1 10); do
        curl -sfL --max-time 60 "$API/$path?limit=1000" \
            | python3 -c "import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    exit(1)
for x in d:
    if x['type']=='directory': print(x['path'])" && return 0
        sleep $((attempt * 3))
    done
    return 1
}

CLIP_LIST=/tmp/hot3d_clip_list.txt
: > "$CLIP_LIST"
for seq in $VALID_SEQ $TRAIN_SEQS; do
    log "listing clips of $seq"
    # The tree API already returns full paths (<split>/<sequence>/clip-NNNNNN).
    api_list "$seq" >> "$CLIP_LIST" || { log "cannot list $seq"; exit 1; }
done
total_clips=$(wc -l < "$CLIP_LIST")
log "$total_clips clips queued, $PARALLEL files in flight"

# --- parallel download: one worker invocation per (clip, file) -------------
: > /tmp/hot3d_download_jobs.txt
while IFS= read -r clip; do
    for f in "${FILES_PER_CLIP[@]}"; do
        printf '%s %s\n' "$clip" "$f" >> /tmp/hot3d_download_jobs.txt
    done
done < "$CLIP_LIST"

HOT3D_DEST="$DEST" xargs -P "$PARALLEL" -n 2 "$0" --fetch-one < /tmp/hot3d_download_jobs.txt


# --- verification ----------------------------------------------------------
log "verifying"
missing=0
while IFS= read -r clip; do
    for f in "${FILES_PER_CLIP[@]}"; do
        if [[ ! -s "$DEST/$clip/$f" ]]; then
            log "missing: $clip/$f"
            missing=$((missing + 1))
        fi
    done
done < "$CLIP_LIST"
log "verification: $missing missing files across $total_clips clips"
exit $((missing > 0))
