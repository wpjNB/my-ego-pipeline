#!/usr/bin/env bash
# Download official HOT3D-Clips (bop-benchmark/hot3d) from the hf-mirror.
#
# Unlike data/hot3d_clips (the LafouCC mirror where the hand reference is
# broken - see data_notes in import_hot3d_clips.py), the official repo hosts:
#   train_aria / train_quest3   full GT: <frame>.hands.json (UmeTrack + MANO
#                               poses, 2D boxes), objects, per-stream camera
#                               calibration incl. FISHEYE624 intrinsics
#   test_aria / test_quest3     same minus the hand/object GT (challenge splits)
#   clip_definitions.json       clip -> source sequence + per-frame timestamps
#   clip_splits.json            train / test_ht_pose / test_ht_shape / test_bop
# Each clip is one tar (~100 MB, 150 frames, 5 s).
#
# The connection to huggingface.co is blocked on this host; hf-mirror.com is
# used instead. Files are fetched with curl --retry and resumed with -C - until
# the size matches the server, several in flight; complete files are skipped,
# so re-running is always safe.
#
# Usage:
#   bash scripts/download_bop_hot3d.sh --list-splits
#   bash scripts/download_bop_hot3d.sh                         # meta + default demo clip (1991)
#   CLIPS="1991 1849 2050" bash scripts/download_bop_hot3d.sh  # explicit clip ids
#   SPLIT=train_quest3 COUNT=3 bash scripts/download_bop_hot3d.sh
#   DEST=/mnt/data/hot3d_official bash scripts/download_bop_hot3d.sh
#
# Clip ids are the global indices (clip-<id>.tar); they are listed in
# clip_splits.json and detailed in clip_definitions.json (sequence, device,
# per-frame timestamps) - both are downloaded by this script.
set -uo pipefail

REPO="datasets/bop-benchmark/hot3d"
BASE="https://hf-mirror.com/${REPO}/resolve/main"
TREE_API="https://hf-mirror.com/api/${REPO}/tree/main?recursive=true"
DEST="${DEST:-data/hot3d_official}"
SPLIT="${SPLIT:-train_aria}"
CLIPS="${CLIPS:-1991}"
COUNT="${COUNT:-0}"
PARALLEL="${PARALLEL:-3}"
PY="${PY:-python3}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

size_of() {  # expected size of a repo file (bytes), empty when unknown
    # LFS files announce x-linked-size in the 302; small files are served
    # directly and only carry the final content-length.
    curl -sIL --max-time 60 "$1" | tr -d '\r' \
        | awk 'tolower($1)=="x-linked-size:" {l=$2} tolower($1)=="content-length:" {c=$2} END {print (l!="" ? l : c)}'
}

http_code() {  # status of the final response (200 = exists)
    curl -sIL --max-time 60 -o /dev/null -w '%{http_code}' "$1"
}

fetch() {  # fetch <repo-relative-path> <dest-path>; skips when complete
    local rel="$1" out="$2"
    local url="$BASE/$rel"
    local code; code=$(http_code "$url")
    if [[ "$code" != "200" ]]; then
        log "ERROR: $rel -> HTTP $code"; return 1
    fi
    local want; want=$(size_of "$url")
    if [[ -f "$out" && "$(wc -c < "$out")" == "$want" ]]; then
        log "skip $rel ($(( want / 1048576 )) MB, complete)"
        return 0
    fi
    mkdir -p "$(dirname "$out")"
    for attempt in $(seq 1 15); do
        curl -sfL --retry 3 --retry-delay 2 --max-time 1800 -C - "$url" -o "$out" \
            || { sleep $((attempt < 5 ? 2 : 10)); continue; }
        want=$(size_of "$url")
        [[ -z "$want" || "$(wc -c < "$out")" == "$want" ]] && return 0
        sleep $((attempt < 5 ? 2 : 10))
    done
    log "ERROR: $rel did not complete after 15 attempts"; return 1
}

# --- worker mode -------------------------------------------------------------
if [[ "${1:-}" == "--fetch-one" ]]; then
    rel="$2"
    fetch "$rel" "$DEST/${rel}"
    exit $?
fi

# --- meta files --------------------------------------------------------------
log "destination: $DEST  (split: $SPLIT)"
for meta in clip_splits.json clip_definitions.json README.md; do
    fetch "$meta" "$DEST/$meta" || true
done

if [[ "${1:-}" == "--list-splits" ]]; then
    "$PY" - "$DEST/clip_splits.json" <<'EOF'
import json, sys
splits = json.load(open(sys.argv[1]))
import pathlib
for name, val in splits.items():
    if isinstance(val, dict):
        total = sum(len(v) for v in val.values())
        parts = ", ".join(f"{dev}: {len(ids)}" for dev, ids in val.items())
        print(f"  {name:14s} {total:5d} clips ({parts})")
    else:
        print(f"  {name:14s} {len(val):5d} clips")
print("\nfolders on HF: train_aria, train_quest3 (full GT) | test_aria, test_quest3 (GT removed)")
EOF
    exit 0
fi

# --- resolve the clip id list ------------------------------------------------
if [[ "$COUNT" -gt 0 ]]; then
    CLIPS=$("$PY" - "$DEST/clip_splits.json" "$SPLIT" "$COUNT" <<'EOF'
import json, sys
splits = json.load(open(sys.argv[1]))
split, count = sys.argv[2], int(sys.argv[3])
key = "train" if split.startswith("train") else "test_bop"
device = "Quest3" if "quest3" in split else "Aria"
ids = splits[key][device] if isinstance(splits.get(key), dict) else splits[key]
print(" ".join(str(i) for i in sorted(ids)[:count]))
EOF
)
    log "COUNT=$COUNT -> first clips of $SPLIT: $CLIPS"
fi

log "clips to fetch: $CLIPS"
fail=0
for cid in $CLIPS; do
    rel=$(printf "%s/clip-%06d.tar" "$SPLIT" "$cid")
    fetch "$rel" "$DEST/$rel" &
    while [[ $(jobs -rp | wc -l) -ge "$PARALLEL" ]]; do wait -n; done
done
wait

# verify: each tar lists exactly one clip's worth of files
for cid in $CLIPS; do
    tar_path="$DEST/$(printf "%s/clip-%06d.tar" "$SPLIT" "$cid")"
    if [[ -f "$tar_path" ]] && tar -tf "$tar_path" > /dev/null 2>&1; then
        n=$(tar -tf "$tar_path" | grep -c '\.info\.json$')
        log "ok: $(basename "$tar_path") ($n frames)"
    else
        log "MISSING/BROKEN: $tar_path"; fail=1
    fi
done

log "done. next: inspect with"
log "  tar -tf $DEST/$SPLIT/clip-$(printf '%06d' "${CLIPS%% *}").tar | head"
log "  (official frames: <id>.image_214-1.jpg is the 1408x1408 fisheye RGB;"
log "   GT hands: <id>.hands.json with mano_pose.thetas/wrist_xform and"
log "   umetrack_pose; UmeTrack is the primary annotation.)"
exit $fail
