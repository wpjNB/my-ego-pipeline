#!/usr/bin/env bash
#
# Download every model weight this project needs into one folder.
#
#   ./scripts/download_weights.sh                     # everything, with resume
#   ./scripts/download_weights.sh --only wilor,hawor  # subset
#   ./scripts/download_weights.sh --dry-run           # print the plan, fetch nothing
#   ./scripts/download_weights.sh --with-repos        # also clone third_party/*
#   DEST=/mnt/weights ./scripts/download_weights.sh   # different destination tree
#
# Air-gapped / mirrored hosts can point the three sources somewhere else (any
# scheme `wget` understands, including file://):
#
#   WILOR_BASE=file:///srv/mirror/wilor \
#   HAWOR_BASE=file:///srv/mirror/hawor \
#   VGGT_URL=file:///srv/mirror/vggt_omega_1b_416_reproduce.pt \
#       ./scripts/download_weights.sh
#
# Layout written (what the runners and scripts/doctor.py look for):
#
#   weights/
#   ├── wilor/      wilor_final.ckpt  model_config.yaml  detector.pt
#   ├── hawor/      checkpoints/hawor.ckpt  checkpoints/infiller.pt
#   ├── vggt-omega/ vggt_omega_1b_416_reproduce.pt
#   └── mano/       MANO_RIGHT.pkl            (licence-gated: never auto-fetched)
#
# Files are downloaded to "<name>.part" first and only moved into place once the
# size check passes, so an interrupted run never leaves a half file that looks
# usable. Re-running resumes with `wget -c` / `curl -C -`.
#
# After a real run, `scripts/verify_weights.sh` is executed automatically: it
# checks every file again (size + container magic) and exits non-zero if
# anything is missing or unusable.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-${ROOT}/weights}"
THIRD_PARTY="${THIRD_PARTY:-${ROOT}/third_party}"
# Source bases (override for a mirror; see the header).
WILOR_BASE="${WILOR_BASE:-https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models}"
HAWOR_BASE="${HAWOR_BASE:-https://huggingface.co/ThunderVVV/HaWoR/resolve/main}"
VGGT_URL="${VGGT_URL:-}"
ONLY="all"
DRY_RUN=0
WITH_REPOS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --dest) DEST="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --with-repos) WITH_REPOS=1; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

wants() {
    [[ "${ONLY}" == "all" ]] && return 0
    [[ ",${ONLY}," == *",$1,"* ]] && return 0
    [[ ",${ONLY}," == *",all,"* ]] && return 0
    return 1
}

# fetch <url> <destination> <label> <min_bytes> [required=1]
#   A failure of an optional asset is reported as [warn] and does not fail the run.
fetch() {
    local url="$1" dest="$2" label="$3" min_bytes="$4" required="${5:-1}"
    local part="${dest}.part"
    local fail_reason=""

    if [[ -f "${dest}" ]]; then
        local size
        size=$(wc -c <"${dest}")
        if [[ "${size}" -ge "${min_bytes}" ]]; then
            echo "  [skip]   ${label}: already present ($(human "${size}"))"
            return 0
        fi
        echo "  [redo]   ${label}: existing file is too small ($(human "${size}")), re-downloading"
        rm -f "${dest}"
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  [plan]   ${label}"
        echo "             ${url}"
        echo "          -> ${dest}"
        return 0
    fi

    mkdir -p "$(dirname "${dest}")"   # a dry run must not touch the filesystem
    echo "  [get]    ${label}"
    local ok=1
    case "${url}" in
        file://*|/*)
            # Local mirror. GNU wget refuses file:// ("Unsupported scheme"), so
            # handle it here and keep the script usable with either tool.
            local src="${url#file://}"
            if [[ ! -f "${src}" ]]; then
                fail_reason="mirror file not found (${src})"
            elif cp -f "${src}" "${part}"; then
                ok=0
            else
                fail_reason="cannot copy from ${src}"
            fi
            ;;
        *)
            if command -v wget >/dev/null 2>&1; then
                if wget -c --progress=dot:giga -O "${part}" "${url}"; then
                    ok=0
                else
                    fail_reason="wget failed (${url})"
                fi
            elif command -v curl >/dev/null 2>&1; then
                if curl -L -C - --fail --progress-bar -o "${part}" "${url}"; then
                    ok=0
                else
                    fail_reason="curl failed (${url})"
                fi
            else
                fail_reason="neither wget nor curl is installed"
            fi
            ;;
    esac

    if [[ "${ok}" != "0" ]]; then
        if [[ "${required}" == "1" ]]; then
            echo "  [FAIL]   ${label}: ${fail_reason}" >&2
        else
            echo "  [warn]   ${label}: ${fail_reason} (optional, ignored)" >&2
        fi
        # A transport that died before writing anything leaves a zero-byte .part;
        # drop it. A genuinely partial file is kept so `wget -c`/`curl -C -` can
        # resume it on the next run.
        if [[ -f "${part}" && ! -s "${part}" ]]; then
            rm -f "${part}"
        elif [[ -s "${part}" ]]; then
            echo "           partial download kept at ${part} (re-run to resume)" >&2
        fi
        if [[ "${required}" == "1" ]]; then
            return 1
        fi
        return 0
    fi

    local size
    size=$(wc -c <"${part}")
    if [[ "${size}" -lt "${min_bytes}" ]]; then
        if [[ "${required}" == "1" ]]; then
            echo "  [FAIL]   ${label}: only ${size} bytes (expected >= ${min_bytes})" >&2
            echo "           the URL may have moved - edit the URL in this script," >&2
            echo "           or download the file by hand; see doc_auto/setup.md" >&2
        else
            echo "  [warn]   ${label}: only ${size} bytes (optional, ignored)" >&2
        fi
        rm -f "${part}"
        [[ "${required}" == "1" ]] && return 1
        return 0
    fi
    mv "${part}" "${dest}"
    echo "  [ok]     ${label}: $(human "${size}")"
}

human() {
    local bytes="$1"
    if [[ "${bytes}" -lt 1024 ]]; then
        printf '%s B' "${bytes}"
    elif [[ "${bytes}" -ge 1073741824 ]]; then
        awk -v b="${bytes}" 'BEGIN{printf "%.2f GiB", b/1073741824}'
    elif [[ "${bytes}" -ge 1048576 ]]; then
        awk -v b="${bytes}" 'BEGIN{printf "%.1f MiB", b/1048576}'
    else
        awk -v b="${bytes}" 'BEGIN{printf "%.0f KiB", b/1024}'
    fi
}

clone_repo() {  # clone_repo <url> <dir> [extra git args...]
    local url="$1" dir="$2"; shift 2
    if [[ -d "${dir}/.git" ]]; then
        echo "  [skip]   ${dir} (already cloned)"
        return 0
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  [plan]   git clone $* ${url} ${dir}"
        return 0
    fi
    echo "  [clone]  ${url} -> ${dir}"
    git clone "$@" "${url}" "${dir}"
}

echo "======================================"
echo " Ego3D Action Model Weight Downloader"
echo "======================================"
echo " destination : ${DEST}"
echo " third_party : ${THIRD_PARTY}"
echo " selection   : ${ONLY}"

if [[ "${WITH_REPOS}" == "1" ]]; then
    echo
    echo "[0/4] Cloning the backend repositories (--recursive where upstream asks for it)"
    clone_repo "https://github.com/rolpotamias/WiLoR.git" "${THIRD_PARTY}/WiLoR" --recursive
    clone_repo "https://github.com/ThunderVVV/HaWoR.git" "${THIRD_PARTY}/HaWoR" --recursive
    clone_repo "https://github.com/facebookresearch/vggt.git" "${THIRD_PARTY}/VGGT-Omega"
fi

status=0

if wants wilor; then
    echo
    echo "[1/4] WiLoR (Phase 1 detector)"
    fetch "${WILOR_BASE}/wilor_final.ckpt" \
        "${DEST}/wilor/wilor_final.ckpt" "WiLoR checkpoint" 10000000 || status=1
    fetch "${WILOR_BASE}/model_config.yaml" \
        "${DEST}/wilor/model_config.yaml" "WiLoR model config" 100 || status=1
    # The YOLO detector is optional: this project detects with WiLoR's own model.
    fetch "${WILOR_BASE}/detector.pt" \
        "${DEST}/wilor/detector.pt" "WiLoR detector" 1000000 0 || status=1
fi

if wants hawor; then
    echo
    echo "[2/4] HaWoR (Phase 2 hand reconstruction)"
    fetch "${HAWOR_BASE}/hawor/checkpoints/hawor.ckpt" \
        "${DEST}/hawor/checkpoints/hawor.ckpt" "HaWoR checkpoint" 10000000 || status=1
    fetch "${HAWOR_BASE}/hawor/checkpoints/infiller.pt" \
        "${DEST}/hawor/checkpoints/infiller.pt" "HaWoR infiller" 1000000 || status=1
    # Also ships inside the HaWoR checkout, so a missing mirror copy is not fatal.
    fetch "${HAWOR_BASE}/hawor/model_config.yaml" \
        "${DEST}/hawor/checkpoints/model_config.yaml" "HaWoR model config" 100 0 || status=1
fi

if wants vggt; then
    echo
    echo "[3/4] VGGT-Omega (Phase 3 camera reconstruction)"
    VGGT_FILE="vggt_omega_1b_416_reproduce.pt"
    if [[ -n "${VGGT_URL}" ]]; then
        fetch "${VGGT_URL}" "${DEST}/vggt-omega/${VGGT_FILE}" "VGGT-Omega 416" 100000000 || status=1
    elif [[ "${DRY_RUN}" == "1" ]]; then
        echo "  [plan]   hf download facebook/VGGT-Omega ${VGGT_FILE} --local-dir ${DEST}/vggt-omega"
    elif command -v hf >/dev/null 2>&1; then
        mkdir -p "${DEST}/vggt-omega"
        hf download facebook/VGGT-Omega "${VGGT_FILE}" --local-dir "${DEST}/vggt-omega" || status=1
    else
        echo "  [FAIL]   the Hugging Face CLI is not installed" >&2
        echo "           pip install -U huggingface_hub && hf auth login" >&2
        echo "           then re-run: ./scripts/download_weights.sh --only vggt" >&2
        status=1
    fi
fi

if wants mano; then
    echo
    echo "[4/4] MANO (optional, licence-gated - cannot be scripted)"
    if [[ -f "${DEST}/mano/MANO_RIGHT.pkl" ]]; then
        echo "  [skip]   MANO_RIGHT.pkl already present"
    else
        cat <<'MANO'
  [manual] MANO_RIGHT.pkl
           1. register and accept the licence at https://mano.is.tue.mpg.de/
           2. download "Models & Code" and copy MANO_RIGHT.pkl to:
                weights/mano/MANO_RIGHT.pkl
           3. convert it once so this project can read it:
                python scripts/convert_mano.py --input weights/mano/MANO_RIGHT.pkl \
                    --output-dir weights/mano
           4. set `paths.mano_model: weights/mano` in configs/hot3d.yaml to get
              21-joint references instead of wrist-only ones.
           MANO_RIGHT alone is enough - the left hand is mirrored automatically.
MANO
    fi
fi

echo
echo "======================================"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo " Plan complete (nothing was downloaded)"
else
    echo " Download finished"
fi
echo "======================================"
for sub in wilor hawor vggt-omega; do
    if [[ -d "${DEST}/${sub}" ]]; then
        echo
        echo "${sub}:"
        ls -lh "${DEST}/${sub}" 2>/dev/null || true
    fi
done

if [[ "${DRY_RUN}" == "0" && -x "${ROOT}/scripts/verify_weights.sh" ]]; then
    echo
    "${ROOT}/scripts/verify_weights.sh" --dest "${DEST}" || status=1
fi

exit "${status}"
