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
# Sources
#   WiLoR      Hugging Face space rolpotamias/WiLoR        (wilor_final.ckpt ...)
#   HaWoR      Hugging Face ThunderVVV/HaWoR              (hawor.ckpt, infiller.pt ...)
#   VGGT-Omega ModelScope facebook/VGGT-Omega (master)    (vggt_omega_1b_416_reproduce.pt,
#             4.58 GB, FAIR Noncommercial Research License; the 512 and 256-text
#             checkpoints live in the same repo and are used as fallbacks)
#   MANO       licence-gated, printed as a manual step
#
# Layout written (what the runners and scripts/doctor.py look for):
#
#   weights/
#   |-- wilor/      wilor_final.ckpt  model_config.yaml  detector.pt
#   |-- hawor/      checkpoints/hawor.ckpt  checkpoints/infiller.pt
#   |-- vggt-omega/ vggt_omega_1b_416_reproduce.pt  configuration.json
#   `-- mano/       MANO_RIGHT.pkl            (licence-gated: never auto-fetched)
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
# Hugging Face is often unreachable from mainland China, so the mirror that the
# machine already configured via HF_ENDPOINT (usually https://hf-mirror.com) is
# tried first, then the official host, then hf-mirror. An explicit WILOR_BASE /
# HAWOR_BASE (a file:// mirror, a private proxy, ...) replaces the list entirely.
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
WILOR_BASE="${WILOR_BASE:-}"
HAWOR_BASE="${HAWOR_BASE:-}"

hf_bases() {
    local -a order=()
    if [[ -n "${HF_ENDPOINT:-}" && "${HF_ENDPOINT}" != *huggingface.co* ]]; then
        order+=("${HF_ENDPOINT%/}")
    fi
    order+=("https://huggingface.co")
    order+=("${HF_MIRROR%/}")
    printf '%s\n' "${order[@]}" | awk '!seen[$0]++'
}

wilor_sources() {  # wilor_sources <file>
    if [[ -n "${WILOR_BASE}" ]]; then
        printf '%s\n' "${WILOR_BASE%/}/$1"
        return
    fi
    while read -r base; do
        printf '%s\n' "${base}/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/$1"
    done < <(hf_bases)
}

hawor_sources() {  # hawor_sources <file>
    if [[ -n "${HAWOR_BASE}" ]]; then
        printf '%s\n' "${HAWOR_BASE%/}/$1"
        return
    fi
    while read -r base; do
        printf '%s\n' "${base}/ThunderVVV/HaWoR/resolve/main/$1"
    done < <(hf_bases)
}
# VGGT-Omega lives on ModelScope as `facebook/VGGT-Omega` (revision master,
# FAIR Noncommercial Research License, updated 2026-09-09). The repository holds:
#   vggt_omega_1b_416_reproduce.pt   4.58 GB   <- what configs/ asks for
#   vggt_omega_1b_512.pt             4.58 GB
#   vggt_omega_1b_256_text.pt        5.40 GB
#   configuration.json / LICENSE.txt / README.md
# The 416 reproduction file is the reference configuration's checkpoint, so it is
# fetched by default; the others are mirrors of the same download slot.
VGGT_MODEL_ID="${VGGT_MODEL_ID:-facebook/VGGT-Omega}"
VGGT_BASE="${VGGT_BASE:-https://www.modelscope.cn/models/${VGGT_MODEL_ID}/resolve/master}"
VGGT_FILE="${VGGT_FILE:-vggt_omega_1b_416_reproduce.pt}"
# Sizes measured on the hubs; each floor is overridable for small mirrors/tests.
WILOR_MIN_BYTES="${WILOR_MIN_BYTES:-2400000000}"
HAWOR_MIN_BYTES="${HAWOR_MIN_BYTES:-3100000000}"
INFILLER_MIN_BYTES="${INFILLER_MIN_BYTES:-400000000}"
# 4.26 GiB on ModelScope: 4.3 GB catches truncation and HTML error pages.
VGGT_MIN_BYTES="${VGGT_MIN_BYTES:-4300000000}"
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

# fetch <destination> <label> <min_bytes> [required=1] <url> [mirror_url...]
#   Mirrors are tried in order until one passes the size check, so a moved host
#   or an alternative provider needs no code change.
#   A failure of an optional asset is reported as [warn] and does not fail the run.
fetch() {
    local dest="$1" label="$2" min_bytes="$3" required="${4:-1}"
    local urls=()
    if [[ $# -ge 5 ]]; then
        urls=("${@:5}")
    fi
    if [[ "${#urls[@]}" -eq 0 ]]; then
        echo "  [FAIL]   ${label}: no source configured" >&2
        return 1
    fi
    local part="${dest}.part"
    local fail_reason=""
    local url

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
        for url in "${urls[@]}"; do
            echo "             ${url}"
        done
        echo "          -> ${dest}"
        return 0
    fi

    mkdir -p "$(dirname "${dest}")"   # a dry run must not touch the filesystem
    local ok=1
    for url in "${urls[@]}"; do
    echo "  [get]    ${label}"
    echo "             ${url}"
    ok=1
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
                # --tries/--timeout/--waitretry keep a stalled connection from
                # hanging a multi-GB download forever; -c resumes what is there.
                if wget -c --tries=3 --timeout=45 --waitretry=10 --progress=dot:giga \
                    -O "${part}" "${url}"; then
                    ok=0
                else
                    fail_reason="wget failed (${url})"
                fi
            elif command -v curl >/dev/null 2>&1; then
                if curl -L -C - --fail --retry 3 --retry-delay 5 --connect-timeout 30 \
                    --progress-bar -o "${part}" "${url}"; then
                    ok=0
                else
                    fail_reason="curl failed (${url})"
                fi
            else
                fail_reason="neither wget nor curl is installed"
            fi
            ;;
    esac
    if [[ "${ok}" == "0" ]]; then
        break
    fi
    echo "           ... trying the next mirror" >&2
    done

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

    local -a plain=()
    local arg
    for arg in "$@"; do
        [[ "${arg}" == "--recursive" ]] || plain+=("${arg}")
    done

    # GitHub and GitLab are both flaky from mainland China (GnuTLS resets mid
    # pack), so a clone is attempted in three escalating ways before giving up.
    for attempt in 1 2 3; do
        case "${attempt}" in
            1) rm -rf "${dir}"; git clone "$@" "${url}" "${dir}" && return 0 ;;
            2)  # smaller, protocol-tuned transfer
                rm -rf "${dir}"
                git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 \
                    clone --depth 1 "${plain[@]}" "${url}" "${dir}" && {
                    echo "  [warn]   ${dir}: shallow clone without submodules" >&2
                    echo "           finish later with: git -C ${dir} submodule update --init --recursive" >&2
                    return 0; } ;;
            3)  # configured mirror (e.g. GITHUB_MIRROR=https://ghproxy.net/)
                if [[ -n "${GITHUB_MIRROR:-}" ]]; then
                    rm -rf "${dir}"
                    git clone --depth 1 "${GITHUB_MIRROR%/}/${url}" "${dir}" && {
                        echo "  [warn]   ${dir}: cloned via ${GITHUB_MIRROR}" >&2
                        return 0; }
                fi ;;
        esac
    done
    if [[ -n "${GITHUB_MIRROR:-}" ]]; then
        echo "  [FAIL]   cannot clone ${url} (also tried ${GITHUB_MIRROR})" >&2
    else
        echo "  [FAIL]   cannot clone ${url}; retry later or set GITHUB_MIRROR" >&2
    fi
    return 1
}

echo "======================================"
echo " Ego3D Action Model Weight Downloader"
echo "======================================"
echo " destination : ${DEST}"
echo " third_party : ${THIRD_PARTY}"
echo " selection   : ${ONLY}"

status=0

if [[ "${WITH_REPOS}" == "1" ]]; then
    echo
    echo "[0/4] Cloning the backend repositories (--recursive where upstream asks for it)"
    clone_repo "https://github.com/rolpotamias/WiLoR.git" "${THIRD_PARTY}/WiLoR" --recursive || status=1
    clone_repo "https://github.com/ThunderVVV/HaWoR.git" "${THIRD_PARTY}/HaWoR" --recursive || status=1
    # VGGT-Omega has its own repository; facebookresearch/vggt is the original
    # release and works as a fallback (VGGT_REPO can point at either).
    clone_repo "${VGGT_REPO:-https://github.com/facebookresearch/vggt-omega.git}" \
        "${THIRD_PARTY}/VGGT-Omega" || status=1
fi

if wants wilor; then
    echo
    echo "[1/4] WiLoR (Phase 1 detector)"
    # 2.39 GiB on the hub
    mapfile -t WILOR_CKPT < <(wilor_sources wilor_final.ckpt)
    fetch "${DEST}/wilor/wilor_final.ckpt" "WiLoR checkpoint" "${WILOR_MIN_BYTES}" 1 \
        "${WILOR_CKPT[@]}" || status=1
    mapfile -t WILOR_CFG < <(wilor_sources model_config.yaml)
    fetch "${DEST}/wilor/model_config.yaml" "WiLoR model config" 100 1 \
        "${WILOR_CFG[@]}" || status=1
    # The YOLO detector is optional: this project detects with WiLoR's own model.
    mapfile -t WILOR_DET < <(wilor_sources detector.pt)
    fetch "${DEST}/wilor/detector.pt" "WiLoR detector" 1000000 0 \
        "${WILOR_DET[@]}" || status=1
fi

if wants hawor; then
    echo
    echo "[2/4] HaWoR (Phase 2 hand reconstruction)"
    # 3.04 GiB / 399 MiB on the hub
    mapfile -t HAWOR_CKPT < <(hawor_sources hawor/checkpoints/hawor.ckpt)
    fetch "${DEST}/hawor/checkpoints/hawor.ckpt" "HaWoR checkpoint" "${HAWOR_MIN_BYTES}" 1 \
        "${HAWOR_CKPT[@]}" || status=1
    mapfile -t HAWOR_INF < <(hawor_sources hawor/checkpoints/infiller.pt)
    fetch "${DEST}/hawor/checkpoints/infiller.pt" "HaWoR infiller" "${INFILLER_MIN_BYTES}" 1 \
        "${HAWOR_INF[@]}" || status=1
    # Also ships inside the HaWoR checkout, so a missing mirror copy is not fatal.
    mapfile -t HAWOR_CFG < <(hawor_sources hawor/model_config.yaml)
    fetch "${DEST}/hawor/checkpoints/model_config.yaml" "HaWoR model config" 100 0 \
        "${HAWOR_CFG[@]}" || status=1
fi

if wants vggt; then
    echo
    echo "[3/4] VGGT-Omega (Phase 3 camera reconstruction)"
    # ModelScope resolve endpoint, then the API form, then the same two for the
    # 512 checkpoint (identical slot), then the provider CLIs.
    VGGT_SOURCES=()
    if [[ -n "${VGGT_URL}" ]]; then
        VGGT_SOURCES+=("${VGGT_URL}")
    fi
    for revision in master main; do
        VGGT_SOURCES+=("https://www.modelscope.cn/models/${VGGT_MODEL_ID}/resolve/${revision}/${VGGT_FILE}")
    done
    VGGT_SOURCES+=("https://www.modelscope.cn/api/v1/models/${VGGT_MODEL_ID}/repo?Revision=master&FilePath=${VGGT_FILE}")
    if [[ "${VGGT_FILE}" != "vggt_omega_1b_512.pt" ]]; then
        VGGT_SOURCES+=("${VGGT_BASE}/vggt_omega_1b_512.pt")
    fi

    # 4.26 GiB on ModelScope
    if fetch "${DEST}/vggt-omega/${VGGT_FILE}" "VGGT-Omega ${VGGT_FILE}" "${VGGT_MIN_BYTES}" 1 \
        "${VGGT_SOURCES[@]}"; then
        :
    elif [[ "${DRY_RUN}" == "1" ]]; then
        :
    elif command -v modelscope >/dev/null 2>&1; then
        echo "  [get]    via the ModelScope CLI"
        mkdir -p "${DEST}/vggt-omega"
        if modelscope download --model "${VGGT_MODEL_ID}" "${VGGT_FILE}" \
            --local_dir "${DEST}/vggt-omega"; then
            # The CLI sometimes nests the checkout; normalise the layout so the
            # runners find the checkpoint where they expect it.
            if [[ ! -f "${DEST}/vggt-omega/${VGGT_FILE}" ]]; then
                nested=$(find "${DEST}/vggt-omega" -name "${VGGT_FILE}" -type f | head -1)
                [[ -n "${nested}" ]] && mv -f "${nested}" "${DEST}/vggt-omega/${VGGT_FILE}"
            fi
        else
            echo "  [FAIL]   modelscope download failed" >&2
            status=1
        fi
    elif command -v hf >/dev/null 2>&1; then
        echo "  [get]    via the Hugging Face CLI"
        mkdir -p "${DEST}/vggt-omega"
        hf download facebook/VGGT-Omega "${VGGT_FILE}" --local-dir "${DEST}/vggt-omega" || status=1
    else
        echo "  [FAIL]   no ModelScope/HF CLI and every direct URL failed" >&2
        echo "           pip install modelscope   # then: modelscope download --model ${VGGT_MODEL_ID} ${VGGT_FILE}" >&2
        status=1
    fi

    # Tiny files that ship with the checkpoint; useful provenance, never fatal.
    fetch "${DEST}/vggt-omega/configuration.json" "VGGT-Omega configuration.json" 10 0 \
        "https://www.modelscope.cn/models/${VGGT_MODEL_ID}/resolve/master/configuration.json" || true
    fetch "${DEST}/vggt-omega/LICENSE.txt" "VGGT-Omega licence" 100 0 \
        "https://www.modelscope.cn/models/${VGGT_MODEL_ID}/resolve/master/LICENSE.txt" || true
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
