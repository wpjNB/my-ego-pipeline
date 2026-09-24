#!/usr/bin/env bash
#
# Install the MANO model where every consumer expects to find it.
#
#   ./scripts/install_mano.sh --from ~/Downloads/mano            # copy
#   ./scripts/install_mano.sh --from ~/Downloads/mano --link      # symlink instead
#   ./scripts/install_mano.sh --from ~/Downloads/mano --dry-run
#
# MANO is licence-gated (https://mano.is.tue.mpg.de/), so it cannot be
# downloaded by a script - this one only places files you already have. Four
# independent consumers look for it under four different paths:
#
#   third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl           HaWoR run_mano  (required)
#   third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl  HaWoR run_mano_left
#   weights/mano/MANO_RIGHT.pkl                                our FK / convert_mano.py
#   third_party/WiLoR/mano_data/MANO_RIGHT.pkl                 only if you also run
#                                                              WiLoR's own 3D model
#                                                              (Phase 1 needs just the detector)
#
# --from may be:
#   * the official archive (mano_v1_2.zip) - it is unpacked and the two pickles
#     are located automatically,
#   * a directory containing MANO_RIGHT.pkl / MANO_LEFT.pkl,
#   * the RIGHT pickle itself.
# MANO_LEFT is optional: HaWoR can run with the right model plus its
# `fix_shapedirs` workaround, and our own FK mirrors the right model when the
# left one is absent.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${THIRD_PARTY:-${ROOT}/third_party}"
DEST_ROOT="${DEST_ROOT:-${ROOT}/weights}"
FROM=""
MODE="copy"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --link) MODE="link"; shift ;;
        --copy) MODE="copy"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --third-party) THIRD_PARTY="$2"; shift 2 ;;
        --dest-root) DEST_ROOT="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${FROM}" ]]; then
    echo "error: --from <dir-with-MANO_RIGHT.pkl> is required" >&2
    echo "       register and download MANO at https://mano.is.tue.mpg.de/" >&2
    exit 2
fi

EXTRACT_DIR=""
# The EXIT trap must not leak a status: a bare `[[ ... ]] && cmd` returns 1 when
# the test fails, which would replace the script's real exit code.
cleanup() {
    if [[ -n "${EXTRACT_DIR}" && -d "${EXTRACT_DIR}" ]]; then
        rm -rf "${EXTRACT_DIR}"
    fi
    return 0
}
trap cleanup EXIT

if [[ -f "${FROM}" && "${FROM}" == *.zip ]]; then
    if ! command -v unzip >/dev/null 2>&1; then
        echo "error: '${FROM}' is a zip but unzip is not installed" >&2
        exit 2
    fi
    EXTRACT_DIR="$(mktemp -d)"
    echo "unpacking ${FROM} ..."
    unzip -q -o "${FROM}" -d "${EXTRACT_DIR}"
    RIGHT="$(find "${EXTRACT_DIR}" -name 'MANO_RIGHT.pkl' -type f | head -1)"
    LEFT="$(find "${EXTRACT_DIR}" -name 'MANO_LEFT.pkl' -type f | head -1)"
    if [[ -z "${RIGHT}" ]]; then
        echo "error: no MANO_RIGHT.pkl inside ${FROM} (looked in the whole archive)" >&2
        exit 1
    fi
elif [[ -d "${FROM}" ]]; then
    RIGHT="${FROM}/MANO_RIGHT.pkl"
    LEFT="${FROM}/MANO_LEFT.pkl"
elif [[ -f "${FROM}" ]]; then
    RIGHT="${FROM}"
    LEFT="$(dirname "${FROM}")/MANO_LEFT.pkl"
else
    echo "error: --from '${FROM}' does not exist" >&2
    exit 2
fi

# A MANO pickle is a chumpy pickle: >= 1 MB and starts with the pickle opcode.
check_pickle() {
    local file="$1" label="$2" magic size
    [[ -f "${file}" ]] || { echo "  [missing] ${label}: ${file}"; return 1; }
    size=$(wc -c <"${file}")
    magic=$(head -c 1 "${file}" | od -An -tx1 | tr -d ' \n')
    if [[ "${size}" -lt 1000000 ]]; then
        echo "  [invalid] ${label}: only ${size} bytes - not a MANO model" >&2
        return 1
    fi
    if [[ "${magic}" != "80" ]]; then
        echo "  [invalid] ${label}: not a pickle (first byte 0x${magic})" >&2
        return 1
    fi
    echo "  [ok]      ${label}: ${size} bytes, pickle"
    return 0
}

echo "======================================"
echo " MANO installer"
echo "======================================"
echo " source     : ${FROM}"
echo " mode       : ${MODE}"
echo " third_party: ${THIRD_PARTY}"
echo
echo "source files:"
have_right=0
have_left=0
check_pickle "${RIGHT}" "MANO_RIGHT.pkl" && have_right=1 || true
check_pickle "${LEFT}" "MANO_LEFT.pkl" && have_left=1 || true
if [[ "${have_right}" == "0" ]]; then
    echo "error: MANO_RIGHT.pkl is required" >&2
    exit 1
fi

install_one() {  # install_one <source> <destination>
    local src="$1" dst="$2"
    if [[ -f "${dst}" ]]; then
        echo "  [skip]   ${dst#"${ROOT}/"} (already present)"
        return 0
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  [plan]   ${src} -> ${dst#"${ROOT}/"} (${MODE})"
        return 0
    fi
    mkdir -p "$(dirname "${dst}")"
    if [[ "${MODE}" == "link" ]]; then
        ln -sfn "$(readlink -f "${src}")" "${dst}"
    else
        cp -f "${src}" "${dst}"
    fi
    echo "  [${MODE}]   ${dst#"${ROOT}/"}"
}

echo
echo "installing:"
install_one "${RIGHT}" "${THIRD_PARTY}/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl"
install_one "${RIGHT}" "${DEST_ROOT}/mano/MANO_RIGHT.pkl"
install_one "${RIGHT}" "${THIRD_PARTY}/WiLoR/mano_data/MANO_RIGHT.pkl"
if [[ "${have_left}" == "1" ]]; then
    install_one "${LEFT}" "${THIRD_PARTY}/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl"
    install_one "${LEFT}" "${THIRD_PARTY}/WiLoR/mano_data/MANO_LEFT.pkl"
    install_one "${LEFT}" "${DEST_ROOT}/mano/MANO_LEFT.pkl"
else
    echo "  [warn]   MANO_LEFT.pkl not provided - HaWoR uses its fix_shapedirs"
    echo "           workaround, and our FK mirrors the right-hand model."
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo
    echo "plan complete (nothing was installed)"
    exit 0
fi

echo
echo "next: convert the pickle once so this project's own forward kinematics can"
echo "      read it (the official pickle needs chumpy, the .npz does not):"
echo
echo "  conda run -n ego3d_hawor python scripts/convert_mano.py \\"
echo "      --input ${DEST_ROOT}/mano/MANO_RIGHT.pkl --output-dir ${DEST_ROOT}/mano"
echo
echo "  # then, to get 21-joint HOT3D references instead of wrist-only ones:"
echo "  #   configs/hot3d.yaml ->  paths.mano_model: weights/mano"

if [[ -f "${DEST_ROOT}/mano/MANO_RIGHT.npz" ]]; then
    echo "  (a .npz is already present - nothing to convert)"
elif command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | awk '{print $1}' | grep -qx ego3d_hawor; then
    echo
    echo "converting now with the ego3d_hawor environment..."
    conda run -n ego3d_hawor python "${ROOT}/scripts/convert_mano.py" \
        --input "${DEST_ROOT}/mano/MANO_RIGHT.pkl" --output-dir "${DEST_ROOT}/mano" \
        && echo "  [ok]      ${DEST_ROOT#"${ROOT}/"}/mano/MANO_RIGHT.npz" \
        || echo "  [warn]   conversion failed - run the command above manually" >&2
fi
