#!/usr/bin/env bash
#
# Verify the downloaded model weights: existence, size and container format.
#
#   ./scripts/verify_weights.sh
#   ./scripts/verify_weights.sh --dest /mnt/weights
#   ./scripts/verify_weights.sh --quiet      # only problems
#
# Why not just `ls`? The classic failure is `wget` reporting success on a
# truncated or HTML-error file, which only blows up later inside `torch.load`.
# So every file is checked for a size floor and for the magic bytes of the
# container it is supposed to be:
#
#   torch.save (modern) -> ZIP      "PK\x03\x04"
#   torch.save (legacy) -> pickle   "\x80"
#   safetensors         -> 8-byte little-endian header length + JSON
#
# Exit code 0 only when every *required* file passes. MANO is listed as optional
# (it is licence-gated and only needed for 21-joint references).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-${ROOT}/weights}"
QUIET=0
STRICT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        --quiet) QUIET=1; shift ;;
        --strict) STRICT=1; shift ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# name|path|min_bytes|expected_format|required(1/0)|sha256(- for unknown)
ENTRIES=(
  "WiLoR checkpoint|${DEST}/wilor/wilor_final.ckpt|10000000|torch|1|-"
  "WiLoR model config|${DEST}/wilor/model_config.yaml|100|text|1|-"
  "WiLoR detector|${DEST}/wilor/detector.pt|1000000|torch|0|-"
  "HaWoR checkpoint|${DEST}/hawor/checkpoints/hawor.ckpt|10000000|torch|1|-"
  "HaWoR infiller|${DEST}/hawor/checkpoints/infiller.pt|1000000|torch|1|-"
  "HaWoR model config|${DEST}/hawor/checkpoints/model_config.yaml|100|text|0|-"
  "VGGT-Omega 416|${DEST}/vggt-omega/vggt_omega_1b_416_reproduce.pt|100000000|torch|1|-"
  "MANO right hand|${DEST}/mano/MANO_RIGHT.pkl|1000000|pickle|0|-"
)

human() {
    awk -v b="$1" 'BEGIN{
        if (b >= 1073741824) printf "%.2f GiB", b/1073741824;
        else if (b >= 1048576) printf "%.1f MiB", b/1048576;
        else printf "%.0f KiB", b/1024 }'
}

# snif fmt <file> -> torch | pickle | safetensors | text | unknown
sniff() {
    local file="$1" magic
    magic=$(head -c 4 "${file}" 2>/dev/null | od -An -tx1 | tr -d ' \n')
    case "${magic}" in
        504b0304*)                       # PK\x03\x04
            if head -c 8192 "${file}" | grep -qa '\.npy'; then echo "npz"; else echo "torch"; fi ;;
        80*)                             echo "pickle" ;;
        *)
            # safetensors: 8-byte little-endian header length, then a JSON dict
            local header
            header=$(head -c 8 "${file}" 2>/dev/null | od -An -tu8 | tr -d ' \n')
            if [[ -n "${header}" && "${header}" -gt 0 && "${header}" -lt 104857600 ]]; then
                if head -c $((8 + header)) "${file}" 2>/dev/null | tail -c "${header}" | head -c 1 | grep -qa '{'; then
                    echo "safetensors"; return 0
                fi
            fi
            if file --brief --mime "${file}" 2>/dev/null | grep -q '^text/'; then
                echo "text"; return 0
            fi
            echo "unknown" ;;
    esac
}

format_ok() {  # format_ok <expected> <actual>
    local expected="$1" actual="$2"
    case "${expected}" in
        torch) [[ "${actual}" == "torch" || "${actual}" == "pickle" ]] ;;
        pickle) [[ "${actual}" == "pickle" || "${actual}" == "torch" ]] ;;
        safetensors) [[ "${actual}" == "safetensors" ]] ;;
        text) [[ "${actual}" == "text" || "${actual}" == "unknown" ]] ;;
        *) true ;;
    esac
}

missing=0
invalid=0
optional_missing=0

printf '%-20s %-9s %-10s %s\n' "ASSET" "STATUS" "SIZE" "PATH"
printf '%-20s %-9s %-10s %s\n' "--------------------" "--------" "---------" "----"

for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r name path min_bytes expected required sha <<<"${entry}"
    rel="${path#"${ROOT}/"}"

    if [[ ! -f "${path}" ]]; then
        if [[ "${required}" == "1" ]]; then
            printf '%-20s %-9s %-10s %s\n' "${name}" "MISSING" "-" "${rel}"
            missing=$((missing + 1))
        else
            [[ "${QUIET}" == "0" ]] && printf '%-20s %-9s %-10s %s\n' "${name}" "optional" "-" "${rel}"
            optional_missing=$((optional_missing + 1))
        fi
        continue
    fi

    size=$(wc -c <"${path}")
    if [[ "${size}" -lt "${min_bytes}" ]]; then
        printf '%-20s %-9s %-10s %s\n' "${name}" "TOO SMALL" "$(human "${size}")" "${rel}"
        invalid=$((invalid + 1))
        continue
    fi

    actual=$(sniff "${path}")
    if ! format_ok "${expected}" "${actual}"; then
        printf '%-20s %-9s %-10s %s\n' "${name}" "BAD FORMAT" "$(human "${size}")" "${rel}"
        echo "                     looks like '${actual}', expected '${expected}' - the file is not a usable checkpoint"
        invalid=$((invalid + 1))
        continue
    fi

    if [[ "${sha}" != "-" ]]; then
        if command -v sha256sum >/dev/null 2>&1; then
            got=$(sha256sum "${path}" | cut -d' ' -f1)
        else
            got=$(shasum -a 256 "${path}" | cut -d' ' -f1)
        fi
        if [[ "${got}" != "${sha}" ]]; then
            printf '%-20s %-9s %-10s %s\n' "${name}" "BAD SHA" "$(human "${size}")" "${rel}"
            echo "                     expected ${sha}, got ${got}"
            invalid=$((invalid + 1))
            continue
        fi
    fi

    [[ "${QUIET}" == "0" ]] && printf '%-20s %-9s %-10s %s\n' "${name}" "ok" "$(human "${size}")" "${rel}"
done

echo
echo "destination: ${DEST}"
if [[ "${missing}" -eq 0 && "${invalid}" -eq 0 ]]; then
    echo "result     : all required weights present and readable"
else
    echo "result     : ${missing} missing, ${invalid} invalid"
    echo "next       : ./scripts/download_weights.sh            # fetch what is missing"
    echo "             ./scripts/download_weights.sh --dry-run   # see the URLs first"
fi
if [[ "${optional_missing}" -gt 0 ]]; then
    echo "note       : ${optional_missing} optional asset(s) absent (fine unless you need them)"
fi
if [[ -f "${DEST}/mano/MANO_RIGHT.pkl" ]]; then
    if [[ -f "${DEST}/mano/MANO_RIGHT.npz" ]]; then
        echo "note       : MANO converted to .npz - remember paths.mano_model in configs/hot3d.yaml"
    else
        echo "note       : MANO_RIGHT.pkl not converted yet - python scripts/convert_mano.py"
    fi
fi

if [[ "${STRICT}" == "1" && "${optional_missing}" -gt 0 ]]; then
    exit 1
fi
[[ "${missing}" -eq 0 && "${invalid}" -eq 0 ]]
