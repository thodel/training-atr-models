#!/usr/bin/env bash
# Build the per-engine training virtualenvs on asteraix.
#
# WHY separate venvs: kraken 7.0.2 and a transformers new enough for Qwen3-VL
# cannot share a dependency tree, and the TrOCR trainer pins transformers
# differently again. The supervising service (atr-train) imports none of them —
# it spawns each job with the right interpreter (src/atr_training/backends.py) —
# so each gets its own.
#
# Usage:
#   bash scripts/make_venvs.sh                     # all three
#   bash scripts/make_venvs.sh vlm-train           # just one (or several)
#
# The SERVICE runs in kraken-train (deploy/systemd/atr-train.service), which is
# why that venv's requirements carry fastapi and uvicorn as well.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The trainer finds its venvs through ATR_TRAIN_VENVS_ROOT (TrainerSettings). This
# script reads the SAME key, so it cannot build one tree while the service looks
# in another. The serving repo's version read a bare VENVS_ROOT, which the
# service never consulted.
if [ -z "${ATR_TRAIN_VENVS_ROOT:-}" ] && [ -f "${ROOT}/.env" ]; then
  ATR_TRAIN_VENVS_ROOT="$(grep '^ATR_TRAIN_VENVS_ROOT=' "${ROOT}/.env" | cut -d= -f2- || true)"
fi
VENVS="${ATR_TRAIN_VENVS_ROOT:-${ROOT}/.venvs}"
VENVS="${VENVS/#\~/$HOME}"
# asteraix ships Python 3.12.3 (measured 16.09.2026).
PY="${PYTHON:-python3.12}"

# pip stages a package's EXISTING files into TMPDIR before overwriting them, so a
# TMPDIR on the research share breaks every *upgrade* while fresh installs keep
# working. That is how it presented on idhefix on 2026-08-07: 60-odd packages
# installed fine, then `pip install -U pip` died with EPERM uninstalling the
# bundled pip. CIFS refuses the ownership work the staging does — the same EPERM
# that forced copyfile over copy2 in the register stage.
case "$(stat -f -c %T "${TMPDIR:-/tmp}" 2>/dev/null || echo unknown)" in
  cifs|smb*|nfs*|9p|fuseblk)
    echo "NOTE: TMPDIR=${TMPDIR} is on a network filesystem, where pip cannot replace" >&2
    echo "      an installed package. Using a local one for this run instead." >&2
    TMPDIR="${LOCAL_TMPDIR:-${HOME}/atr-cache/tmp}"
    mkdir -p "${TMPDIR}"
    export TMPDIR
    ;;
esac

ALL=(kraken-train vlm-train trocr-train)
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("${ALL[@]}")
for t in "${TARGETS[@]}"; do
  case " ${ALL[*]} " in
    *" ${t} "*) ;;
    *) echo "unknown venv '${t}'. Known: ${ALL[*]}" >&2; exit 2 ;;
  esac
done

declare -A REQS=(
  [kraken-train]=engines/kraken_train_svc/requirements.txt
  [vlm-train]=engines/vlm_train_svc/requirements.txt
  [trocr-train]=engines/trocr_train_svc/requirements.txt
)

mkdir -p "${VENVS}"
for t in "${TARGETS[@]}"; do
  echo "== ${t} venv → ${VENVS}/${t} =="
  "${PY}" -m venv "${VENVS}/${t}"
  # Best-effort. pip replacing itself failed on idhefix (2026-08-07) and, under
  # set -e, aborted the build before a single real dependency was installed.
  if ! "${VENVS}/${t}/bin/pip" install -U pip wheel; then
    echo "  WARNING: could not upgrade pip/wheel in ${t}; continuing with the bundled pip" >&2
  fi
  # torch FIRST, from the cu128 index. Left to the requirements file, pip takes
  # the default index, which serves a wheel built against the newest CUDA — and a
  # GPU job then silently falls back to CPU rather than failing. 2.8.0+cu128 is
  # the build proven on these A40s.
  "${VENVS}/${t}/bin/pip" install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
  "${VENVS}/${t}/bin/pip" install -r "${ROOT}/${REQS[${t}]}"
done

echo "Done: ${TARGETS[*]}"
echo "Check: bash scripts/check_venvs.sh"
