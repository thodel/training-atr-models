#!/usr/bin/env bash
# Post-provisioning smoke test for the three training venvs.
#
#   bash scripts/check_venvs.sh          # every venv that exists
#   bash scripts/check_venvs.sh -v       # also list satisfied requirements
#
# Exit 0 only when every present venv passes. TWO checks per venv:
#
#   1. an IMPORT smoke test — catches a broken or incomplete dependency tree;
#   2. a VERSION check against the venv's own requirements.txt.
#
# Imports alone are not enough. The transformers 5.x incident passed every import
# there was — `TrainingArguments(...)` constructed fine on 5.14.1 against code
# written for 4.57 — and so did the failed repair, which died with EPERM and left
# 5.14.1 in place. Only the version comparison catches either.
#
# The serving repo's version of this script had no trocr-train entry at all,
# although make_venvs.sh built it and backends.py declared it. The gate after
# provisioning did not check one of the venvs it had just provisioned.
# tests/test_venv_scripts.py now pins that the two lists agree.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
VERBOSE=""
{ [ "${1:-}" = "-v" ] || [ "${1:-}" = "--verbose" ]; } && VERBOSE="-v"

# The SAME key the service reads (TrainerSettings: ATR_TRAIN_VENVS_ROOT), so this
# check and the service can never be looking at two different trees.
if [ -z "${ATR_TRAIN_VENVS_ROOT:-}" ] && [ -f "${ROOT}/.env" ]; then
  ATR_TRAIN_VENVS_ROOT="$(grep '^ATR_TRAIN_VENVS_ROOT=' "${ROOT}/.env" | cut -d= -f2- || true)"
fi
VENVS_ROOT="${ATR_TRAIN_VENVS_ROOT:-${ROOT}/.venvs}"
VENVS_ROOT="${VENVS_ROOT/#\~/$HOME}"

if [ ! -d "${VENVS_ROOT}" ]; then
  echo "ERROR: venv root not found: ${VENVS_ROOT}" >&2
  exit 1
fi

#   name | requirements file | import smoke test
declare -a VENV_ENTRIES=(
  # The service runs here too, hence fastapi/uvicorn in the smoke test.
  "kraken-train|engines/kraken_train_svc/requirements.txt|import kraken, datasets, fastapi, uvicorn; from importlib.metadata import version; print('kraken-train', version('kraken'))"
  # qwen3_vl must be a model transformers knows — precisely what a version below
  # 4.57 fails at, and it costs no download to ask.
  "vlm-train|engines/vlm_train_svc/requirements.txt|import peft, bitsandbytes, datasets; from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments; from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES; assert 'qwen3_vl' in CONFIG_MAPPING_NAMES, 'this transformers does not know qwen3_vl'; print('vlm-train ok')"
  "trocr-train|engines/trocr_train_svc/requirements.txt|import datasets, accelerate; from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer; print('trocr-train ok')"
)

passed=0; failed=0; skipped=0
for entry in "${VENV_ENTRIES[@]}"; do
  IFS='|' read -r name reqs smoke <<< "${entry}"
  python="${VENVS_ROOT}/${name}/bin/python"
  if [ ! -x "${python}" ]; then
    echo "SKIP  ${name} — venv not present"
    skipped=$((skipped + 1)); continue
  fi
  ok=true; detail=""
  if ! out=$("${python}" -c "${smoke}" 2>&1); then
    ok=false; detail="${detail}
      imports: ${out}"
  fi
  if ! out=$("${python}" "${SCRIPT_DIR}/check_requirements.py" ${VERBOSE} "${ROOT}/${reqs}" 2>&1); then
    ok=false; detail="${detail}
      versions (${reqs}):
$(echo "${out}" | sed 's/^/        /')"
  fi
  if $ok; then
    echo "PASS  ${name}"; passed=$((passed + 1))
  else
    echo "FAIL  ${name}${detail}"; failed=$((failed + 1))
  fi
done

echo "────────────────────────────────────────"
echo "Results: ${passed} passed, ${failed} failed, ${skipped} not present"
if [ ${failed} -gt 0 ]; then
  echo "A version MISMATCH usually means a pip install failed without you noticing:" >&2
  echo "pip exits non-zero but leaves the version it was replacing in place. Re-run" >&2
  echo "with TMPDIR on LOCAL disk and check again." >&2
  exit 1
fi
exit 0
