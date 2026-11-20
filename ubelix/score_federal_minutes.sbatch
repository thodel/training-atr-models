#!/bin/bash
# Score ONE finished adapter on the Federal Council held-out test set
# (Hodel & Schoch 2021, doi:10.5281/zenodo.4746342), all 2,751 lines.
#   ubelix/submit.sh ubelix/score_federal_minutes.sbatch -- --export=ALL,JOB_ID=<id>,TAG=<tag>[,SIF=...][,QUANT=4bit]
#
# The image follows the base model, as it does for the trainer
# (ubelix/sif_for_base.py): a Qwen3.5 or Gemma 4 base runs in vlm-train-tf5.sif,
# everything else in vlm-train.sif. SIF= still overrides. With a fixed
# vlm-train.sif default, the Qwen3.5 line arms of #427 died after 40 s on
# "model type `qwen3_5` ... does not recognize this architecture" (22861171,
# 22861173, 20.11.2026).
#   ubelix/submit.sh ubelix/score_federal_minutes.sbatch -- \
#       --export=ALL,ADAPTER=dh-unibe/<published adapter>,BASE_MODEL=<base>,TAG=<tag>
#                     # a run whose scratch checkpoint and job.json are purged (#875)
#
# QUANT picks how the base is loaded, and it has to match how the arm was
# TRAINED: an adapter fitted over a 4-bit base belongs over a 4-bit base when it
# is scored. The default stays bf16, which is what the seven reports already on
# this eval set were produced with — every one of them trained in bf16 too. A
# fanned-out arm with `load_in_4bit=true` needs QUANT=4bit, or the number lands
# in the same table at a different quantisation and nothing says so.
#
# Where the set and the reports are (#875). The lines are read from
# EVALSET, by default eval_sets/federal-minutes on the research share, which
# ubelix/materialize_benchmark.sbatch builds; the copy under
# $SCRATCH/runs/jobs/evalset-federal-minutes was lost to the 30-day purge, and
# is still taken when it exists and the share copy does not. The report goes to
# REPORT_DIR, by default that scratch directory, because the results MCP's
# `report` tool reads it there; copy report_<TAG>.json and .raw.jsonl to
# $EVALSET/readings/ before the purge takes them too (docs/EVAL_SETS.md).
#SBATCH --job-name=fedmin1
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out
set -euo pipefail
S=/scratch/network/users/$USER
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}
# Run the commit this job was submitted with (#147): REPO becomes a worktree of it.
# shellcheck source=pin_code.sh
source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
: "${TAG:?}"
if [ -n "${ADAPTER:-}" ]; then
  : "${BASE_MODEL:?with ADAPTER, set BASE_MODEL: there is no job.json to read it from}"
else
  : "${JOB_ID:?set JOB_ID, or ADAPTER and BASE_MODEL}"
  ADAPTER=$S/runs/checkpoints/$JOB_ID
fi
# PROMPT defaults to training_render since 08.10.2026, because it was measured:
# the same 12B adapter on the same 2,751 lines scored 22.94 % asked to continue
# from add_generation_prompt=True and 9.95 % asked to continue from the training
# render, while the E4B control — whose two renders agree — came back
# byte-identical at 10.24 %. PROMPT=generation restores the old behaviour for
# reproducing one of the seven reports made before that date.
case "${PROMPT:-training_render}" in
  training_render) PROMPT_FLAG=--prompt-from-training-render ;;
  generation)      PROMPT_FLAG=--prompt-from-generation-prompt ;;
  *) echo "PROMPT must be generation or training_render, got '${PROMPT}'" >&2; exit 2 ;;
esac
case "${QUANT:-bf16}" in
  4bit) QUANT_FLAG=--load-in-4bit ;;
  bf16) QUANT_FLAG=--no-load-in-4bit ;;
  *)    echo "QUANT must be 4bit or bf16, got '${QUANT}'" >&2; exit 2 ;;
esac
OLD_E=$S/runs/jobs/evalset-federal-minutes
E=${EVALSET:-/storage/research/wbkolleg_dh_1/Textrecognition_Training/eval_sets/federal-minutes}
if [ ! -s "$E/data/val.jsonl" ] && [ -z "${EVALSET:-}" ] && [ -s "$OLD_E/data/val.jsonl" ]; then
  echo "!! no set under $E: reading the scratch copy $OLD_E"; E=$OLD_E
fi
[ -s "$E/data/val.jsonl" ] || { echo "no $E/data/val.jsonl — run ubelix/materialize_benchmark.sbatch first" >&2; exit 2; }
R=${REPORT_DIR:-$OLD_E}
mkdir -p "$R"
BASE=${BASE_MODEL:-$(python3 -c "import json;print(json.load(open('$S/runs/jobs/$JOB_ID/job.json'))['request']['base_model'])")}
SIF=${SIF:-$(python3 "$REPO/ubelix/sif_for_base.py" "$BASE" "$HOME")} || exit 3
[ -f "$SIF" ] || { echo "no image at $SIF" >&2; exit 3; }
if [ -s "$HOME/.hf_token" ]; then HF_TOKEN=$(tr -d '[:space:]' < "$HOME/.hf_token"); export HF_TOKEN; fi
echo "== $TAG: ${JOB_ID:-$ADAPTER} ($BASE) ${QUANT:-bf16} prompt=${PROMPT:-training_render} on $(hostname) $(date -Is)"
echo "== image: $SIF"
echo "== lines: $E/data/val.jsonl  report: $R/report_$TAG.json"
apptainer exec --nv \
  --bind /storage/research:/storage/research --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
  --env HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub \
  --env HF_TOKEN="${HF_TOKEN:-}" --env PYTHONPATH="$REPO/src:$REPO/engines" \
  "$SIF" /opt/vlm-train/bin/python -m vlm_train_svc.evaluate_qlora \
    --adapter "$ADAPTER" \
    --val-jsonl "$E/data/val.jsonl" --report "$R/report_$TAG.json" \
    --base-model "$BASE" --data-root "$E" \
    --prompt "Transcribe the handwritten text in this image exactly as written." \
    --granularity line --max-pixels 262144 --max-seq-len 1024 --seed 42 \
    --device cuda:0 --max-samples 2751 --max-new-tokens 256 "$QUANT_FLAG" \
    ${PROMPT_FLAG:+"$PROMPT_FLAG"}
echo "== done $(date -Is)"
