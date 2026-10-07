#!/bin/bash
# Score ONE finished adapter on the Federal Council held-out test set
# (Hodel & Schoch 2021, doi:10.5281/zenodo.4746342), all 2,751 lines.
#   sbatch --export=ALL,JOB_ID=<id>,TAG=<tag>[,SIF=...][,QUANT=4bit] score_federal_minutes.sbatch
#
# QUANT picks how the base is loaded, and it has to match how the arm was
# TRAINED: an adapter fitted over a 4-bit base belongs over a 4-bit base when it
# is scored. The default stays bf16, which is what the seven reports already on
# this eval set were produced with — every one of them trained in bf16 too. A
# fanned-out arm with `load_in_4bit=true` needs QUANT=4bit, or the number lands
# in the same table at a different quantisation and nothing says so.
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
SIF=${SIF:-$HOME/ubelix/vlm-train.sif}
: "${JOB_ID:?}" "${TAG:?}"
# PROMPT=training_render asks the model to continue from the training render
# instead of from add_generation_prompt=True. Only one family needs it so far:
# gemma-4-12B-it's generation prompt opens an empty thinking channel that its
# training render never contains, and that arm scored 22.94 % with 17,027
# insertions where its E4B sibling, which has no such block, scored 10.24 %.
# For Qwen the two renders agree, so the flag is a no-op there.
case "${PROMPT:-generation}" in
  training_render) PROMPT_FLAG=--prompt-from-training-render ;;
  generation)      PROMPT_FLAG= ;;
  *) echo "PROMPT must be generation or training_render, got '${PROMPT}'" >&2; exit 2 ;;
esac
case "${QUANT:-bf16}" in
  4bit) QUANT_FLAG=--load-in-4bit ;;
  bf16) QUANT_FLAG=--no-load-in-4bit ;;
  *)    echo "QUANT must be 4bit or bf16, got '${QUANT}'" >&2; exit 2 ;;
esac
E=$S/runs/jobs/evalset-federal-minutes
BASE=$(python3 -c "import json;print(json.load(open('$S/runs/jobs/$JOB_ID/job.json'))['request']['base_model'])")
if [ -s "$HOME/.hf_token" ]; then HF_TOKEN=$(tr -d '[:space:]' < "$HOME/.hf_token"); export HF_TOKEN; fi
echo "== $TAG: $JOB_ID ($BASE) ${QUANT:-bf16} prompt=${PROMPT:-generation} on $(hostname) $(date -Is)"
apptainer exec --nv \
  --bind /storage/research:/storage/research --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
  --env HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub \
  --env HF_TOKEN="${HF_TOKEN:-}" --env PYTHONPATH="$REPO/src:$REPO/engines" \
  "$SIF" python -m vlm_train_svc.evaluate_qlora \
    --adapter "$S/runs/checkpoints/$JOB_ID" \
    --val-jsonl "$E/data/val.jsonl" --report "$E/report_$TAG.json" \
    --base-model "$BASE" --data-root "$E" \
    --prompt "Transcribe the handwritten text in this image exactly as written." \
    --granularity line --max-pixels 262144 --max-seq-len 1024 --seed 42 \
    --device cuda:0 --max-samples 2751 --max-new-tokens 256 "$QUANT_FLAG" \
    ${PROMPT_FLAG:+"$PROMPT_FLAG"}
echo "== done $(date -Is)"
