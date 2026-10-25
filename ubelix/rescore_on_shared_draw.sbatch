#!/bin/bash
# Re-score one finished run's adapter on another run's evaluation draw.
#
# Generation only — no training, no record change, so the gratis QoS rather than a
# preemptable H100.
#
# An hour is enough for a line arm (200 crops, a few minutes) and **not** for a
# page arm: 164 pages of roughly 1 200 reference characters each ran past 40
# minutes with the generation loop still going. Pass TIME for those.
#
#   JOB=<id> DRAW_FROM=<id> ubelix/submit.sh ubelix/rescore_on_shared_draw.sbatch
#   JOB=<id> DRAW_FROM=<id> ubelix/submit.sh ubelix/rescore_on_shared_draw.sbatch -- --time=04:00:00
#   JOB=<id> DRAW_FILE=<lines.jsonl> DATA_ROOT=<corpus root> LABEL=<name> \
#       ubelix/submit.sh ubelix/rescore_on_shared_draw.sbatch     # a draw as a file (#74)
#   JOB=<id> DRAW_FILE=<eval_sets>/german-medieval-v1/vlm-pages-test/pages.jsonl \
#       DATA_ROOT=<eval_sets>/german-medieval-v1/vlm-pages-test LABEL=german-medieval-v1-test \
#       HELDOUT=1 ubelix/submit.sh ubelix/rescore_on_shared_draw.sbatch -- --time=<…>
#                                       # a held-out page set (#301): refuses on overlap.
#       695 pages, four times the 164 above — size TIME from that run's log.
#
# Run from the checkout: ~/ubelix holds the container images, the logs and the
# specs, and a helper copied there goes stale (tests/test_ubelix_layout.py).
#SBATCH --job-name=rescore-shared-draw
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out
set -euo pipefail
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}
# Run the commit this job was submitted with (#147): REPO becomes a worktree of it.
# shellcheck source=pin_code.sh
source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
export ATR_TRAIN_JOBS_ROOT=${ATR_TRAIN_JOBS_ROOT:-/scratch/network/users/$USER/runs/jobs}
export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
if [ -s "$HOME/.hf_token" ]; then HF_TOKEN=$(tr -d '[:space:]' < "$HOME/.hf_token"); export HF_TOKEN; fi
apptainer exec --nv --bind /storage --bind /scratch --bind /rs_scratch \
  --env PYTHONPATH="$REPO/src:$REPO/engines" --env HF_HOME="$HF_HOME" \
  --env HF_TOKEN="${HF_TOKEN:-}" --env ATR_TRAIN_JOBS_ROOT="$ATR_TRAIN_JOBS_ROOT" \
  "$HOME/ubelix/vlm-train-tf5.sif" /opt/vlm-train/bin/python \
  "$REPO/ubelix/rescore_on_shared_draw.py" --job "$JOB" \
  ${DRAW_FILE:+--draw-file "$DRAW_FILE" --data-root "$DATA_ROOT" --label "$LABEL"} \
  ${HELDOUT:+--heldout} \
  ${DRAW_FROM:+--draw-from "$DRAW_FROM"}
