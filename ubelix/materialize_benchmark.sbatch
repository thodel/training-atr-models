#!/bin/bash
# Build a published benchmark project as an evaluation set on the research share
# (agentic_historian#875): data/val.jsonl, the line crops it names, the pages,
# evalset.json. scripts/materialize_benchmark.py says how and why.
#
# The default is the Federal Council lines (Hodel & Schoch 2021,
# doi:10.5281/zenodo.4746342), 2,751 lines. Their earlier copy lived under
# $SCRATCH/runs/jobs/evalset-federal-minutes and went with the 30-day purge;
# this one goes to eval_sets/, which is not purged.
#
#   ubelix/submit.sh ubelix/materialize_benchmark.sbatch
#   HF_REPO=<repo> PROJECT=<project> OUT=<dir> [SPLIT=test] [REV=<sha>] [EXPECT_LINES=<n>] \
#     ubelix/submit.sh ubelix/materialize_benchmark.sbatch
#
# CPU only (download ~220 MB of parquet, write ~2,800 pages and crops): epyc2,
# like make_line_dataset.sbatch. It refuses an OUT that already has a val.jsonl —
# a set reports were scored on is moved aside by hand, never overwritten.
#
# Run from the checkout: ~/ubelix holds images, logs and specs, and a helper
# copied there goes stale (tests/test_ubelix_layout.py).
#SBATCH --job-name=materialize-benchmark
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out
set -euo pipefail
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}
# shellcheck source=pin_code.sh
source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
EVAL_SETS=${EVAL_SETS:-/storage/research/wbkolleg_dh_1/Textrecognition_Training/eval_sets}
HF_REPO=${HF_REPO:-dh-unibe/image-text_federal-minutes-testset}
PROJECT=${PROJECT:-TEST_federal_minutes}
OUT=${OUT:-$EVAL_SETS/federal-minutes}
if [ "$PROJECT" = TEST_federal_minutes ]; then EXPECT_LINES=${EXPECT_LINES:-2751}; fi
export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
# Arrow to scratch, downloads on the share — see make_line_dataset.sbatch.
export HF_DATASETS_CACHE=/scratch/network/users/$USER/hf_datasets
mkdir -p "$HF_DATASETS_CACHE"
if [ -s "$HOME/.hf_token" ]; then HF_TOKEN=$(tr -d '[:space:]' < "$HOME/.hf_token"); export HF_TOKEN; fi
echo "== $HF_REPO / $PROJECT -> $OUT on $(hostname) $(date -Is)"
apptainer exec --bind /storage --bind /scratch \
  --env PYTHONPATH="$REPO/src:$REPO/engines" --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" --env HF_TOKEN="${HF_TOKEN:-}" \
  "$HOME/ubelix/vlm-train-tf5.sif" /opt/vlm-train/bin/python \
  "$REPO/scripts/materialize_benchmark.py" --hf-repo "$HF_REPO" --project "$PROJECT" \
  --split "${SPLIT:-train}" --out "$OUT" ${REV:+--revision "$REV"} \
  ${EXPECT_LINES:+--expect-lines "$EXPECT_LINES"}
echo "== done $(date -Is)"
