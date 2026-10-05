#!/bin/bash
# Build one line-level dataset from a page-shaped one and push it private.
#
# CPU only: materialize, crop, pack parquet, upload. No GPU is involved, so this
# goes to epyc2 like the prepare stage rather than competing for a card.
#
#   REPO_ID=dh-unibe/image-text_x REV=<40-hex> OUT=dh-unibe/image-text_x-lines \
#     sbatch ubelix/make_line_dataset.sbatch
#
# MAX_PAGES bounds what is read; DatasetSpec refuses all_projects without one.
# The default is above every one of these datasets, so it means "everything" and
# the job reports what that turned out to be.
#
# Run from the checkout: ~/ubelix holds images, logs and specs, and a helper
# copied there goes stale (tests/test_ubelix_layout.py).
#SBATCH --job-name=line-dataset
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out
set -euo pipefail
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}
# shellcheck source=pin_code.sh
source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
if [ -s "$HOME/.hf_token" ]; then HF_TOKEN=$(tr -d '[:space:]' < "$HOME/.hf_token"); export HF_TOKEN; fi
: "${REPO_ID:?set REPO_ID to the page-shaped source dataset}"
: "${REV:?set REV to the source revision, a full 40-hex SHA}"
: "${OUT:?set OUT to the dataset repo to create}"
apptainer exec --bind /storage --bind /scratch --bind /rs_scratch \
  --env PYTHONPATH="$REPO/src:$REPO/engines" --env HF_HOME="$HF_HOME" \
  --env HF_TOKEN="${HF_TOKEN:-}" --env USER="$USER" \
  "$HOME/ubelix/vlm-train-tf5.sif" /opt/vlm-train/bin/python \
  "$REPO/ubelix/make_line_dataset.py" --repo "$REPO_ID" --revision "$REV" \
  --out "$OUT" --max-pages "${MAX_PAGES:-200000}"
