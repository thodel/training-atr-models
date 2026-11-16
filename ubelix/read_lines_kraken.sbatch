#!/bin/bash
# One kraken model's reading of every line of an evaluation set, line by line
# (agentic_historian#875): <READINGS>/<LABEL>.json and <LABEL>.raw.jsonl, keyed by
# the eval set's image paths, for measure_scorer_tau.py's cross-engine pairs.
# `ketos test` gives the CER and no line; scripts/read_lines_kraken.py says why
# this is a separate reader.
#
#   MODEL=<trained>/<id>/<id>.mlmodel LABEL=kraken-german-xix-v1 \
#     ubelix/submit.sh ubelix/read_lines_kraken.sbatch
#   EVALSET=<eval_sets>/<set> ... — another set than the Federal Council lines
#   LINE_TYPE=baselines — read each crop as a baseline line instead of a box
#
# CPU: a kraken line model reads the 2,751 lines in minutes on a few cores. The
# image is kraken-train.sif (kraken 7.0.2), the one the models were trained with.
#SBATCH --job-name=read-lines-kraken
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out
set -euo pipefail
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}
# shellcheck source=pin_code.sh
source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
: "${MODEL:?set MODEL to a kraken recognition model (.mlmodel or .safetensors)}"
: "${LABEL:?set LABEL, the model name the readings are filed under}"
EVAL_SETS=${EVAL_SETS:-/storage/research/wbkolleg_dh_1/Textrecognition_Training/eval_sets}
E=${EVALSET:-$EVAL_SETS/federal-minutes}
READINGS=${READINGS:-$E/readings}
[ -s "$E/data/val.jsonl" ] || { echo "no $E/data/val.jsonl — run ubelix/materialize_benchmark.sbatch first" >&2; exit 2; }
SIF=${SIF:-$HOME/ubelix/kraken-train.sif}
echo "== $LABEL: $MODEL on $E on $(hostname) $(date -Is)"
apptainer exec --bind /storage --bind /scratch \
  --env PYTHONPATH="$REPO/src" \
  "$SIF" /opt/kraken-train/bin/python "$REPO/scripts/read_lines_kraken.py" \
  --model "$MODEL" --val-jsonl "$E/data/val.jsonl" --data-root "$E" \
  --report "$READINGS/$LABEL.json" --label "$LABEL" --device cpu \
  --line-type "${LINE_TYPE:-bbox}"
echo "== done $(date -Is)"
