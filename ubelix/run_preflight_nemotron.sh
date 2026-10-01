#!/bin/bash
# Stage A of the Nemotron feasibility package. No GPU, no weights: it loads config,
# vendor code, tokenizer and image processor only, so it runs where there is an
# internet route. Pass OFFLINE=1 for the second pass.
set -u
REPO=$HOME/training-atr-models
export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
if [ -s "$HOME/.hf_token" ]; then HF_TOKEN=$(tr -d "[:space:]" < "$HOME/.hf_token"); export HF_TOKEN; fi
apptainer exec --bind /storage --bind /scratch --bind /rs_scratch \
  --env PYTHONPATH="$REPO/src:$REPO/engines" --env HF_HOME="$HF_HOME" \
  --env HF_TOKEN="${HF_TOKEN:-}" --env HF_HUB_OFFLINE="${OFFLINE:-0}" \
  "$HOME/ubelix/vlm-train-tf5.sif" /opt/vlm-train/bin/python \
  "$HOME/ubelix/preflight_nemotron.py" "${1:-nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1}"
