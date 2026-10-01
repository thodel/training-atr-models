#!/bin/bash
# Pre-download the *small* files of a trust_remote_code model: config, remote code,
# tokenizer, processor — everything but the weights.
#
# prefetch_bases.sh cannot do this. Its allow_patterns are
# ['*.json','*.txt','*.jinja','*.model','*.safetensors'] — no '*.py' — so for a
# model that loads vendor code it fetches the config and leaves the code behind,
# and the failure surfaces only on an offline compute node.
set -u
export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
if [ -s "$HOME/.hf_token" ]; then
  HF_TOKEN=$(tr -d '[:space:]' < "$HOME/.hf_token"); export HF_TOKEN
fi
for rid in "$@"; do
  echo "== $(date -Is) $rid"
  apptainer exec --bind /storage/research --env HF_HOME="$HF_HOME" \
    --env HF_TOKEN="${HF_TOKEN:-}" "$HOME/ubelix/vlm-train-tf5.sif" \
    /opt/vlm-train/bin/python -c "
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], ignore_patterns=['*.safetensors','*.bin','*.pt','*.gguf','images/*'])
print('   ->', p)
import os
print('   files:', sorted(os.listdir(p))[:40])
" "$rid" || echo "   FAILED $rid"
done
echo "== done $(date -Is)"
