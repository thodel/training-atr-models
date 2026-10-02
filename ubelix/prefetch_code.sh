#!/bin/bash
# Pre-download the *small* files of a trust_remote_code model: config, remote code,
# tokenizer, processor — everything but the weights.
#
# prefetch_bases.sh cannot do this. Its allow_patterns are
# ['*.json','*.txt','*.jinja','*.model','*.safetensors'] — no '*.py' — so for a
# model that loads vendor code it fetches the config and leaves the code behind,
# and the failure surfaces only on an offline compute node.
#
# It fails loudly. A gated repo, a typo in a repo id, a missing container or an
# apptainer that is not on PATH used to produce one scrolled-past line and exit 0,
# which is indistinguishable from a complete prefetch to whatever runs next.
set -u
export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
SIF=$HOME/ubelix/vlm-train-tf5.sif
if [ "$#" -eq 0 ]; then
  echo "usage: $(basename "$0") <repo-id> [<repo-id> ...]" >&2
  exit 2
fi
command -v apptainer >/dev/null || { echo "apptainer is not on PATH" >&2; exit 2; }
[ -f "$SIF" ] || { echo "no container at $SIF" >&2; exit 2; }
if [ -s "$HOME/.hf_token" ]; then
  HF_TOKEN=$(tr -d '[:space:]' < "$HOME/.hf_token"); export HF_TOKEN
fi
failed=0
for rid in "$@"; do
  echo "== $(date -Is) $rid"
  apptainer exec --bind /storage/research --env HF_HOME="$HF_HOME" \
    --env HF_TOKEN="${HF_TOKEN:-}" "$SIF" \
    /opt/vlm-train/bin/python -c "
import os, sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1],
                      ignore_patterns=['*.safetensors','*.bin','*.pt','*.pth','*.ckpt',
                                       '*.gguf','*.h5','*.msgpack','*.onnx','images/*'])
files = sorted(os.listdir(p))
print('   ->', p)
print('   files:', files[:40])
# The whole point of this script: a trust_remote_code repo without its .py files is
# a cache entry that looks complete and fails offline.
if not any(f.endswith('.py') for f in files):
    print('   NOTE: no .py in this snapshot — fine for a plain model, a problem if it '
          'declares an auto_map', file=sys.stderr)
" "$rid" || { echo "   FAILED $rid" >&2; failed=$((failed + 1)); }
done
if [ "$failed" -gt 0 ]; then
  echo "== $failed of $# repo(s) FAILED $(date -Is)" >&2
  exit 1
fi
echo "== done, $# repo(s) $(date -Is)"
