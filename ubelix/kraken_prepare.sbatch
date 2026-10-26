#!/bin/bash
# STAGE 1 of 2 for kraken on UBELIX (#171) — prepare and `ketos compile` on a
# CPU node, in the kraken image. No GPU is allocated.
#
# The same split as prepare.sbatch for the VLM: streaming pages, cutting lines
# and compiling arrows are CPU, network and disk, and a GPU job that waited two
# days should not spend its first hours on them. This leaves the job in
# `training`; stage 2 is kraken_train.sbatch with the SAME job id, which finds
# the corpus in the job's own directory (`_resume_artifacts`).
#
# The image is kraken-train.sif, built from ubelix/kraken-train.def — ketos
# 7.0.2 exactly, as on asteraix. vlm-train.sif has no ketos at all.
#
#   ubelix/submit.sh ubelix/kraken_prepare.sbatch $HOME/ubelix/specs/<kraken-spec>.json
#
# A spec for the free RTX 4090 needs params.batch_size <= 128: the default 256
# needs ~43.5 GB and the card has 23 (kraken_batch_fits.py, checked again by
# stage 2 on the GPU node before anything trains).
#
#SBATCH --job-name=kraken-prep
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --ntasks=1
# ketos compile's --workers, the KrakenTrainParams default.
#SBATCH --cpus-per-task=8
# Half of job_gratis's 11,520 CPU-minutes, as for prepare.sbatch (#212).
#SBATCH --time=12:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -euo pipefail

S=/scratch/network/users/$USER
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}
source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
SIF=${SIF:-$HOME/ubelix/kraken-train.sif}
SPEC=${SPEC:?set SPEC=/path/to/spec.json}

# Downloads on the share, the datasets' arrow cache on scratch, and the HF
# token from a private file — all three for the reasons prepare.sbatch gives.
export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
export HF_DATASETS_CACHE=$S/hf_datasets
mkdir -p "$HF_DATASETS_CACHE"
HF_TOKEN=""
if [ -s "$HOME/.hf_token" ]; then
  HF_TOKEN=$(tr -d '[:space:]' < "$HOME/.hf_token")
fi
case "$HF_TOKEN" in
  hf_?*) export HF_TOKEN; echo "== HF token: present (${#HF_TOKEN} chars)" ;;
  "")    unset HF_TOKEN; echo "== HF token: ABSENT or EMPTY (~/.hf_token) — anonymous rate limits apply" ;;
  *)     unset HF_TOKEN; echo "== HF token: ~/.hf_token does not look like a token (expected hf_…) — ignoring it" ;;
esac
export ATR_TRAIN_JOBS_ROOT=$S/runs/jobs
export ATR_TRAIN_CHECKPOINT_ROOT=$S/runs/checkpoints
export ATR_TRAIN_TRAINED_ROOT=$S/runs/trained
export ATR_TRAIN_ARTEFACT_CACHE_ROOT=$S/expA/artefacts
export PYTHONPATH=$REPO/src:$REPO/engines
# ketos is <venvs root>/kraken-train/bin/ketos, and the image puts it at /opt.
export ATR_TRAIN_VENVS_ROOT=/opt
export ATR_TRAIN_IMAGE_PYTHON=/opt/kraken-train/bin/python
mkdir -p "$ATR_TRAIN_JOBS_ROOT" "$ATR_TRAIN_CHECKPOINT_ROOT" "$ATR_TRAIN_TRAINED_ROOT" "$S/runs"

run() {
  apptainer exec \
    --bind /storage/research:/storage/research \
    --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
    --env HF_HOME="$HF_HOME" --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env ATR_TRAIN_JOBS_ROOT="$ATR_TRAIN_JOBS_ROOT" \
    --env ATR_TRAIN_CHECKPOINT_ROOT="$ATR_TRAIN_CHECKPOINT_ROOT" \
    --env ATR_TRAIN_TRAINED_ROOT="$ATR_TRAIN_TRAINED_ROOT" \
    --env ATR_TRAIN_ARTEFACT_CACHE_ROOT="$ATR_TRAIN_ARTEFACT_CACHE_ROOT" \
    --env PYTHONPATH="$PYTHONPATH" --env ATR_TRAIN_VENVS_ROOT=/opt \
    --env ATR_TRAIN_IMAGE_PYTHON="$ATR_TRAIN_IMAGE_PYTHON" \
    "$SIF" "$@"
}

echo "== $(run /opt/kraken-train/bin/ketos --version 2>&1 | tail -1) in $SIF"
JOB_ID=$(run python "$REPO/ubelix/submit_job.py" "$SPEC")
echo "$JOB_ID" > "$S/runs/prep-$SLURM_JOB_ID.jobid"
echo "== job $JOB_ID"
echo "== node $(hostname), $(nproc) cpus, $(date -Is)"

run python -m kraken_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" \
    --job-id "$JOB_ID" --stop-after compile

echo
echo "== corpus ready. Stage 2:"
echo "   JOB_ID=$JOB_ID ubelix/submit.sh ubelix/kraken_train.sbatch"
