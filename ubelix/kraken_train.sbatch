#!/bin/bash
# STAGE 2 of 2 for kraken on UBELIX (#171) — `ketos train` on a free RTX 4090,
# for a job kraken_prepare.sbatch compiled:
#
#   JOB_ID=<id> ubelix/submit.sh ubelix/kraken_train.sbatch
#
# Three things make it survive the wall and a requeue, as in
# train_resumable.sbatch:
#
#   1. --requeue and the requeue trap, so a walltime expiry continues instead of
#      ending as TIMEOUT with the record in `training`;
#   2. the job id kept in a state file, so every attempt is the same job;
#   3. ATR_TRAIN_PREEMPTABLE=1: the runner reads SIGTERM as "resume later",
#      passes it on to ketos (which writes checkpoint_abort.ckpt), and the next
#      attempt continues with `ketos train --resume` from the newest checkpoint.
#
# Before anything trains, kraken_batch_fits.py holds the job's batch against
# this card: the default 256 needs ~43.5 GB, the 4090 has 23 (#171).
#
#SBATCH --job-name=kraken-train
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
# ketos' data loader workers (the KrakenTrainParams default is 8).
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
# 8 x 20 h = 9,600 of job_gratis's 11,520 CPU-minutes; the requeue trap carries
# a longer run across walls.
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -uo pipefail

S=/scratch/network/users/$USER
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}
source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
SIF=${SIF:-$HOME/ubelix/kraken-train.sif}
STATE=$S/runs/slurm-$SLURM_JOB_ID.jobid

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
export ATR_TRAIN_JOBS_ROOT=$S/runs/jobs
export ATR_TRAIN_CHECKPOINT_ROOT=$S/runs/checkpoints
export ATR_TRAIN_TRAINED_ROOT=$S/runs/trained
export ATR_TRAIN_ARTEFACT_CACHE_ROOT=$S/expA/artefacts
export PYTHONPATH=$REPO/src:$REPO/engines
export ATR_TRAIN_VENVS_ROOT=/opt
export ATR_TRAIN_IMAGE_PYTHON=/opt/kraken-train/bin/python
export ATR_TRAIN_GPU=0
export ATR_TRAIN_PREEMPTABLE=1
mkdir -p "$ATR_TRAIN_JOBS_ROOT" "$ATR_TRAIN_CHECKPOINT_ROOT" "$ATR_TRAIN_TRAINED_ROOT" "$S/runs"

run() {
  apptainer exec --nv \
    --bind /storage/research:/storage/research \
    --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
    --env HF_HOME="$HF_HOME" \
    --env ATR_TRAIN_JOBS_ROOT="$ATR_TRAIN_JOBS_ROOT" \
    --env ATR_TRAIN_CHECKPOINT_ROOT="$ATR_TRAIN_CHECKPOINT_ROOT" \
    --env ATR_TRAIN_TRAINED_ROOT="$ATR_TRAIN_TRAINED_ROOT" \
    --env ATR_TRAIN_ARTEFACT_CACHE_ROOT="$ATR_TRAIN_ARTEFACT_CACHE_ROOT" \
    --env PYTHONPATH="$PYTHONPATH" --env ATR_TRAIN_VENVS_ROOT=/opt \
    --env ATR_TRAIN_IMAGE_PYTHON="$ATR_TRAIN_IMAGE_PYTHON" \
    --env ATR_TRAIN_GPU=0 --env ATR_TRAIN_PREEMPTABLE=1 \
    "$SIF" "$@"
}

# The stable id: from the state file on a requeue, else JOB_ID (stage 1's).
if [ -f "$STATE" ]; then
  JOB_ID=$(cat "$STATE")
  echo "== attempt $((SLURM_RESTART_COUNT + 1)) — resuming training job $JOB_ID"
elif [ -n "${JOB_ID:-}" ]; then
  echo "$JOB_ID" > "$STATE"
  echo "== attempt 1 — training job $JOB_ID (compiled by kraken_prepare.sbatch)"
else
  echo "set JOB_ID to the id kraken_prepare.sbatch printed" >&2
  exit 2
fi
echo "== node $(hostname) $(date -Is)"
echo "== $(run /opt/kraken-train/bin/ketos --version 2>&1 | tail -1) in $SIF"

# Refused here, not as an OOM a few minutes into the first epoch. Exit 3 ends
# the job without a requeue; the record is not touched and stays resumable.
if ! run python "$REPO/ubelix/kraken_batch_fits.py" "$ATR_TRAIN_JOBS_ROOT/$JOB_ID/job.json"; then
  echo "== not started: the batch does not fit this card (see above)" >&2
  exit 3
fi

# Requeue on SIGTERM (walltime or preemption), once; see train_resumable.sbatch.
. "$REPO/ubelix/requeue_trap.sh"
REQUEUED=
trap '
  if [ -n "$REQUEUED" ]; then
    echo "== SIGTERM at $(date -Is) — again; $SLURM_JOB_ID is already requeueing"
  else
    REQUEUED=1
    echo "== SIGTERM at $(date -Is) — requeueing $SLURM_JOB_ID"
    kill -TERM "$PID" 2>/dev/null
    requeue_once
  fi
' TERM

run python -m kraken_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" --job-id "$JOB_ID" &
PID=$!

# `wait` returns on a trapped signal; the loop collects the runner's own exit.
wait "$PID"; CODE=$?
while [ "$CODE" -gt 128 ]; do
  wait "$PID"; CODE=$?
done

if [ "$CODE" = 75 ]; then
  echo "== preempted at $(date -Is); the record stays resumable."
  [ -n "$REQUEUED" ] || requeue_once
  exit 0
fi

echo "== runner exited $CODE"
python3 "$REPO/ubelix/report.py" "$ATR_TRAIN_JOBS_ROOT" "$JOB_ID" || true
exit "$CODE"
