#!/bin/bash
# STAGE 1 of 2 — build the corpus on a CPU node. No GPU is allocated.
#
# prepare and compile are network, CPU and disk: streaming pages from the hub,
# parsing PageXML, cutting line crops, writing JSONL. On the German corpus that
# was 1 h 40 + 1 h 27. None of it touches a GPU, and doing it inside a GPU
# allocation wastes the scarce half of the machine — on UBELIX doubly so, where
# the GPU job can wait two days to start and would then spend its first three
# hours not using the GPU it waited for.
#
# This leaves the job in `training`, which is the same state a preemption leaves
# it in, so stage 2 takes the ordinary resume path. Run stage 2 with
# `train.sbatch` (or train_resumable.sbatch) and the SAME --job-id.
#
# Compute nodes do have internet (verified: huggingface.co -> 200 from bnode009),
# so the corpora do not need to be pre-cached on the share.
#
#   ubelix/submit.sh ubelix/prepare.sbatch $HOME/ubelix/specs/<spec>.json
#   (submit.sh refuses a stale checkout and an over-cap request first, #147)
#
#SBATCH --job-name=prep
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# More cores would buy little: the work is dominated by streaming and PIL.
#
# THE WALLTIME HAS TO FIT BESIDE A RUNNING JOB, not just inside the cap (#212).
# job_gratis's MaxCpuRunMinsPerUser is 11,520 CPU-minutes, counted as cpus x
# *remaining* walltime over all RUNNING jobs of the user, GPU jobs included.
# 8 x 20 h = 9,600 left 1,920 for everything else, so this prepare sat PENDING on
# `MaxCpuRunMinsPerUser` behind any other run of ours and was unblocked by hand
# three times. 8 x 12 h = 5,760 is half the cap, and it does not go lower than
# that because the longest build here took 11:15:04 (job 17795292).
#SBATCH --time=12:00:00
# A prepare that only ADOPTS a cached artefact is done in seconds — 7 s for each
# of the four olmOCR page preps on 09.10.2026 — and has no business booking the
# wall of a from-scratch build. Declaring ATR_TIME_CACHED is what lets submit.sh
# ask artefact_probe.py which of the two cases this is and request the short wall
# on a hit; it reads only that one, and anything it cannot answer keeps the
# --time above. Plain `sbatch` reads neither and gets the --time above too, which
# is why that one has to be the build value.
#
# ATR_TIME_BUILD is here so the pair can be read in one place, and it is pinned
# to the --time directive by tests/test_ubelix_prepare_walltime.py so it cannot
# quietly become a second, wrong answer.
#ATR_TIME_CACHED=00:30:00
#ATR_TIME_BUILD=12:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -euo pipefail

S=/scratch/network/users/$USER
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}
# Run the commit this job was submitted with (#147): REPO becomes a worktree of it.
# shellcheck source=pin_code.sh
source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
# Overridable so the same job can run under the transformers 5.x image:
#   sbatch --export=ALL,SIF=$HOME/ubelix/vlm-train-tf5.sif …
SIF=${SIF:-$HOME/ubelix/vlm-train.sif}
SPEC=${SPEC:?set SPEC=/path/to/spec.json}

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub

# The *arrow* cache goes to scratch, the downloads stay on the share.
#
# `HF_HOME` alone puts both under it, and on 05.10.2026 that cost a job: the
# research fileset stood at 11.83 T of a 12 T hard limit, so materializing
# `image-text_zh-regierungsratsprotokolle` (153'035 pages) died in
# `datasets/arrow_writer.py` with `OSError: [Errno 122] Disk quota exceeded`,
# which `datasets` re-raised as a bare `DatasetGenerationError` — a storage
# problem wearing a data problem's clothes (job 17328122).
#
# The split is deliberate. `$HF_HOME/hub` is 780 G of downloaded parquet that
# the whole `wbkolleg_dh_1` group benefits from and that is expensive to fetch
# again; it stays. `$HF_HOME/datasets` is 371 G of arrow that this stage
# rewrites anyway, and it is the part that grows without bound; it moves to
# scratch, which has 13.7 T free against a 15 T quota. Scratch is purge-eligible,
# which for a cache costs time and nothing else.
export HF_DATASETS_CACHE=$S/hf_datasets
mkdir -p "$HF_DATASETS_CACHE"

# HF token, if one is present. Read from a PRIVATE file in $HOME, never from
# $HF_HOME/token: HF_HOME points at the group-readable research share, so the
# default location would expose the token to everyone in wbkolleg_dh_1.
#
# Without it, prepare is rate-limited as an anonymous IP and a corpus with many
# project directories dies mid-stream:
#   429 … We had to rate limit your IP … make sure you pass a HF_TOKEN
# (job 14710443, koenigsfelden-charters-post-1500, 1,185 project globs.)
# `-r` (readable) is not enough, and that is not hypothetical: the token file
# existed, was mode 600, and held ZERO bytes, while every job logged "present".
# The trainer's "unauthenticated requests" warning was the true signal. Require a
# non-empty file whose content looks like a token, and say which check failed.
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
export ATR_TRAIN_VENVS_ROOT=/opt
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
    "$SIF" "$@"
}

JOB_ID=$(run python "$REPO/ubelix/submit_job.py" "$SPEC")
echo "$JOB_ID" > "$S/runs/last-prepared.jobid"
# Keyed by this Slurm job too, so a dependent job reads THIS prepare's id and
# not whichever prepare happened to write last-prepared.jobid most recently.
echo "$JOB_ID" > "$S/runs/prep-$SLURM_JOB_ID.jobid"
echo "== job $JOB_ID"
echo "== node $(hostname), $(nproc) cpus, $(date -Is)"

run python -m vlm_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" \
    --job-id "$JOB_ID" --stop-after compile

echo
echo "== corpus ready. Stage 2:"
echo "   JOB_ID=$JOB_ID ubelix/submit.sh ubelix/train.sbatch -- --partition=gpu --qos=job_gratis --cpus-per-task=12 --time=15:00:00"
