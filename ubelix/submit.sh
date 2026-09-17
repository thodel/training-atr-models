#!/usr/bin/env bash
# Check, validate, then sbatch. Run this instead of sbatch.
#
#   ./submit.sh <file.sbatch> [spec.json] [-- sbatch options ...]
#
#   ./submit.sh prepare.sbatch specs/german-xix-v2.json
#   JOB_ID=<id> ./submit.sh train.sbatch -- --partition=gpu --qos=job_gratis \
#       --cpus-per-task=12 --time=15:00:00
#
# 1. preflight.py refuses a submission that is known to fail or to run the wrong
#    code: a checkout behind origin/main or with uncommitted changes, or CPUs x
#    walltime above the QoS cap (job_gratis: 11,520 CPU-minutes, GPU jobs
#    included). #147.
#    Then the code is PINNED: ATR_CODE_COMMIT=HEAD travels with the job, and the
#    batch file runs a worktree of that commit (pin_code.sh) — so a job that
#    queues for two days, or is requeued, runs what was submitted.
# 2. A spec, if the job takes one, is validated here rather than inside the batch
#    job — a bad spec otherwise costs a queue wait and an allocation before
#    anything says so. Job 14431367 died 13 s in on a capital letter in model_id.
# 3. The validated spec is what the job receives: SPEC is exported and sbatch
#    passes the environment through. Without that, a file with SPEC=${SPEC:?}
#    died in 0 s (prep 15207246), and a file with a default SPEC silently ran the
#    default instead of the spec that was validated.
set -euo pipefail

SBATCH_FILE=${1:?usage: submit.sh <file.sbatch> [spec.json] [-- sbatch options ...]}
shift
SPEC=""
if [ $# -gt 0 ] && [ "$1" != "--" ]; then SPEC=$1; shift; fi
if [ $# -gt 0 ] && [ "$1" = "--" ]; then shift; fi
EXTRA=("$@")
UB=$HOME/ubelix
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}

bash -n "$SBATCH_FILE"
python3 "$REPO/ubelix/preflight.py" "$SBATCH_FILE" ${EXTRA[@]+"${EXTRA[@]}"}

# Pin the code (#147, part 2): the job runs this commit, however long it queues
# and however often it is requeued — see pin_code.sh. An explicit ATR_CODE_COMMIT
# is kept (re-running an old commit on purpose); ATR_UNPINNED=1 opts out.
if [ "${ATR_UNPINNED:-0}" = "1" ]; then
  unset ATR_CODE_COMMIT
  echo "code: UNPINNED — the job runs whatever $REPO holds when it starts"
else
  # Resolved here, to a full SHA, so a typo or a short or symbolic name fails now
  # and not on a compute node after the queue wait.
  WANT=${ATR_CODE_COMMIT:-HEAD}
  if ! ATR_CODE_COMMIT=$(git -C "$REPO" rev-parse --verify --quiet "$WANT^{commit}"); then
    echo "ATR_CODE_COMMIT=$WANT is not a commit in $REPO" >&2; exit 2
  fi
  export ATR_CODE_COMMIT
  if [ "$ATR_CODE_COMMIT" != "$(git -C "$REPO" rev-parse HEAD)" ]; then
    echo "!! code: pinned to $ATR_CODE_COMMIT, which is NOT this checkout's HEAD ($WANT)"
  else
    echo "code: pinned to $ATR_CODE_COMMIT"
  fi
fi

# An --export without ALL would drop ATR_CODE_COMMIT and SPEC from the job's
# environment, and with them the pin and the validated spec.
for opt in ${EXTRA[@]+"${EXTRA[@]}"}; do
  case "$opt" in
    --export=ALL|--export=ALL,*) ;;
    --export|--export=*)
      echo "refusing $opt: use --export=ALL,NAME=value so the job keeps ATR_CODE_COMMIT and SPEC" >&2
      exit 2 ;;
  esac
done

# Which spec: the one given, else the default the file names. A file whose SPEC is
# mandatory (${SPEC:?}) needs one; a file that names no SPEC takes none.
if [ -z "$SPEC" ]; then
  SPEC=$(sed -n 's/^SPEC=${SPEC:-\(.*\)}$/\1/p' "$SBATCH_FILE" | head -1)
  SPEC=${SPEC/\$HOME/$HOME}
fi
if [ -z "$SPEC" ] && grep -q '^SPEC=\${SPEC:?' "$SBATCH_FILE"; then
  echo "$SBATCH_FILE requires a spec: submit.sh $SBATCH_FILE <spec.json>" >&2; exit 2
fi

if [ -n "$SPEC" ]; then
  [ -f "$SPEC" ] || { echo "no such spec: $SPEC" >&2; exit 2; }
  echo "validating $SPEC"
  apptainer exec --bind /storage/research --bind /scratch --bind /rs_scratch \
    --env PYTHONPATH="$REPO/src:$REPO/engines" \
    "$UB/vlm-train.sif" python - "$SPEC" <<'PY'
import json, sys
from atr_training.contracts import TrainRequest
req = TrainRequest.model_validate(json.load(open(sys.argv[1])))
print(f"  OK  {req.model_id}  engine={req.engine}")
for d in req.datasets:
    print(f"      {d.hf_repo}  all_projects={d.all_projects} "
          f"max_pages={d.max_pages} partition={d.partition}")
PY
  export SPEC
fi

echo "submitting $SBATCH_FILE${SPEC:+ with SPEC=$SPEC}${EXTRA[*]:+ ${EXTRA[*]}}"
# No --export: sbatch's default passes the whole environment, SPEC and JOB_ID
# included, and an --export in EXTRA still sees them through ALL.
sbatch ${EXTRA[@]+"${EXTRA[@]}"} "$SBATCH_FILE"
