# The working directory of a line-dataset job, and its removal — sourced by
# make_line_dataset.sbatch (#183). Needs SLURM_JOB_ID.
#
# make_line_dataset.py names its directory job-$SLURM_JOB_ID under the base and
# removes it when it ends, TERM included. This file is the second line: the
# batch shell survives an OOM kill of the python process (the killer takes the
# process, not the step), and its EXIT trap then removes what python could not.
# Both sides derive the same path from the same two facts, so neither has to be
# told the other's.

# The directory a line-dataset job works in: <base>/job-<SLURM_JOB_ID>.
line_dataset_workdir() {
  local base=${1:?base directory}
  echo "$base/job-${SLURM_JOB_ID:?SLURM_JOB_ID}"
}

# Remove the directory and say so. Prints one line; returns 0 always, because a
# trap must not turn a finished job into a failed one over a leftover.
cleanup_line_workdir() {
  local dir=${1:?directory}
  case "$dir" in
    */job-[0-9]*) ;;
    *) echo "   not removing $dir: not a job-<id> working directory"; return 0 ;;
  esac
  if [ ! -e "$dir" ]; then
    echo "   $dir already gone (make_line_dataset.py removed it)"
    return 0
  fi
  rm -rf "$dir" 2>/dev/null || true
  if [ -e "$dir" ]; then
    echo "   $dir could NOT be removed entirely — ubelix/clean_line_workdirs.py lists it against sacct"
  else
    echo "   removed $dir (this trap)"
  fi
  return 0
}
