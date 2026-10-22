# The requeue a TERM trap does, and what it says — sourced by train.sbatch and
# train_resumable.sbatch (#150, #232). Needs SLURM_JOB_ID; uses scontrol and squeue.
#
# Two things send the trap's TERM and they need opposite things from it:
#
#   * the WALL (--signal=B:TERM@120): Slurm ends a TIMEOUT, it does not requeue
#     one, so `scontrol requeue` here is what keeps a six-day run going in 24 h
#     chunks. Load-bearing: do not remove it as redundant.
#   * a PREEMPTION: Slurm requeues the job itself (--requeue) and signals the
#     step while doing so, and `scontrol requeue` of a job that is already being
#     requeued is NOT accepted. For fifteen preemptions of 16298912 the trap then
#     printed "(requeue refused; job will end)" — and the job ran on every time.
#     Read in the morning, that line said a week-long run had died (#150).
#
# So a requeue that is not accepted is not a verdict. The trap tells the two
# apart by the one fact it has: how much walltime is left. The wall's signal
# arrives 120 s before the end; anything else is not the wall.

#: Seconds the wall signal arrives before the limit (--signal=B:TERM@120), with
#: room for the signal to be delivered late.
REQUEUE_WALL_WINDOW_S=${REQUEUE_WALL_WINDOW_S:-240}

# "1-02:03:04", "02:03:04", "03:04" or "4" -> seconds; "" for anything else.
slurm_seconds() {
  local text=$1 days=0 rest h m s
  case "$text" in
    ''|UNLIMITED|NOT_SET|INVALID|N/A) echo ""; return ;;
    *-*) days=${text%%-*}; rest=${text#*-} ;;
    *) rest=$text ;;
  esac
  case "$rest" in
    *:*:*) h=${rest%%:*}; rest=${rest#*:}; m=${rest%%:*}; s=${rest#*:} ;;
    *:*)   h=0; m=${rest%%:*}; s=${rest#*:} ;;
    *)     h=0; m=0; s=$rest ;;
  esac
  case "$days$h$m$s" in *[!0-9]*) echo ""; return ;; esac
  echo $(( (10#$days * 24 + 10#$h) * 3600 + 10#$m * 60 + 10#$s ))
}

# Requeue once and say what happened — never "refused" or "will end" when the
# job is in fact going on. Prints one line; returns 0 always (a trap must not
# end the shell on the requeue's exit status).
requeue_once() {
  local job=${SLURM_JOB_ID:?} out left seconds
  if out=$(scontrol requeue "$job" 2>&1); then
    echo "   requeued $job by this trap"
    return 0
  fi
  left=$(squeue -h -j "$job" -o %L 2>/dev/null | head -1)
  seconds=$(slurm_seconds "$left")
  if [ -z "$seconds" ]; then
    echo "   scontrol requeue not accepted (${out:-no message}), and squeue does not say how much wall is left — whether $job continues is not known from here; sacct -j $job -D says"
  elif [ "$seconds" -le "$REQUEUE_WALL_WINDOW_S" ]; then
    echo "   requeue REFUSED at the wall (${left} left: ${out:-no message}) — $job will end as TIMEOUT; resubmit it"
  else
    echo "   not the wall (${left} left), so a preemption: Slurm requeues $job itself (--requeue), which is why this trap's requeue was not accepted (${out:-no message}); the job goes on"
  fi
  return 0
}
