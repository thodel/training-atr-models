# shellcheck shell=bash
# Run a batch job on the commit it was submitted with (serving-atr-inference#147, part 2).
#
#   REPO=...; source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
#
# The container imports the training code from $REPO when the job STARTS. A job
# that waits two days in the queue, or is preempted and requeued, would otherwise
# run whatever the checkout holds by then — twice that was a day's worth of main
# that nobody had meant to run.
#
# submit.sh exports ATR_CODE_COMMIT (HEAD at submission). pin_code then points
# REPO at a git worktree of exactly that commit, creating it once and reusing it:
# a requeued job, a chained train/score job and every fan-out arm inherit the
# variable through sbatch's environment and land on the same tree. Being a real
# checkout, the worktree lets atr_training.codeversion record the commit in the
# job record, and the stage records stop reporting drift.
#
# Without ATR_CODE_COMMIT (plain sbatch, or ATR_UNPINNED=1 at submission) the job
# runs the checkout as it is, and says so — with a warning if it is behind main.
#
# Worktrees live under ${ATR_CODE_ROOT:-$HOME/.cache/training-atr-models/worktrees}
# ($HOME: backed up, not purged like scratch). They are small and never removed
# automatically, because a requeued job may still need one; list and prune with
#   git -C ~/training-atr-models worktree list
#   git -C ~/training-atr-models worktree remove <dir>

pin_code() {
  local commit=${ATR_CODE_COMMIT:-}
  if [ -z "$commit" ]; then
    echo "== code: UNPINNED — running the checkout as it is now: $(git -C "$REPO" log -1 --format='%h %cd %s' --date=short 2>/dev/null)"
    if git -C "$REPO" fetch -q origin main 2>/dev/null; then
      local behind
      behind=$(git -C "$REPO" rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
      if [ "${behind:-0}" -gt 0 ]; then
        echo "!! WARNING: checkout is $behind commit(s) behind origin/main — submit with ubelix/submit.sh to pin the code"
      fi
    fi
    return 0
  fi

  local root=${ATR_CODE_ROOT:-$HOME/.cache/training-atr-models/worktrees}
  local dir=$root/$commit
  mkdir -p "$root" || return 1
  if [ ! -e "$dir/.git" ]; then
    if command -v flock >/dev/null 2>&1; then
      # Fan-out arms start together; one creates the worktree, the rest wait.
      ( flock -w 600 9 || { echo "!! pin_code: no lock on $root after 600 s" >&2; exit 1; }
        _pin_code_add "$commit" "$dir" ) 9>"$root/.lock" || return 1
    else
      echo "== pin_code: flock not available, creating the worktree without a lock" >&2
      _pin_code_add "$commit" "$dir" || return 1
    fi
  fi
  local head
  head=$(git -C "$dir" rev-parse HEAD 2>/dev/null)
  if [ "$head" != "$commit" ]; then
    echo "!! pin_code: $dir is at ${head:-nothing}, not at $commit — refusing to run" >&2
    return 1
  fi
  REPO=$dir
  echo "== code: pinned to $(git -C "$REPO" log -1 --format='%h %cd %s' --date=short)"
}

_pin_code_add() {  # commit dir — no-op when another job created it meanwhile
  [ -e "$2/.git" ] && return 0
  git -C "$REPO" worktree prune 2>/dev/null
  if ! git -C "$REPO" cat-file -e "$1^{commit}" 2>/dev/null; then
    git -C "$REPO" fetch -q origin 2>/dev/null
  fi
  if ! git -C "$REPO" worktree add --detach "$2" "$1" >/dev/null 2>&1; then
    echo "!! pin_code: cannot create a worktree of $1 from $REPO — is the commit in that checkout?" >&2
    return 1
  fi
}
