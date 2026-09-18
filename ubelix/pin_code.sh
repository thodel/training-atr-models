# shellcheck shell=bash
# Run a batch job on the commit it was submitted with (serving-atr-inference#147, part 2).
#
#   REPO=...; source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
#
# The container imports the training code from $REPO when the job STARTS. A job
# that waits two days in the queue, or is preempted and requeued, would otherwise
# run whatever the checkout holds by then.
#
# submit.sh exports ATR_CODE_COMMIT (a full SHA, HEAD at submission). pin_code
# points REPO at a git worktree of exactly that commit, creating it once and
# reusing it: a requeued job, a chained train/score job and every fan-out arm
# inherit the variable through sbatch's environment and land on the same tree.
# Being a real checkout, the worktree lets atr_training.codeversion record the
# commit in the job record.
#
# A worktree counts as ready only once <root>/<sha>.ready exists. `git worktree
# add` writes .git and HEAD before it checks the files out, so "the directory has
# a .git" is true of a half-made tree — one a job started a second later would
# run with most files missing, and one a SIGKILL mid-checkout would leave behind
# for every later start. Creating, replacing and marking happen under a lock;
# only the marker opens the fast path. Before running, the tree must be at the
# commit and have no changes to tracked files.
#
# Without ATR_CODE_COMMIT (plain sbatch, or ATR_UNPINNED=1 at submission) the job
# runs the checkout as it is, and says so — with a warning if it is behind main.
#
# Worktrees live under ${ATR_CODE_ROOT:-$HOME/.cache/training-atr-models/worktrees}
# ($HOME: backed up, not purged like scratch), one per commit AND NODE
# (<sha>.<node>), and are never removed automatically, because a requeued job may
# still need one:
#   git -C ~/training-atr-models worktree list
#   git -C ~/training-atr-models worktree remove <dir> && rm <dir>.ready

_PIN_CODE_SELF=${BASH_SOURCE[0]}

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
  if ! [[ $commit =~ ^[0-9a-f]{40}$ ]]; then
    echo "!! pin_code: ATR_CODE_COMMIT must be a full 40-character SHA, got '$commit' — submit with ubelix/submit.sh, which resolves it" >&2
    return 1
  fi

  # One tree per node. $HOME is shared across the cluster, and flock on it does
  # NOT serialise between nodes: three arms starting together (15560727/28/29 on
  # gnode25 and gnode26) each saw an unmarked tree, and two removed the one the
  # third was checking out — "Could not write new index file", both dead in a
  # second. A path per node makes the lock a node-local question, which is the
  # only kind flock can answer. A tree is ~1 MB of small files; the alternative,
  # a lock that works across GPFS, is the thing jobstore.claim needed O_EXCL for.
  local root=${ATR_CODE_ROOT:-$HOME/.cache/training-atr-models/worktrees}
  local node=${SLURMD_NODENAME:-$(hostname -s)}
  local dir=$root/$commit.$node
  mkdir -p "$root" || return 1
  if ! [ -f "$dir.ready" ] || ! [ -e "$dir/.git" ]; then
    _pin_code_locked "$root/.lock.$node" bash "$_PIN_CODE_SELF" --create "$commit" "$dir" "$REPO" || return 1
  fi

  local head changes
  head=$(git -C "$dir" rev-parse HEAD 2>/dev/null)
  if [ "$head" != "$commit" ]; then
    echo "!! pin_code: $dir is at ${head:-nothing}, not at $commit — refusing to run" >&2
    return 1
  fi
  if ! changes=$(git -C "$dir" status --porcelain --untracked-files=no 2>&1) || [ -n "$changes" ]; then
    echo "!! pin_code: $dir is not a clean checkout of $commit — refusing to run:" >&2
    echo "$changes" | head -20 >&2
    echo "   remove it (git -C $REPO worktree remove --force $dir; rm -f $dir.ready) and resubmit" >&2
    return 1
  fi
  REPO=$dir
  echo "== code: pinned to $(git -C "$REPO" log -1 --format='%h %cd %s' --date=short)"
}

_pin_code_locked() {  # lockfile command... — run command holding an exclusive lock
  local lock=$1; shift
  if [ "${ATR_PIN_LOCK:-}" != "python" ] && command -v flock >/dev/null 2>&1; then
    flock -w 600 "$lock" "$@"
  else
    # No flock (macOS): the same exclusive lock through fcntl.
    python3 -c '
import fcntl, subprocess, sys
with open(sys.argv[1], "a") as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    sys.exit(subprocess.call(sys.argv[2:]))
' "$lock" "$@"
  fi
}

_pin_code_create() {  # commit dir repo — runs under the lock
  local commit=$1 dir=$2 repo=$3
  if [ -f "$dir.ready" ] && [ -e "$dir/.git" ]; then
    return 0                     # another job finished it while we waited
  fi
  rm -f "$dir.ready"
  if [ -e "$dir" ]; then
    # Half made, or left by a killed job: `git worktree add` locks a tree while it
    # initialises, hence the double force.
    echo "== pin_code: replacing an unfinished worktree at $dir" >&2
    git -C "$repo" worktree remove --force --force "$dir" >&2 || rm -rf "$dir"
  fi
  if ! git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$repo" fetch -q origin >&2
  fi
  if ! git -C "$repo" worktree add --force --force --detach "$dir" "$commit" >&2; then
    echo "!! pin_code: cannot create a worktree of $commit from $repo — is the commit in that checkout?" >&2
    return 1
  fi
  touch "$dir.ready"
}

if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--create" ]; then
  _pin_code_create "$2" "$3" "$4"
  exit $?
fi
