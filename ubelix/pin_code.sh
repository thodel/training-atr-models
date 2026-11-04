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
# for every later start. Creating, replacing and marking happen under two locks:
# a flock per node (jobs on one node), and inside it a directory in the main
# checkout's .git (pin_code.lock.d), taken with mkdir because every node's
# `git worktree add` writes into the same .git/worktrees/ (#360). Only the
# marker opens the fast path, which takes neither. Before running, the tree must be at the
# commit and have no changes to tracked files.
#
# Without ATR_CODE_COMMIT (plain sbatch, or ATR_UNPINNED=1 at submission) the job
# runs the checkout as it is, and says so — with a warning if it is behind main.
#
# Worktrees live under ${ATR_CODE_ROOT:-$HOME/.cache/training-atr-models/worktrees}
# ($HOME: backed up, not purged like scratch), one per commit AND NODE
# (<sha>.<node>), and are never removed automatically, because a requeued job may
# still need one. The marker goes first: a job that starts between the two
# steps must not find a tree that is being removed (#41):
#   git -C ~/training-atr-models worktree list
#   rm -f <dir>.ready && git -C ~/training-atr-models worktree remove <dir>

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

  local head changes errors
  head=$(git -C "$dir" rev-parse HEAD 2>/dev/null)
  if [ "$head" != "$commit" ]; then
    echo "!! pin_code: $dir is at ${head:-nothing}, not at $commit — refusing to run" >&2
    if [ -z "$head" ] && [ -e "$dir/.git" ]; then
      # The tree points at an admin directory the main checkout no longer has
      # — it was re-cloned, or `worktree prune` ran (#41). Say what fixes it.
      echo "   its .git points at an admin directory $REPO no longer has (re-cloned?):" >&2
      echo "   git -C $REPO worktree prune; rm -rf $dir $dir.ready; then resubmit" >&2
    fi
    return 1
  fi
  # stdout only: a warning git prints on stderr is not a change (#41). An
  # untracked *.py counts — a stray module.py shadows the one in the tree —
  # while a marker or a log a requeued job left does not; and a file marked
  # assume-unchanged counts, because `status` would not show it.
  errors=$(mktemp)
  if ! changes=$(git -C "$dir" status --porcelain --untracked-files=all 2>"$errors"); then
    echo "!! pin_code: git status failed in $dir — refusing to run:" >&2
    head -20 "$errors" >&2; rm -f "$errors"
    return 1
  fi
  rm -f "$errors"
  changes=$(echo "$changes" | grep -v '^?? ' ; echo "$changes" | grep '^?? .*\.py$' || true)
  changes=$(echo "$changes" | sed '/^$/d')
  local hidden
  hidden=$(git -C "$dir" ls-files -v 2>/dev/null | grep '^[a-z]' || true)
  if [ -n "$changes" ] || [ -n "$hidden" ]; then
    echo "!! pin_code: $dir is not a clean checkout of $commit — refusing to run:" >&2
    { echo "$changes"; [ -n "$hidden" ] && echo "$hidden" | sed 's/^/assume-unchanged: /'; } | sed '/^$/d' | head -20 >&2
    echo "   remove it (rm -f $dir.ready; git -C $REPO worktree remove --force $dir) and resubmit" >&2
    return 1
  fi
  REPO=$dir
  echo "== code: pinned to $(git -C "$REPO" log -1 --format='%h %cd %s' --date=short)"
}

#: How long a job waits for another job's checkout of the same commit. A wait
#: that runs out says so (#41): flock's silent exit 1 looked like the command
#: failing, and the fcntl fallback waited for ever.
ATR_PIN_LOCK_WAIT_S=${ATR_PIN_LOCK_WAIT_S:-600}

_pin_code_locked() {  # lockfile command... — run command holding an exclusive lock
  local lock=$1 rc; shift
  if [ "${ATR_PIN_LOCK:-}" != "python" ] && command -v flock >/dev/null 2>&1; then
    flock -w "$ATR_PIN_LOCK_WAIT_S" -E 75 "$lock" "$@"; rc=$?
  else
    # No flock (macOS): the same exclusive lock through fcntl, with the same wait.
    python3 -c '
import fcntl, subprocess, sys, time
deadline = time.monotonic() + float(sys.argv[2])
with open(sys.argv[1], "a") as fh:
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.monotonic() >= deadline:
                sys.exit(75)
            time.sleep(0.2)
    sys.exit(subprocess.call(sys.argv[3:]))
' "$lock" "$ATR_PIN_LOCK_WAIT_S" "$@"; rc=$?
  fi
  if [ "$rc" = 75 ]; then
    echo "!! pin_code: waited $ATR_PIN_LOCK_WAIT_S s for $lock — another job is still checking out this commit, or a dead one left the lock; refusing to run" >&2
  fi
  return "$rc"
}

#: A lock on the main checkout older than this was left by a dead job (#360).
#: `worktree add` of a ~1 MB tree takes seconds; a fetch, under a minute.
ATR_PIN_REPO_LOCK_STALE_S=${ATR_PIN_REPO_LOCK_STALE_S:-300}

_pin_code_repo_lockdir() {  # repo — where the cross-node lock on its worktrees lives
  local common
  common=$(git -C "$1" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) \
    || common=$1/.git
  echo "$common/pin_code.lock.d"
}

_pin_code_repo_lock() {  # repo — hold the lock on its worktree admin, across nodes (#360)
  # The per-node flock keeps two jobs on one node apart; it cannot keep nodes
  # apart, and every node's `git worktree add` writes into the SAME
  # $REPO/.git/worktrees/ and reads the same refs. mkdir is atomic on GPFS as
  # anywhere, which flock on a shared $HOME is not — the lock jobstore.claim
  # takes with O_EXCL for the same reason.
  local lockdir age now mtime
  lockdir=$(_pin_code_repo_lockdir "$1")
  local deadline=$((SECONDS + ATR_PIN_LOCK_WAIT_S))
  until mkdir "$lockdir" 2>/dev/null; do
    now=$(date +%s)
    mtime=$(stat -c %Y "$lockdir" 2>/dev/null || stat -f %m "$lockdir" 2>/dev/null || echo "$now")
    age=$((now - mtime))
    if [ "$age" -gt "$ATR_PIN_REPO_LOCK_STALE_S" ]; then
      # Renamed before it is removed: of two jobs that both find it stale, only
      # one rename succeeds, so neither removes a lock the other just took.
      if mv "$lockdir" "$lockdir.stale.$$" 2>/dev/null; then
        echo "== pin_code: removing a lock on $1 held for ${age} s: $(cat "$lockdir.stale.$$/owner" 2>/dev/null)" >&2
        rm -rf "$lockdir.stale.$$"
      fi
      continue
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "!! pin_code: waited $ATR_PIN_LOCK_WAIT_S s for $lockdir — another node is creating a worktree of $1 ($(cat "$lockdir/owner" 2>/dev/null)); refusing to run" >&2
      return 75
    fi
    sleep "0.$((RANDOM % 9 + 1))"
  done
  echo "$(hostname -s 2>/dev/null) pid $$ since $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$lockdir/owner"
  _PIN_CODE_HELD_LOCK=$lockdir
}

_pin_code_repo_unlock() {  # — release the lock this process holds, if any
  # A global, not the caller's local: the EXIT trap runs after the function that
  # set it has returned, when its locals are gone.
  [ -n "${_PIN_CODE_HELD_LOCK:-}" ] && rm -rf "$_PIN_CODE_HELD_LOCK"
  _PIN_CODE_HELD_LOCK=
}

_pin_code_create() {  # commit dir repo — runs under the node's lock
  local commit=$1 dir=$2 repo=$3
  if [ -f "$dir.ready" ] && [ -e "$dir/.git" ]; then
    return 0                     # another job finished it while we waited
  fi
  _pin_code_repo_lock "$repo" || return 1
  # This runs as its own process (--create), so EXIT is the end of the creation.
  trap _pin_code_repo_unlock EXIT
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
    if git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null; then
      echo "!! pin_code: git worktree add of $commit into $dir failed (git's message above) — the commit is in $repo" >&2
    else
      echo "!! pin_code: cannot create a worktree of $commit from $repo — the commit is not in that checkout, and fetching origin did not bring it" >&2
    fi
    return 1
  fi
  touch "$dir.ready"
}

if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--create" ]; then
  _pin_code_create "$2" "$3" "$4"
  exit $?
fi
