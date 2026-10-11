#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# check_env.sh — validate the interactive environment against what the services
# expect (#54)
#
# Run this before make_venvs.sh, before any manual training/eval command, and
# whenever something works in a systemd unit but not in your shell.
#
# What it checks:
#   1. effective TMPDIR / HF_HOME and their filesystem types  (network = FAIL)
#   2. whether ~/.cache/huggingface/hub is on the research share, and at the
#      level UBELIX caches at (…/hf_hub/hub, not …/hf_hub; #207)
#   3. free space on / and the share
#   4. what atr-train.service actually sets, vs. what the shell carries
#   5. ~/.bashrc TMPDIR override that would shadow the service's TMPDIR
#   6. the trainer's keys in .env: ATR_TRAIN_API_KEY (what idhefix's gateway
#      sends), ATR_TRAIN_ALLOWED_CLIENTS, ATR_TRAIN_GATEWAY_API_KEY and
#      ATR_TRAIN_GATEWAY_URL (what the promotion gate uses) — and the launcher's
#      own verdict (python -m atr_training.serve --check) when the venv exists
#
# Exit 0 = clean.  Exit 1 = problem detected (explainers printed).
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail

FAILURES=0
WARNINGS=0

print_header() {
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "══════════════════════════════════════════════════════════════"
}

pass()  { echo "  ✓  $1"; }
fail()  { echo "  ✗  $1" >&2; FAILURES=$((FAILURES + 1)); }
warn()  { echo "  ⚠  $1"; WARNINGS=$((WARNINGS + 1)); }
info()  { echo "      $1"; }

fingerprint() {  # 8 hex chars — enough to compare two values without revealing either
  # sha256sum is coreutils (the box); shasum is what macOS ships. An empty
  # fingerprint would quietly defeat the point of comparing without printing.
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$1" | sha256sum | cut -c1-8
  elif command -v shasum >/dev/null 2>&1; then
    printf '%s' "$1" | shasum -a 256 | cut -c1-8
  else
    echo "no-sha256"
  fi
}

# ── 1. Variables and their backing filesystems ───────────────────────────────

print_header "1 — Environment variables and backing filesystems"

# stat -f -c %T : filesystem type (-f) on Linux; works for both local and network fs
fs_type() {
  stat -f -c %T "$1" 2>/dev/null || echo "unknown"
}

# The share on idhefix reports **smb2**, not "smb", so every check that compared
# against the literal string passed it as local — including the TMPDIR one, which
# exists because a network TMPDIR broke compile (0ca2379). Globs, not equality,
# and one function so a new mount type is fixed in one place.
is_network_fs() {
  case "$1" in
    nfs*|cifs*|smb*|fuse.sshfs|fuseblk|9p|afs|glusterfs|ceph) return 0 ;;
    *) return 1 ;;
  esac
}

# TMPDIR
TMPDIR_VAL="${TMPDIR:-$(mktemp -u)}"
TMPDIR_FS=$(fs_type "$TMPDIR_VAL")
echo "  TMPDIR      = ${TMPDIR_VAL}"
echo "  TMPDIR fs   = ${TMPDIR_FS}"

if is_network_fs "$TMPDIR_FS"; then
  fail "TMPDIR is on a network filesystem (${TMPDIR_FS}) — the trainer refuses this"
  info "  ketos compile dies with 'Directory not empty' on network TMPDIR, and pip"
  info "  upgrades silently fail with EPERM.  Fix:"
  info "    mkdir -p ~/atr-cache/tmp"
  info "    echo 'TMPDIR=~/atr-cache/tmp' >> ~/.bashrc"
  info "    # then open a new shell"
elif [[ "$TMPDIR_FS" == "unknown" ]]; then
  warn "TMPDIR ${TMPDIR_VAL} does not exist yet — mktemp will use it as prefix"
  info "  it will be created on first use; make sure it is on a local fs"
else
  pass "TMPDIR is on local filesystem (${TMPDIR_FS})"
fi

# datasets Arrow generation cache (#60)
# Distinct from the hub cache, and the distinction is what an 11.5-hour failure
# turned on: hub/ stores downloaded files and is fine on the share, while
# datasets/ is written by pyarrow as a long streaming write, which SMB cannot
# hold a handle open for.
if [ -n "${HF_DATASETS_CACHE:-}" ]; then
  DS_CACHE="$HF_DATASETS_CACHE"
elif [ -n "${HF_HOME:-}" ]; then
  DS_CACHE="${HF_HOME}/datasets"
else
  DS_CACHE="${HOME}/.cache/huggingface/datasets"
fi
DS_RESOLVED=$(readlink -f "$DS_CACHE" 2>/dev/null || echo "$DS_CACHE")
DS_FS=$(fs_type "$DS_RESOLVED")
echo "  datasets cache = ${DS_CACHE}"
[ "$DS_RESOLVED" != "$DS_CACHE" ] && echo "  resolves to    = ${DS_RESOLVED}"
echo "  datasets fs    = ${DS_FS}"

if is_network_fs "$DS_FS"; then
  fail "the datasets Arrow cache is on a network filesystem (${DS_FS})"
  info "  pyarrow cannot hold a write handle open there for a generation pass:"
  info "  'ValueError: I/O operation on closed file' after 11.5 h, zero pages."
  info "  The trainer refuses a CACHED job in this state.  Fix:"
  info "    [ -L ~/.cache/huggingface/datasets ] && rm ~/.cache/huggingface/datasets"
  info "    mkdir -p ~/.cache/huggingface/datasets"
  info "  or stream instead:  ATR_TRAIN_CACHE_DATASETS=false"
else
  pass "datasets Arrow cache is on local filesystem (${DS_FS})"
fi

# HF_HOME
HF_HOME_VAL="${HF_HOME:-}"
HF_HOME_RESOLVED=""
if [ -n "$HF_HOME_VAL" ]; then
  HF_HOME_RESOLVED="$HF_HOME_VAL"
  HF_HOME_FS=$(fs_type "$HF_HOME_RESOLVED")
  echo "  HF_HOME     = ${HF_HOME_VAL}"
  echo "  HF_HOME fs  = ${HF_HOME_FS}"
  if is_network_fs "$HF_HOME_FS"; then
    warn "HF_HOME is on a network filesystem (${HF_HOME_FS}) — downloads will be slow"
  fi
else
  echo "  HF_HOME     = (not set — good; will use ~/.cache/huggingface)"
fi

# HF hub cache — the symlink that matters
HF_HUB_LINK="$HOME/.cache/huggingface/hub"
HF_HUB_TARGET=$(readlink -f "$HF_HUB_LINK" 2>/dev/null || echo "")
echo "  ~/.cache/huggingface/hub → ${HF_HUB_TARGET:-(broken or missing)}"
HF_HUB_FS=$(fs_type "$HF_HUB_LINK" 2>/dev/null)
if [ ! -e "$HF_HUB_LINK" ]; then
  warn "hub cache symlink does not exist yet"
elif is_network_fs "$HF_HUB_FS"; then
  warn "hub cache resolves to a network filesystem (${HF_HUB_FS}) — downloads go there"
  info "  the service deliberately avoids HF_HOME to prevent this; if you set it"
  info "  in ~/.bashrc, your interactive commands will re-download what the service"
  info "  already cached (and fill the share with 16 GB per base model)"
else
  pass "hub cache is on ${HF_HUB_FS}"
fi

# The level inside the share (#207). UBELIX sets HF_HOME to …/hf_hub and
# huggingface_hub caches in HF_HOME/hub, so the one directory both sides share
# is …/hf_hub/hub. A link that ends in …/hf_hub keeps a second cache in the
# old layout one level up, which UBELIX never reads and which the storage pass
# deletes as "old form" — on 09.10.2026 it took asteraix's whole cache with it.
case "$HF_HUB_TARGET" in
  */hf_hub)
    fail "hub cache symlink points at …/hf_hub, one level above the cache UBELIX uses"
    info "  nothing of UBELIX's downloads is found here, and entries written here are"
    info "  the 'old form' the storage pass removes.  Fix (when no job is running):"
    info "    ln -sfn ${HF_HUB_TARGET}/hub ~/.cache/huggingface/hub"
    ;;
  */hf_hub/hub)
    pass "hub cache symlink ends in hf_hub/hub — the level UBELIX caches at"
    ;;
  "")
    ;;
  *)
    warn "hub cache resolves to ${HF_HUB_TARGET}, not under the share's hf_hub/hub"
    info "  the shared cache both machines use is …/Textrecognition_Training/hf_hub/hub"
    ;;
esac

# ── 2. Free disk space ───────────────────────────────────────────────────────

print_header "2 — Free disk space"

check_space() {
  local path="$1"
  local label="${2:-${path}}"
  if df -k "$path" >/dev/null 2>&1; then
    local avail_k
    avail_k=$(df -k --output=avail "$path" 2>/dev/null | tail -1)
    local avail_g=$((avail_k / 1024 / 1024))
    echo "  ${label}: ${avail_g} GB available"
    if [ "$avail_k" -lt 524288 ]; then   # < 512 MB
      fail "${label} has < 512 MB free — installs and compiles will fail"
    elif [ "$avail_k" -lt 5242880 ]; then # < 5 GB
      warn "${label} has < 5 GB free — ensure TMPDIR is local (see §1)"
    else
      pass "${label} has ${avail_g} GB free"
    fi
  else
    warn "cannot check ${label} (path not accessible)"
  fi
}

check_space "/"
check_space "/tmp"
check_space "${TMPDIR_VAL}"

SHARE_MOUNT=""
for cand in "/mnt/wbkolleg_dh_1" "/mnt/research" "/mnt/share"; do
  if [ -d "$cand" ]; then
    SHARE_MOUNT="$cand"
    break
  fi
done
if [ -n "$SHARE_MOUNT" ]; then
  check_space "$SHARE_MOUNT"
fi

# ── 3. Systemd unit environment vs. shell ────────────────────────────────────

print_header "3 — atr-train.service Environment vs. current shell"

UNIT_MAINPID=$(systemctl --user show atr-train -p MainPID --value --no-pager 2>/dev/null || echo "")
if [ -n "$UNIT_MAINPID" ] && [ "$UNIT_MAINPID" != "0" ] && [ -r "/proc/${UNIT_MAINPID}/environ" ]; then
  # The running process's own environment: Environment=, EnvironmentFile= and
  # anything set-environment put there, which is the only view that matches what
  # the service actually sees.
  UNIT_VARS=$(tr '\0' '\n' < "/proc/${UNIT_MAINPID}/environ" | grep -E '^(TMPDIR|HF_HOME|CUDA_VISIBLE_DEVICES|PYTHONPATH|ATR_)' | tr '\n' ' ')
else
  UNIT_VARS=$(systemctl --user show atr-train -p Environment --no-pager 2>/dev/null || echo "")
fi
if [ -z "$UNIT_VARS" ]; then
  warn "cannot read atr-train.service environment (systemctl failed — probably not on idhefix)"
  info "  run this script on the box where the service is installed"
else
  # Secrets are shown as a fingerprint, never a value. Reading the process
  # environment means ATR_*_KEY is in scope here, and this script's output gets
  # pasted into issues — the same rule as section 5.
  echo "  Unit sets:"
  echo "$UNIT_VARS" | tr ' ' '\n' | while IFS= read -r pair; do
    [ -n "$pair" ] || continue
    case "${pair%%=*}" in
      *KEY|*TOKEN|*SECRET|*PASSWORD)
        echo "    ${pair%%=*}=<set, fingerprint $(fingerprint "${pair#*=}")>" ;;
      *) echo "    ${pair}" ;;
    esac
  done

  # Check TMPDIR
  UNIT_TMPDIR=$(echo "$UNIT_VARS" | tr ' ' '\n' | grep '^TMPDIR=' | cut -d= -f2-)
  if [ -n "$UNIT_TMPDIR" ]; then
    echo ""
    info "  Unit TMPDIR  = ${UNIT_TMPDIR}"
    info "  Shell TMPDIR = ${TMPDIR_VAL}"
    if [ "$UNIT_TMPDIR" != "$TMPDIR_VAL" ]; then
      if is_network_fs "$TMPDIR_FS"; then
        fail "Shell TMPDIR differs from unit AND is on a network fs — unit is right, fix your shell"
      else
        warn "Shell TMPDIR differs from unit — if you run commands by hand while"
        warn "  the service is running, they see different temp space"
      fi
    else
      pass "Shell TMPDIR matches unit"
    fi
  fi

  # Check HF_HOME — unit must NOT have it
  if echo "$UNIT_VARS" | tr ' ' '\n' | grep -q '^HF_HOME='; then
    fail "Unit sets HF_HOME — it must NOT, to avoid filling the share with downloads"
  else
    pass "Unit does NOT set HF_HOME (correct)"
  fi
fi

# ── 4. Shell profile drift ───────────────────────────────────────────────────

print_header "4 — Shell profile drift (TMPDIR override in ~/.bashrc)"

BASHRC_TMPDIR=$(grep -v '^#' "$HOME/.bashrc" 2>/dev/null | grep 'TMPDIR=' | grep -v 'export TMPDIR=' | head -1)
if [ -n "$BASHRC_TMPDIR" ]; then
  warn "~/.bashrc sets TMPDIR without 'export': ${BASHRC_TMPDIR}"
  info "  A systemd unit loads .env, not .bashrc — if your TMPDIR is in .bashrc"
  info "  but the unit's TMPDIR is in .env, they differ and the unit is right."
  info "  Move the setting to .env so both the unit and interactive shell agree:"
  info "    echo 'TMPDIR=~/atr-cache/tmp' >> ~/.env"
  info "    sed -i '/TMPDIR=/d' ~/.bashrc"
  info "    source ~/.bashrc"
else
  pass "~/.bashrc does not override TMPDIR"
fi

# Also check for HF_HOME in bashrc
BASHRC_HFHOME=$(grep -v '^#' "$HOME/.bashrc" 2>/dev/null | grep 'HF_HOME=' | head -1)
if [ -n "$BASHRC_HFHOME" ]; then
  fail "~/.bashrc sets HF_HOME: ${BASHRC_HFHOME}"
  info "  This causes the interactive shell to re-download base models to the share"
  info "  instead of using the service's cache.  Remove it:"
  info "    sed -i '/HF_HOME=/d' ~/.bashrc"
  info "    source ~/.bashrc"
else
  pass "~/.bashrc does not set HF_HOME"
fi

# ── 5. API keys ──────────────────────────────────────────────────────────────

print_header "5 — The trainer's keys (.env) and the launcher's verdict"

# Values are NEVER printed. This script's output gets pasted into issues; a key
# that leaks that way has to be rotated everywhere it is configured.
#
# This box is the TRAINER, not the gateway. It never reads ATR_API_KEY (#9): the
# gateway on idhefix holds that one. What .env here must carry is the key the
# gateway sends on /train/* (ATR_TRAIN_API_KEY, the same name and value on both
# hosts), the one address it may come from (ATR_TRAIN_ALLOWED_CLIENTS), and the
# gateway's key under its own name here (ATR_TRAIN_GATEWAY_API_KEY), which the
# promotion gate presents to /ocr. Until 11.10.2026 this section still asked for
# ATR_API_KEY and failed on every correct asteraix .env.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_DIR}/.env"

env_value() {  # $1 = variable name; empty string when absent
  [ -f "$ENV_FILE" ] || { echo ""; return; }
  grep -E "^${1}=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2-
}

TRAIN_KEY=$(env_value ATR_TRAIN_API_KEY)
ALLOWED=$(env_value ATR_TRAIN_ALLOWED_CLIENTS)
GATE_KEY=$(env_value ATR_TRAIN_GATEWAY_API_KEY)
GATE_URL=$(env_value ATR_TRAIN_GATEWAY_URL)

if [ ! -f "$ENV_FILE" ]; then
  fail "no .env at ${ENV_FILE} — the unit loads its configuration from it"
else
  if [ -z "$TRAIN_KEY" ]; then
    fail ".env has no ATR_TRAIN_API_KEY — the launcher refuses a non-loopback bind without it"
    info "  Generate one and set the SAME value as ATR_TRAIN_API_KEY on idhefix (OPERATIONS.md):"
    info "    python3 -c \"import secrets; print(secrets.token_urlsafe(48))\""
  elif [ "${#TRAIN_KEY}" -lt 32 ]; then
    fail "ATR_TRAIN_API_KEY is shorter than 32 characters — the launcher refuses it"
  else
    pass ".env sets ATR_TRAIN_API_KEY (fingerprint $(fingerprint "$TRAIN_KEY"))"
    info "  idhefix must hold the same value; compare without printing, on both hosts:"
    info "    grep '^ATR_TRAIN_API_KEY=' .env | cut -d= -f2- | sha256sum"
  fi

  if [ -z "$ALLOWED" ]; then
    fail ".env has no ATR_TRAIN_ALLOWED_CLIENTS — the launcher refuses a non-loopback bind"
    info "  ufw does not filter :8204 here; the allowlist is the only source restriction."
    info "    ATR_TRAIN_ALLOWED_CLIENTS=130.92.59.240"
  else
    pass "ATR_TRAIN_ALLOWED_CLIENTS = ${ALLOWED}"
  fi

  # The promotion gate (#36) posts a held-out page to the gateway's /ocr. With no
  # key it authenticates as nobody, every kraken model registers disabled with
  # that as the reason, and nothing is ever promoted into /models. Nothing fails
  # loudly: the job completes, the record says "not promoted", and the cause is
  # a blank variable set weeks earlier.
  if [ -z "$GATE_KEY" ]; then
    fail ".env has no ATR_TRAIN_GATEWAY_API_KEY — the promotion gate cannot authenticate"
    info "  Trained kraken models will register disabled and never be promoted, silently."
    info "  The value is idhefix's ATR_API_KEY; compare the hashes as above."
  elif [ "$GATE_KEY" = "$TRAIN_KEY" ]; then
    fail "ATR_TRAIN_GATEWAY_API_KEY equals ATR_TRAIN_API_KEY — the two keys must differ (#9)"
    info "  A leaked inference key must not also be able to start training runs."
  else
    pass "ATR_TRAIN_GATEWAY_API_KEY is set (fingerprint $(fingerprint "$GATE_KEY")); hash it against idhefix's ATR_API_KEY"
  fi

  if [ -z "$GATE_URL" ]; then
    warn ".env has no ATR_TRAIN_GATEWAY_URL — the gate does not run, every model stays disabled"
    info "    ATR_TRAIN_GATEWAY_URL=http://130.92.59.240:8200"
  else
    pass "ATR_TRAIN_GATEWAY_URL = ${GATE_URL}"
  fi
fi

# The launcher's own verdict is the one that decides whether the unit starts
# (exit 2 = refused). It reads .env from the working directory, so run it there.
LAUNCHER_PY="${REPO_DIR}/.venvs/kraken-train/bin/python"
if [ -x "$LAUNCHER_PY" ]; then
  if LAUNCHER_OUT=$(cd "$REPO_DIR" && PYTHONPATH="${REPO_DIR}/src" "$LAUNCHER_PY" -m atr_training.serve --check --host 0.0.0.0 2>&1); then
    pass "launcher check: the unit would start (python -m atr_training.serve --check)"
  else
    fail "launcher check refused the configuration:"
    echo "$LAUNCHER_OUT" | tail -5 | sed 's/^/        /'
  fi
else
  warn "no ${LAUNCHER_PY} — the launcher's own check was skipped (bash scripts/make_venvs.sh)"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

print_header "Summary"
echo ""
if [ "$FAILURES" -gt 0 ]; then
  echo "  ✗  ${FAILURES} failure(s) — MUST fix before training"
elif [ "$WARNINGS" -gt 0 ]; then
  echo "  ⚠  ${WARNINGS} warning(s) — review before production training"
else
  echo "  ✓  Environment is clean"
fi

echo ""
echo "Most common fix:"
echo "    mkdir -p ~/atr-cache/tmp"
echo "    echo 'TMPDIR=~/atr-cache/tmp' >> ~/.env"
echo "    sed -i '/TMPDIR=/d' ~/.bashrc"
echo "    source ~/.bashrc"

if [ "$FAILURES" -gt 0 ]; then
  exit 1
fi
exit 0