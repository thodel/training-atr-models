#!/usr/bin/env bash
# Install the dataset-discovery timer as a systemd USER unit on asteraix (#32).
#
#   bash scripts/install_discovery_timer.sh            # install, enable, arm the timer
#   bash scripts/install_discovery_timer.sh --run-now  # … and take the baseline now
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${HOME}/.config/systemd/user"
PY="${ROOT}/.venvs/kraken-train/bin/python"

# The units hardcode ~/Repo/training-atr-models, like atr-train.service.
if [ "${ROOT}" != "${HOME}/Repo/training-atr-models" ]; then
  echo "ERROR: this checkout is at ${ROOT}, but the units expect ~/Repo/training-atr-models" >&2
  exit 1
fi
[ -x "${PY}" ] || { echo "ERROR: kraken-train venv missing — bash scripts/make_venvs.sh" >&2; exit 1; }
"${PY}" -c "import huggingface_hub" 2>/dev/null || {
  echo "ERROR: the kraken-train venv has no huggingface_hub" >&2; exit 1; }
# The dh-unibe datasets are gated. Whether a token is there is checked by the
# library itself; its value is never read here.
"${PY}" -c "import huggingface_hub as h, sys; sys.exit(0 if h.get_token() else 1)" || {
  echo "ERROR: no HuggingFace token — run 'hf auth login' on this box first" >&2; exit 1; }

mkdir -p "${DST}"
cp "${ROOT}/deploy/systemd/atr-discovery.service" "${DST}/atr-discovery.service"
cp "${ROOT}/deploy/systemd/atr-discovery.timer" "${DST}/atr-discovery.timer"
systemctl --user daemon-reload
systemctl --user enable --now atr-discovery.timer

if ! loginctl show-user "${USER}" 2>/dev/null | grep -q 'Linger=yes'; then
  echo "WARNING: linger is OFF — the timer stops on logout. Try: loginctl enable-linger" >&2
fi

if [ "${1:-}" = "--run-now" ]; then
  systemctl --user start atr-discovery.service
  journalctl --user -u atr-discovery.service -n 5 --no-pager
fi
systemctl --user list-timers atr-discovery.timer --no-pager
