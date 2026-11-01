#!/usr/bin/env bash
# Install the weekly image-path probe as a systemd USER unit on asteraix (#199).
#
#   bash scripts/install_image_probe_timer.sh            # install, enable, arm the timer
#   bash scripts/install_image_probe_timer.sh --run-now  # … and take the first measurement now
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

mkdir -p "${DST}"
cp "${ROOT}/deploy/systemd/atr-image-probe.service" "${DST}/atr-image-probe.service"
cp "${ROOT}/deploy/systemd/atr-image-probe.timer" "${DST}/atr-image-probe.timer"
systemctl --user daemon-reload
systemctl --user enable --now atr-image-probe.timer

if ! loginctl show-user "${USER}" 2>/dev/null | grep -q 'Linger=yes'; then
  echo "WARNING: linger is OFF — the timer stops on logout. Try: loginctl enable-linger" >&2
fi

if [ "${1:-}" = "--run-now" ]; then
  systemctl --user start atr-image-probe.service
  journalctl --user -u atr-image-probe.service -n 20 --no-pager
fi
systemctl --user list-timers atr-image-probe.timer --no-pager
