#!/usr/bin/env bash
# Install the atr-train systemd USER unit on asteraix (no root needed).
#
#   bash scripts/install_user_unit.sh             # install, enable, start
#   bash scripts/install_user_unit.sh --no-start  # install and enable only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${HOME}/.config/systemd/user"
START=1
[ "${1:-}" = "--no-start" ] && START=0

# The unit hardcodes ~/Repo/training-atr-models. Refuse to install a unit that
# points somewhere this checkout is not.
if [ "${ROOT}" != "${HOME}/Repo/training-atr-models" ]; then
  echo "ERROR: this checkout is at ${ROOT}, but the unit expects ~/Repo/training-atr-models" >&2
  exit 1
fi
[ -f "${ROOT}/.env" ] || { echo "ERROR: ${ROOT}/.env missing — copy .env.example first" >&2; exit 1; }
[ -x "${ROOT}/.venvs/kraken-train/bin/python" ] || {
  echo "ERROR: kraken-train venv missing — the service runs in it. bash scripts/make_venvs.sh" >&2; exit 1; }

mkdir -p "${DST}"
cp "${ROOT}/deploy/systemd/atr-train.service" "${DST}/atr-train.service"
systemctl --user daemon-reload
systemctl --user enable atr-train.service

if ! loginctl show-user "${USER}" 2>/dev/null | grep -q 'Linger=yes'; then
  echo "WARNING: linger is OFF — the service stops on logout. Try: loginctl enable-linger" >&2
fi

if [ "${START}" -eq 1 ]; then
  systemctl --user start atr-train.service
  sleep 3
  systemctl --user --no-pager status atr-train.service | head -8 || true
  echo; curl -s --max-time 10 localhost:8204/health | head -c 400; echo
fi
