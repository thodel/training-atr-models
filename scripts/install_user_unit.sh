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

# The unit binds beyond loopback (#13), and the launcher refuses that bind unless
# .env holds the key, the allowlist and require_auth. Ask it now, with the unit's
# own --host/--port, rather than install a unit that exits 2 on its first start.
# From ${ROOT}, so the settings read ${ROOT}/.env the way the unit's
# EnvironmentFile= does.
UNIT="${ROOT}/deploy/systemd/atr-train.service"
BIND_HOST="$(sed -n 's/^ExecStart=.* --host \([^ ]*\).*/\1/p' "${UNIT}")"
BIND_PORT="$(sed -n 's/^ExecStart=.* --port \([^ ]*\).*/\1/p' "${UNIT}")"
[ -n "${BIND_HOST}" ] && [ -n "${BIND_PORT}" ] || {
  echo "ERROR: no --host/--port on the ExecStart line of ${UNIT}" >&2; exit 1; }
(cd "${ROOT}" && PYTHONPATH="${ROOT}/src" "${ROOT}/.venvs/kraken-train/bin/python" \
   -m atr_training.serve --check --host "${BIND_HOST}" --port "${BIND_PORT}") || {
  echo "ERROR: fix .env as above; the unit was NOT installed" >&2; exit 1; }

mkdir -p "${DST}"
cp "${ROOT}/deploy/systemd/atr-train.service" "${DST}/atr-train.service"
systemctl --user daemon-reload
systemctl --user enable atr-train.service

if ! loginctl show-user "${USER}" 2>/dev/null | grep -q 'Linger=yes'; then
  echo "WARNING: linger is OFF — the service stops on logout. Try: loginctl enable-linger" >&2
fi

if [ "${START}" -eq 1 ]; then
  # `start` does nothing to a running unit, so a redeploy would keep serving the
  # old code — and the old bind — while this script reports success.
  if systemctl --user --quiet is-active atr-train.service; then
    echo "NOTE: atr-train is already running the previous code. Apply this with:" >&2
    echo "      systemctl --user restart atr-train.service" >&2
    echo "      (KillMode=process: a running job survives the restart)" >&2
  fi
  systemctl --user start atr-train.service
  sleep 3
  systemctl --user --no-pager status atr-train.service | head -8 || true
  echo; curl -s --max-time 10 localhost:8204/health | head -c 400; echo
fi
