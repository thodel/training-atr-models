#!/bin/bash
# Einen Löschplan ausführen, als Slurm-Job (#387).
#
#   ubelix/apply_purge.sh ~/ubelix/plans/purge-20261106T091233Z.json
#
# Den Plan schreibt ubelix/plan_purge.py --write; er ist vorher zu lesen, auf
# dem Login-Knoten oder über die ATR-MCP. Diese eine Zeile ist dann alles, was
# ein Mensch tippt: kein Block zum Kopieren, keine Liste von Hand (#204).
#
# Auf dem Login-Knoten prüft das Skript den Plan (Prüfsumme der Einträge),
# nimmt die sha256 der Datei und reicht SICH SELBST mit diesen Kopfzeilen ein —
# epyc2, job_gratis, und --nodes=1, ohne das sbatch ein --mem ablehnt. Im Job
# (SLURM_JOB_ID gesetzt) prüft plan_purge.py --apply vor jedem Eintrag erneut:
# Datei unverändert, Status unverändert, nicht geschützt. Nur dann fällt data/;
# job.json, logs/ und die Energie-Dateien bleiben. Das Log steht unter
# ~/ubelix/logs/purge-<slurm-id>.out: df vorher und nachher, eine Zeile je
# Eintrag (removed oder skip mit Grund), am Ende die Zählung.
#SBATCH --job-name=purge
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  # ── Login-Knoten: prüfen und einreichen ──
  if [ $# -ne 1 ]; then
    echo "usage: $0 <plan.json>   (geschrieben von ubelix/plan_purge.py --write)" >&2
    exit 2
  fi
  here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  plan=$(readlink -f "$1")
  if [ ! -f "$plan" ]; then
    echo "refused: kein Plan unter $1" >&2
    exit 1
  fi
  python3 "$here/plan_purge.py" --verify "$plan" || exit 1
  sha=$(sha256sum "$plan" | cut -d' ' -f1)
  repo=$(cd "$here/.." && pwd)
  commit=$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)
  mkdir -p "$HOME/ubelix/logs"
  exec sbatch --export="ALL,PURGE_PLAN=$plan,PURGE_SHA256=$sha,ATR_TRAIN_REPO=$repo,ATR_CODE_COMMIT=$commit" \
    "$here/apply_purge.sh"
fi

# ── Im Slurm-Job ──
: "${PURGE_PLAN:?PURGE_PLAN fehlt — mit ubelix/apply_purge.sh <plan> einreichen}"
: "${PURGE_SHA256:?PURGE_SHA256 fehlt — mit ubelix/apply_purge.sh <plan> einreichen}"
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}
# shellcheck source=pin_code.sh
source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
root=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["jobs_root"])' "$PURGE_PLAN")
echo "== $(date -Is) Plan $PURGE_PLAN"
echo "   sha256 $PURGE_SHA256"
echo "== df vorher"
df -h "$root"
status=0
python3 "$REPO/ubelix/plan_purge.py" --apply "$PURGE_PLAN" --sha256 "$PURGE_SHA256" || status=$?
echo "== df nachher"
df -h "$root"
echo "== $(date -Is) fertig (exit $status)"
exit "$status"
