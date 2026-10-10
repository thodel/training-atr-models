#!/bin/bash
# Das teure du über den research-storage, einmal die Woche.
#
# `research_storage.py --check` ist sofort (ein statfs). Die Aufschlüsselung je
# Verzeichnis ist es nicht: über 11 TB auf GPFS braucht sie Minuten bis Stunden,
# und sie gehört deshalb nicht in eine tägliche Prüfung, sondern hierher. Die
# tägliche Prüfung liest das Ergebnis und sagt, wie alt es ist.
#
#   sbatch ubelix/storage_inventory.sbatch
#
# Gemessen am 09./10.10.2026, und beides hat diese Datei geformt:
#
#   Ein voller Durchlauf über alle 39 Verzeichnisse der obersten Ebene lief
#   SECHS STUNDEN in den Timeout, ohne etwas zu schreiben. Die 38 fremden
#   Verzeichnisse sind rund 6 TB und tragen keine Vorschläge — nur die Zeile
#   "die grössten ausserhalb unseres Bereichs". Darum `--ours-only`: unser
#   Unterbaum ist in Minuten gemessen und ist der Teil, auf den es ankommt.
#   Das Skript schreibt seit dem 10.10. nach jeder Messung, nicht am Ende.
#
#   Mit ATR_STORAGE_FULL=1 wird doch alles gemessen — dann aber mit viel Zeit,
#   und das Teilergebnis steht nach jedem Verzeichnis in der Datei.
#
# CPU only, 2 Kerne: job_gratis deckelt cpus x Minuten bei 11.520 über die
# laufenden Jobs, und ein Inventarlauf soll keinem Training die Minuten
# wegnehmen (ein prepare mit der Vorgabe von 20 h blockierte sich selbst).
#SBATCH --job-name=storage-inventory
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out
set -euo pipefail
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}
# shellcheck source=pin_code.sh
source "$REPO/ubelix/pin_code.sh"; pin_code || exit 1
echo "== $(date -Is) Inventar des research-storage"
if [ "${ATR_STORAGE_FULL:-0}" = "1" ]; then
  echo "   voller Durchlauf, auch die fremden Verzeichnisse — das dauert Stunden"
  python3 "$REPO/scripts/research_storage.py" --inventory
else
  python3 "$REPO/scripts/research_storage.py" --inventory --ours-only
fi
echo
python3 "$REPO/scripts/research_storage.py" --suggest
echo "== $(date -Is) fertig"
