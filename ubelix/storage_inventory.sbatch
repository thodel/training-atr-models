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
# CPU only, 2 Kerne, und absichtlich mit kurzer Wandzeit: job_gratis deckelt
# cpus x Minuten bei 11.520 über die laufenden Jobs, und ein Inventarlauf soll
# keinem Training die Minuten wegnehmen (gemessen 09.10.2026: ein prepare mit
# der Vorgabe von 20 h blockierte sich selbst).
#SBATCH --job-name=storage-inventory
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out
set -euo pipefail
REPO=${ATR_TRAIN_REPO:-$HOME/training-atr-models}
echo "== $(date -Is) Inventar des research-storage"
python3 "$REPO/scripts/research_storage.py" --inventory
echo
python3 "$REPO/scripts/research_storage.py" --suggest
echo "== $(date -Is) fertig"
