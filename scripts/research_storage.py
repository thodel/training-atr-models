#!/usr/bin/env python3
"""Wie voll der research-storage ist, und was sich woanders besser aufhebt.

Der research-storage ist `/storage/research/wbkolleg_dh_1` auf UBELIX — der
Gruppenbereich des Walter Benjamin Kollegs DH, auf dem GPFS-Dateisystem
`rs_gpfs`. Dort liegt auch `HF_HOME`, also der HuggingFace-Zwischenspeicher der
Trainingsläufe.

**Stand 09.10.2026: 642 GiB von 12 TiB frei, also 5,2 %.** Gemessen liegen allein
in `Textrecognition_Training` rund 5,7 TB: etwa **2,7 TB HF-Zwischenspeicher**
(vollständig aus dem Hub wiederherstellbar, und zu einem guten Teil doppelt) und
**1,8 TB Arbeitsverzeichnisse** abgeschlossener Läufe, für die hier — anders als
auf `/scratch` — keine Löschregel gilt.

    python3 scripts/research_storage.py --check       # nur df, sofort
    python3 scripts/research_storage.py --suggest     # + Vorschläge
    python3 scripts/research_storage.py --inventory    # volles du, als Slurm-Job

Nur Standardbibliothek: das soll auf einem UBELIX-Login-Knoten mit `python3`
laufen, ohne venv und ohne Container.

**Was dieses Skript nicht tut: über fremde Daten urteilen.** Der Bereich hat 39
Verzeichnisse auf der obersten Ebene, die zu vielen Projekten und Personen
gehören — `Projekt_Bullinger`, `Backup_Humboldt`, `Lehre`, `Omeka` und so weiter.
Vorschläge macht es ausschliesslich innerhalb von `Textrecognition_Training`, weil
nur dort bekannt ist, was wiederherstellbar ist. Alles andere wird mit seiner
Grösse berichtet und ohne Empfehlung.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("ATR_RESEARCH_STORAGE",
                           "/storage/research/wbkolleg_dh_1"))
#: Unser Unterbereich. Nur hier wird etwas vorgeschlagen.
OURS = "Textrecognition_Training"
#: Wohin der Inventarlauf schreibt. Im Home, nicht auf dem Bereich selbst:
#: ein Bericht über einen vollen Speicher soll nicht auf ihm liegen.
INVENTORY = Path(os.environ.get(
    "ATR_STORAGE_INVENTORY",
    str(Path.home() / "ubelix" / "research-storage-inventory.json")))
#: Unter diesem Anteil frei werden Vorschläge gemacht.
THRESHOLD = 0.10
#: Ab wann ein Inventar als veraltet gilt. Ein alter Bericht ist nicht falsch,
#: aber er ist etwas anderes als ein frischer, und das muss dastehen.
STALE_DAYS = 14

#: Was im Zwischenspeicher liegt, ist aus dem Hub wiederherstellbar — das ist
#: die Eigenschaft eines Zwischenspeichers. Die Reihenfolge ist die Rangfolge
#: der Vorschläge: zuerst, was nachweislich doppelt liegt.
RECONSTRUCTIBLE = (
    ("hf_hub/datasets--*", "HF-Zwischenspeicher in der ALTEN Form. Am 09.10.2026 "
     "gemessen: 36 der 37 Datensätze liegen auch in `hf_hub/hub/`, also 97 % "
     "doppelt. Wird beim nächsten Zugriff neu geholt."),
    ("hf_hub/models--*", "Basismodelle in der alten Cache-Form. Alle aus dem Hub, "
     "und `prefetch_bases.sh` holt sie gezielt wieder."),
    ("hf_hub/xet", "Xet-Brockenspeicher des Zwischenspeichers."),
    ("training_folder/tmp", "Temporär."),
    ("tmp", "Temporär."),
)

#: Wiederherstellbar, aber **nicht** lokal doppelt — die Wiederherstellung ist
#: ein Download in Terabyte-Grösse. Eigene Klasse, weil der Unterschied zur
#: ersten Gruppe der ganze Punkt ist: dort wird eine zweite Kopie gelöscht, hier
#: die einzige.
#:
#: Dieser Eintrag ist ein Fehler von mir, den die Messung gefangen hat. Ich hatte
#: `hub/` als Dublette geführt, weil derselbe Datensatzname auch in `hf_hub/hub/`
#: steht. Gemessen am 09.10.2026: `hub/` hält **992 GB**, der Eintrag in
#: `hf_hub/hub/` **21 KB** — ein Stummel aus Metadaten mit einem einzigen Blob.
#: Gleiche Revision notiert, Daten nur an einer Stelle. Übereinstimmende Namen
#: sind kein Beweis für Doppelung.
EXPENSIVE_TO_RESTORE = (
    ("hub", "Ein zweiter HF-Zwischenspeicher neben `hf_hub`, entstanden als "
     "HF_HOME eine Ebene höher zeigte. Er hält genau einen Datensatz, "
     "`image-text_medieval-scripts_xiv-xv-xvi`, und zwar **als einzige lokale "
     "Kopie** — der gleichnamige Eintrag in `hf_hub/hub/` ist ein 21-KB-Stummel. "
     "Der Datensatz liegt öffentlich auf HuggingFace mit genau dieser Revision "
     "(`729e9b2721ba`, 1.098 GB), ist also wiederherstellbar — aber als "
     "Terabyte-Download, nicht als freier Gewinn."),
)

#: Materialisierte Arbeitsverzeichnisse abgeschlossener Läufe. Auch
#: wiederherstellbar, aber **teuer**: die Korpora entstehen in der
#: prepare-Stufe, und die braucht Stunden (gemessen: 2 h 35 für elf Datensätze).
#: Darum eigene Klasse — und je Job zu entscheiden, nicht als Ganzes.
DERIVED = (
    ("training_folder/jobs", "Arbeitsverzeichnisse der Läufe: zugeschnittene "
     "Zeilenbilder, Manifeste, Logs. Aus den HF-Datensätzen wiederherstellbar, "
     "aber nur über die prepare-Stufe, die Stunden braucht. **Je Job "
     "entscheiden**: ein abgeschlossener Lauf braucht sein Verzeichnis nicht "
     "mehr, ein vorbereiteter schon. Anders als `/scratch/.../runs/jobs` läuft "
     "hier keine 30-Tage-Regel, es wird also nie von selbst leer."),
)

#: Trainierte Gewichte. Die gehören auf HuggingFace, und wo sie dort liegen, ist
#: die lokale Kopie entbehrlich — aber das ist je Modell zu prüfen, nicht
#: pauschal, darum eigene Klasse.
PUBLISHABLE = (
    ("training_folder/trained", "Trainierte Gewichte. Wo das Modell unter "
     "`dh-unibe/` auf HuggingFace liegt, ist die lokale Kopie entbehrlich — "
     "je Modell prüfen, nicht pauschal."),
    ("trained-ubelix", "Trainierte Gewichte aus UBELIX-Läufen, dito."),
    ("training_folder/bases", "Vier Einträge, am 09.10.2026 zusammen 47 MB — "
     "kraken-Modelle aus Zenodo-Hinterlegungen, nicht die HF-Basismodelle. "
     "Lohnt die Löschung nicht; steht hier, damit niemand sie dafür hält."),
)


def df(path: Path) -> dict:
    """Belegung, aus `shutil.disk_usage` statt aus dem Text von `df`."""
    total, used, free = shutil.disk_usage(path)
    return {"total": total, "used": used, "free": free,
            "free_fraction": free / total if total else 0.0}


def gib(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.1f} {unit}".replace(",", "'")
        n /= 1024
    return f"{n} TiB"


def du(path: Path) -> int:
    """Bytes unter `path`, über `du -sb`. Fehlt der Pfad, ist es 0."""
    if not path.exists():
        return 0
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True,
                         text=True, check=False).stdout
    try:
        return int(out.split("\t", 1)[0])
    except (ValueError, IndexError):
        return 0


def inventory(root: Path = ROOT, write: Path | None = None,
              with_top: bool = True) -> dict:
    """Der teure Lauf: je Verzeichnis eine Grösse. Gehört in einen Slurm-Job.

    **Schreibt nach jeder Messung, nicht am Ende.** Der erste Versuch am
    09.10.2026 lief 6 Stunden in den Timeout und hinterliess **nichts**, weil
    erst der Abschluss geschrieben hätte. Ein Teilergebnis mit Datum ist
    brauchbar; sechs Stunden ohne Ergebnis sind es nicht.

    **Unser Unterbaum zuerst.** Nur er trägt die Vorschläge, und er ist in
    Minuten gemessen. Die 38 übrigen Verzeichnisse der obersten Ebene sind rund
    6 TB fremder Projektdaten, kosten Stunden und liefern nur die Zeile „die
    grössten ausserhalb unseres Bereichs". Mit ``with_top=False`` bleiben sie
    weg.
    """
    record = {"root": str(root), "measured": time.time(), "usage": df(root),
              "top": {}, "ours": {}, "complete": False}

    def save() -> None:
        if write is None:
            return
        write.parent.mkdir(parents=True, exist_ok=True)
        tmp = write.with_suffix(write.suffix + ".part")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(write)

    ours = root / OURS
    if ours.is_dir():
        for entry in sorted(p for p in ours.iterdir() if p.is_dir()):
            record["ours"][entry.name] = du(entry)
            save()
        cache = ours / "hf_hub"
        if cache.is_dir():
            for pattern in ("datasets--*", "models--*"):
                record["ours"][f"hf_hub/{pattern}"] = sum(
                    du(p) for p in cache.glob(pattern))
                save()
            for name in ("hub", "xet", "modules"):
                record["ours"][f"hf_hub/{name}"] = du(cache / name)
                save()
        folder = ours / "training_folder"
        if folder.is_dir():
            for entry in sorted(p for p in folder.iterdir() if p.is_dir()):
                record["ours"][f"training_folder/{entry.name}"] = du(entry)
                save()
    record["ours_complete"] = True
    save()

    if with_top:
        for entry in sorted(p for p in root.iterdir() if p.is_dir()):
            record["top"][entry.name] = du(entry)
            save()
    record["complete"] = True
    save()
    return record


def load_inventory() -> tuple[dict | None, str]:
    """Das letzte Inventar und wie alt es ist.

    Kein Inventar und ein altes Inventar sind zwei Zustände, nicht einer (#165):
    ohne Inventar gibt es keine Vorschläge, mit einem alten schon, aber mit
    Datum daneben.
    """
    if not INVENTORY.exists():
        return None, (f"kein Inventar unter {INVENTORY} — ohne eines gibt es "
                      "keine Vorschläge. Mit --inventory erheben, am besten als "
                      "Slurm-Job.")
    try:
        record = json.loads(INVENTORY.read_text())
    except (OSError, ValueError) as exc:
        return None, f"{INVENTORY} ist nicht lesbar: {exc}"
    age_days = (time.time() - record.get("measured", 0)) / 86400
    note = f"Inventar vom {time.strftime('%d.%m.%Y %H:%M', time.localtime(record['measured']))}"
    if age_days > STALE_DAYS:
        note += f" — **{age_days:.0f} Tage alt**, die Zahlen können abweichen"
    if not record.get("ours_complete"):
        note += (" — **abgebrochen, bevor unser Unterbaum fertig war**: die "
                 "Vorschläge sind unvollständig")
    elif not record.get("complete"):
        note += (" — unser Unterbaum ist vollständig, die übrigen Verzeichnisse "
                 "der obersten Ebene nicht")
    return record, note


def suggestions(record: dict) -> list[dict]:
    """Was sich woanders besser aufhebt, nach Grösse geordnet.

    Nur innerhalb von `Textrecognition_Training`. Jeder Vorschlag nennt, wo das
    Material sonst liegt — ein Vorschlag ohne diese Angabe wäre eine Bitte,
    etwas auf Verdacht zu löschen.
    """
    out = []
    for key, why in RECONSTRUCTIBLE:
        size = record.get("ours", {}).get(key, 0)
        if size:
            out.append({"path": f"{OURS}/{key}", "bytes": size,
                        "class": "wiederherstellbar", "why": why})
    for key, why in EXPENSIVE_TO_RESTORE:
        size = record.get("ours", {}).get(key, 0)
        if size:
            out.append({"path": f"{OURS}/{key}", "bytes": size,
                        "class": "einzige lokale Kopie, aber auf HF",
                        "why": why})
    for key, why in DERIVED:
        size = record.get("ours", {}).get(key, 0)
        if size:
            out.append({"path": f"{OURS}/{key}", "bytes": size,
                        "class": "abgeleitet, Wiederherstellung teuer", "why": why})
    for key, why in PUBLISHABLE:
        size = record.get("ours", {}).get(key, 0)
        if size:
            out.append({"path": f"{OURS}/{key}", "bytes": size,
                        "class": "gehört auf HuggingFace", "why": why})
    return sorted(out, key=lambda s: -s["bytes"])


def report(check_only: bool, with_suggestions: bool) -> int:
    usage = df(ROOT)
    free_pct = usage["free_fraction"] * 100
    low = usage["free_fraction"] < THRESHOLD
    print(f"research-storage {ROOT}")
    print(f"  {gib(usage['free'])} frei von {gib(usage['total'])} "
          f"— {free_pct:.1f} %")
    print(f"  Schwelle: {THRESHOLD * 100:.0f} % — "
          f"{'UNTERSCHRITTEN' if low else 'eingehalten'}")
    if check_only and not low:
        return 0
    if not (low or with_suggestions):
        return 0

    record, note = load_inventory()
    print(f"\n  {note}")
    if record is None:
        return 1 if low else 0

    items = suggestions(record)
    if not items:
        print("  keine Vorschläge aus dem Inventar")
        return 1 if low else 0

    print(f"\n  Was sich woanders besser aufhebt "
          f"(nur unter {OURS}/):\n")
    total = 0
    for item in items:
        total += item["bytes"]
        print(f"  {gib(item['bytes']):>12}  {item['path']}")
        print(f"                {item['class']}: {item['why']}")
    print(f"\n  {gib(total)} zusammen — das wären "
          f"{(usage['free'] + total) / usage['total'] * 100:.1f} % frei.")

    top = record.get("top", {})
    rest = {k: v for k, v in top.items() if k != OURS}
    if rest:
        biggest = sorted(rest.items(), key=lambda kv: -kv[1])[:5]
        print("\n  Die grössten Verzeichnisse ausserhalb unseres Bereichs — "
              "ohne Empfehlung,\n  weil sie anderen Projekten gehören:")
        for name, size in biggest:
            print(f"  {gib(size):>12}  {name}")
    return 1 if low else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="nur die Belegung; Vorschläge nur bei Unterschreitung")
    p.add_argument("--suggest", action="store_true",
                   help="Vorschläge auch, wenn die Schwelle eingehalten ist")
    p.add_argument("--inventory", action="store_true",
                   help="das teure du erheben und schreiben (Slurm-Job)")
    p.add_argument("--ours-only", action="store_true",
                   help=f"nur {OURS} messen, nicht die 38 fremden Verzeichnisse "
                        "der obersten Ebene — das ist der Teil, der Stunden kostet "
                        "und keine Vorschläge trägt")
    p.add_argument("--json", action="store_true", help="Maschinenform")
    args = p.parse_args(argv)

    if not ROOT.exists():
        print(f"{ROOT} gibt es hier nicht — dieses Skript läuft auf UBELIX",
              file=sys.stderr)
        return 2

    if args.inventory:
        record = inventory(write=INVENTORY, with_top=not args.ours_only)
        print(f"Inventar geschrieben: {INVENTORY}")
        print(f"  {len(record['ours'])} Posten unter {OURS}"
              f"{' (vollständig)' if record.get('ours_complete') else ''}")
        print(f"  {len(record['top'])} Verzeichnisse auf der obersten Ebene"
              f"{' (vollständig)' if record.get('complete') else ' — unvollständig'}")
        return 0

    if args.json:
        record, note = load_inventory()
        print(json.dumps({"usage": df(ROOT), "inventory_note": note,
                          "below_threshold": df(ROOT)["free_fraction"] < THRESHOLD,
                          "suggestions": suggestions(record) if record else []},
                         indent=2))
        return 0

    return report(check_only=args.check, with_suggestions=args.suggest)


if __name__ == "__main__":
    raise SystemExit(main())
