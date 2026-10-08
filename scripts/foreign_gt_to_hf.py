#!/usr/bin/env python3
"""Fremde Ground Truth → HF-Datensatz, eine Quelle je Aufruf (#187).

Klont eine Quelle aus dem Register (:mod:`atr_training.foreign_gt`) flach, wählt
ihre PAGE-Dokumente nach Git-Modus und Seitenstamm aus, holt die Seitenbilder bei
der haltenden Einrichtung und schreibt den dh-unibe-Datensatzaufbau —
``data/train/<project>/0000.parquet`` mit ``image`` / ``xml_content`` /
``filename`` / ``project_name``, denselben vier Spalten, die der Trainingspfad
ohnehin liest.

    .venvs/kraken-train/bin/python scripts/foreign_gt_to_hf.py --list
    .venvs/kraken-train/bin/python scripts/foreign_gt_to_hf.py reichsanzeiger-gt --dry-run
    .venvs/kraken-train/bin/python scripts/foreign_gt_to_hf.py reichsanzeiger-gt

**Der Datensatz wird privat angelegt, und ``--public`` wird verweigert, solange die
Bildrechte der Quelle ``ungeprüft`` sind.** Die Lizenz der Transkription sagt
nichts über die Rechte am Scan: das sind zwei Fragen mit zwei Rechteinhabern
(#193). Für unser eigenes Training ändert das nichts — die Bilder sind im
Datensatz. Es begrenzt nur, wer ihn sehen darf. Öffentlich schaltet ein Mensch,
in der HF-Oberfläche, nachdem die Bildrechte geklärt sind.

Clone-Verzeichnisse bleiben stehen: ein zweiter Lauf über dieselbe Quelle soll
nicht erneut 61 MB ziehen, und ein Klon auf der Platte ist der einzige Weg, die
Auswahl später nachzuprüfen.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.foreign_gt import (  # noqa: E402
    SOURCES,
    ForeignGtError,
    Source,
    dataset_card,
    deduplicate,
    head_commit,
    image_urls,
    source_by_id,
    tracked_pages,
)

#: Sekunden zwischen zwei Bildabrufen. Das sind die öffentlichen Bildserver von
#: Bibliotheken, keine CDN-Kapazität; reichsanzeiger-gt braucht 101 Abrufe, und
#: eine Einfuhr, die ihren eigenen Verkehr nicht überlebt, ist unser Problem.
DEFAULT_DELAY = 0.4
UA = "training-atr-models/foreign-gt (DH Bern; Forschung)"

#: Kurz, und das ist Absicht. archive.org verteilt ``/download/…`` per Redirect auf
#: wechselnde ``ia*.us.archive.org``-Knoten, und ein toter Knoten antwortet gar
#: nicht. Gemessen am 08.10.2026: sechs Abrufe desselben Bildes brauchten 3,5–4,6 s,
#: ein einziger hängender Knoten davor kostete mit 120 s Limit und vier Versuchen
#: acht Minuten — für ein Bild. Kurzes Limit plus Wiederholung landet beim nächsten
#: Versuch auf einem anderen Knoten; langes Limit wartet auf einen toten.
DEFAULT_TIMEOUT = 30


def _get(url: str, timeout: int = DEFAULT_TIMEOUT, retries: int = 4) -> bytes:
    """GET mit Backoff auf 429/5xx. Alles andere fliegt sofort."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
            wait = min(2 ** attempt, 30)
            print(f"      HTTP {exc.code}, warte {wait}s ({attempt}/{retries})", flush=True)
            time.sleep(wait)
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
            last = exc
    raise RuntimeError(f"unerreichbar: {last}")


def clone(source: Source, workdir: Path) -> Path:
    target = workdir / source.id
    if (target / ".git").is_dir():
        print(f"  Klon vorhanden: {target}")
        return target
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"  klone {source.clone_url} (flach) …", flush=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "-b", source.branch, source.clone_url, str(target)], check=True)
    return target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source_id", nargs="?", help="Quelle aus dem Register")
    p.add_argument("--list", action="store_true", help="Register zeigen und beenden")
    p.add_argument("--workdir", type=Path, default=Path.home() / "gt-src")
    p.add_argument("--out", type=Path, default=None,
                   help="wohin das Parquet geschrieben wird (Vorgabe: <workdir>/_hf/<id>)")
    p.add_argument("--limit", type=int, default=None, help="nur so viele Seiten")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="Sekunden je Bildabruf; kurz halten, siehe "
                        "DEFAULT_TIMEOUT")
    p.add_argument("--dry-run", action="store_true",
                   help="auswählen und berichten; kein Bild holen, nichts hochladen")
    p.add_argument("--keep-imageless", action="store_true",
                   help="Seiten ohne Bild-URL mitnehmen statt auslassen; die "
                        "Bildspalte ist dort leer")
    p.add_argument("--no-images", action="store_true",
                   help="nur XML einlesen, Bildspalte leer lassen")
    p.add_argument("--public", action="store_true",
                   help="öffentliches Repo — wird verweigert, solange die "
                        "Bildrechte ungeprüft sind")
    args = p.parse_args(argv)

    if args.list or not args.source_id:
        print(f"{'Quelle':<22}{'Lizenz':<12}{'Bildrechte':<12}Ziel")
        for s in SOURCES:
            print(f"{s.id:<22}{s.licence:<12}{s.image_rights:<12}{s.target}")
        return 0

    try:
        source = source_by_id(args.source_id)
    except ForeignGtError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.public and source.image_rights != "geprüft":
        print(f"--public verweigert: die Bildrechte von {source.id} sind "
              f"'{source.image_rights}'.\nDie Transkriptionslizenz "
              f"({source.licence}) deckt die Scans nicht. Erst klären "
              f"({source.image_source}), dann in der HF-Oberfläche von Hand "
              f"öffentlich schalten.", file=sys.stderr)
        return 3

    print(f"── {source.id} ─────────────────────────────────────────")
    print(f"  Lizenz:     {source.licence}  ({source.licence_at})")
    print(f"  Bildquelle: {source.image_source}")
    print(f"  Bildrechte: {source.image_rights}")

    root = clone(source, args.workdir)
    commit = head_commit(root)
    print(f"  Stand:      {commit[:12]} (Branch {source.branch})")
    sel = tracked_pages(root, source.xml_root)
    sel.source_id = source.id
    deduplicate(sel, source.prefer, source.variants)
    print(f"  {sel.report()}")
    if not sel.pages:
        print("  keine Seite ausgewählt", file=sys.stderr)
        return 1

    urls = {} if args.no_images else image_urls(root, source, sel.pages)
    have = sum(1 for page in sel.pages if page.path in urls)
    print(f"  Bild-URLs:  {have} von {len(sel.pages)} ausgewählten Seiten "
          f"zugeordnet")

    if not source.images and not args.no_images:
        print(f"\n  {source.id} hat keinen Bezugsweg für Bilder — nichts "
              f"hochgeladen.\n  Die Bilder liegen bei {source.image_source}. "
              f"Ein Datensatz ohne Bildspalte kann nicht trainieren und sieht in "
              f"einer Spezifikation aus wie einer, der es könnte; darum braucht "
              f"er die ausdrückliche Zustimmung --no-images.", file=sys.stderr)
        return 5

    pages = sel.pages
    if urls and not args.keep_imageless:
        imageless = [p for p in pages if p.path not in urls]
        if imageless:
            # Eine Seite ohne Bild ist für das Training nichts und sieht in der
            # Spalte aus wie eine, die eines hat. Lieber weniger Seiten als ein
            # Datensatz, der zur Hälfte stumm leer ist (#165).
            print(f"  {len(imageless)} Seiten ohne Bild-URL werden ausgelassen "
                  f"(--keep-imageless behält sie), z.B. "
                  f"{[p.stem for p in imageless[:4]]}")
            pages = [p for p in pages if p.path in urls]
            sel.pages = pages
    if args.limit:
        pages = pages[: args.limit]

    if args.dry_run:
        print("\n  --dry-run: nichts geholt, nichts geschrieben, nichts hochgeladen")
        missing = [p.stem for p in pages if p.path not in urls]
        if missing:
            print(f"  ohne Bild-URL: {len(missing)}  z.B. {missing[:4]}")
        print(f"\n  Karte (Anfang):\n")
        print("\n".join(dataset_card(source, sel, with_images=have,
                                     commit=commit).splitlines()[:22]))
        return 0

    rows: list[dict] = []
    fetched = failed = 0
    for i, page in enumerate(pages, start=1):
        xml = (root / page.path).read_text(encoding="utf-8", errors="replace")
        blob = b""
        name = page.stem
        url = urls.get(page.path)
        if url:
            try:
                blob = _get(url, timeout=args.timeout)
                fetched += 1
                name = Path(url.split("?")[0]).name or page.stem
            except Exception as exc:  # noqa: BLE001 — ein fehlendes Scan ist nicht fatal
                failed += 1
                print(f"    [{i}/{len(pages)}] {page.stem}: kein Bild ({exc})", flush=True)
            time.sleep(args.delay)
        rows.append({
            "image": {"bytes": blob, "path": name},
            "xml_content": xml,
            "filename": page.stem,
            "project_name": source.project,
            # Der Rückbezug je Eintrag, nicht nur je Datensatz: welche Datei im
            # Ursprungs-Repo diese Seite war, und ein Link darauf, der an den
            # gelesenen Commit genagelt ist.
            "source_path": page.path,
            "source_url": source.entry_url(page.path, commit),
        })
        if i % 20 == 0 or i == len(pages):
            print(f"    [{i}/{len(pages)}] {fetched} Bilder, {failed} fehlend", flush=True)

    if failed and not args.keep_imageless:
        # Dieselbe Regel wie für Seiten ohne URL, nur eine Schicht später: ein
        # fehlgeschlagener Abruf hinterlässt eine Zeile, die in der Spalte aussieht
        # wie eine mit Bild. Bei dach-gt waren das 27 von 98 — HTTP 404, weil die
        # Bild-URLs im Dokument veraltet sind.
        before = len(rows)
        rows = [r for r in rows if r["image"]["bytes"]]
        sel.pages = [p for p in sel.pages
                     if any(r["filename"] == p.stem for r in rows)]
        print(f"  {before - len(rows)} Zeilen ohne geholtes Bild werden "
              f"ausgelassen (--keep-imageless behält sie)")

    if not rows:
        print("\n  keine Zeile mit Bild übrig — nichts hochgeladen", file=sys.stderr)
        return 6

    print(f"\n  {len(rows)} Zeilen, {fetched} mit Bild, {failed} ohne")

    # Der dritte Zustand (#165): "kein Bild geholt" ist nicht "diese Quelle hat
    # keine Bilder". Wo Bilder geplant waren und keines ankam, ist der Abruf
    # kaputt oder der Bildserver weg — ein Datensatz mit leerer Bildspalte sieht
    # aber genauso aus wie einer, der nie Bilder haben sollte, und träte später
    # als stumm leerer Trainingsarm auf.
    if urls and fetched == 0:
        print(f"\n  {len(urls)} Bild-URLs geplant, keine einzige geholt — nichts "
              f"hochgeladen.\n  Entweder ist {source.image_source} nicht erreichbar, "
              f"oder der Bezugsweg stimmt nicht mehr.\n  Mit --no-images wird daraus "
              f"bewusst ein Datensatz ohne Bildspalte.", file=sys.stderr)
        return 4

    from datasets import Dataset, Features, Image, Value

    features = Features({
        "image": Image(decode=False),
        "xml_content": Value("string"),
        "filename": Value("string"),
        "project_name": Value("string"),
        "source_path": Value("string"),
        "source_url": Value("string"),
    })
    dataset = Dataset.from_list(rows, features=features)

    out = args.out or (args.workdir / "_hf" / source.id)
    shard_dir = out / "data" / "train" / source.project
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / "0000.parquet"
    dataset.to_parquet(str(shard))
    (out / "README.md").write_text(
        dataset_card(source, sel, with_images=fetched, commit=commit),
        encoding="utf-8")
    print(f"  schrieb {shard} ({shard.stat().st_size / 1e6:.1f} MB)")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(source.target, repo_type="dataset",
                    private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=source.target, repo_type="dataset",
                      folder_path=str(out),
                      commit_message=f"{len(rows)} Seiten, {sel.lines} Zeilen aus "
                                     f"{source.origin}@{commit[:12]} "
                                     f"({source.licence})")
    print(f"  hochgeladen: {source.target} (privat)")
    print(f"\n  Training:  \"hf_repo\": \"{source.target}\", "
          f"\"train_projects\": [\"{source.project}\"], \"granularity\": \"line\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
