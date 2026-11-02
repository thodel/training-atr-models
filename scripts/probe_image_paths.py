#!/usr/bin/env python3
"""Je Quelle fünf Bild-URLs abrufen und melden, was sich verändert hat (#199).

    .venvs/kraken-train/bin/python scripts/probe_image_paths.py            # alle Quellen
    .venvs/kraken-train/bin/python scripts/probe_image_paths.py --dry-run  # nichts schreiben
    .venvs/kraken-train/bin/python scripts/probe_image_paths.py dach-gt    # eine Quelle

Läuft wöchentlich aus ``deploy/systemd/atr-image-probe.timer`` auf asteraix, auf
den Klonen, die auch der Import benutzt (``~/gt-src``). Der Lauf hängt bewusst
**nicht** am Import: der läuft, wenn jemand eine Quelle einliest, und dann ist es
für eine gealterte URL zu spät.

Zustand in ``--state-dir`` (Vorgabe ``~/atr-cache/image-probe``):

    snapshot.json          die Lage jeder geprüften URL beim letzten Lauf
    reports/YYYY-MM-DD.md  nur, wenn sich etwas verändert hat
    runs.jsonl             eine Zeile je Lauf, mit oder ohne Veränderung

Der erste Lauf berichtet die Quellen, die nicht vollständig erreichbar sind —
eine schon tote Quelle ist genau der Fall, für den es diesen Lauf gibt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atr_training import image_probe  # noqa: E402
from atr_training.foreign_gt import (  # noqa: E402
    SOURCES,
    Source,
    deduplicate,
    image_urls,
    tracked_pages,
)
from atr_training.foreign_gt_zenodo import ZENODO_SOURCES, ZenodoRecord  # noqa: E402

#: Zwischen zwei Abrufen. Fünf je Quelle an öffentliche Bibliotheksserver.
DEFAULT_DELAY = 1.0


def urls_of(source: Source | ZenodoRecord, workdir: Path) -> dict[str, str]:
    """``{Seite: Bild-URL}`` einer Quelle, auf dem Klon, den der Import benutzt.

    Bei einer Zenodo-Hinterlegung (#198) liegen die Bilder in den Archiven; geprüft
    werden deren Datei-URLs: ein verschwundenes Archiv ist ein verschwundener Bildweg.
    """
    if isinstance(source, ZenodoRecord):
        keys = dict.fromkeys(k for p in source.parts for k in p.files)
        return {key: source.file_url(key) for key in keys}
    from foreign_gt_to_hf import clone  # noqa: PLC0415 — das Skript nebenan

    root = clone(source, workdir)
    sel = tracked_pages(root, source.xml_root, source.alto)
    deduplicate(sel, source.prefer, source.variants)
    return image_urls(root, source, sel.pages) if source.images else {}


def measure(sources, workdir: Path, *, delay: float, timeout: float,
            list_urls=urls_of, probe=image_probe.probe) -> image_probe.Snapshot:
    snap = image_probe.Snapshot(taken_at=image_probe.Snapshot.now())
    for source in sources:
        try:
            urls = list_urls(source, workdir)
        except Exception as exc:  # noqa: BLE001 — eine kaputte Quelle hält die anderen nicht auf
            print(f"  {source.id}: nicht auflistbar ({exc})", file=sys.stderr)
            snap.sources[source.id] = {"(Auflistung)": image_probe.LISTING_FAILED}
            continue
        if not urls:
            print(f"  {source.id}: kein Bildweg im Register — nichts zu prüfen")
            continue
        states = {}
        for _, url in image_probe.sample(urls):
            states[url] = probe(url, timeout=timeout)
            time.sleep(delay)
        snap.sources[source.id] = states
        print(f"  {source.id}: {dict(Counter(states.values()))}", flush=True)
    return snap


def main(argv: list[str] | None = None, *, list_urls=urls_of,
         probe=image_probe.probe) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source_ids", nargs="*", help="nur diese Quellen (Vorgabe: alle)")
    p.add_argument("--workdir", type=Path, default=Path.home() / "gt-src")
    p.add_argument("--state-dir", type=Path,
                   default=Path.home() / "atr-cache" / "image-probe")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--dry-run", action="store_true", help="Bericht zeigen, nichts merken")
    args = p.parse_args(argv)

    register = (*SOURCES, *ZENODO_SOURCES)
    sources = [s for s in register if not args.source_ids or s.id in args.source_ids]
    unknown = set(args.source_ids) - {s.id for s in register}
    if unknown:
        print(f"unbekannte Quelle(n): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    state = args.state_dir
    snapshot_path = state / "snapshot.json"
    previous = (image_probe.Snapshot.loads(snapshot_path.read_text(encoding="utf-8"))
                if snapshot_path.is_file() else None)
    current = measure(sources, args.workdir, delay=args.delay, timeout=args.timeout,
                      list_urls=list_urls, probe=probe)
    if args.source_ids and previous is not None:
        # Ein Teillauf vergleicht nur, was er gemessen hat, und behält den Rest.
        previous = image_probe.Snapshot(
            previous.taken_at,
            {k: v for k, v in previous.sources.items() if k in current.sources})

    changes = image_probe.diff(previous, current)
    report = image_probe.render(changes, previous, current)
    if report:
        print(report)
    else:
        print(f"image-probe: keine Veränderung seit "
              f"{previous.taken_at if previous else 'dem ersten Lauf'}")
    if args.dry_run:
        return 0
    if report:
        _write(state / "reports" / f"{time.strftime('%Y-%m-%d')}.md", report)
    merged = image_probe.Snapshot.loads(snapshot_path.read_text(encoding="utf-8")) \
        if snapshot_path.is_file() else image_probe.Snapshot(current.taken_at)
    merged.taken_at = current.taken_at
    merged.sources.update(current.sources)
    _write(snapshot_path, merged.dumps())
    state.mkdir(parents=True, exist_ok=True)
    with (state / "runs.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps({
            "at": current.taken_at, "sources": len(current.sources),
            "changes": [c.source_id for c in changes],
            "not_ok": {sid: sum(1 for v in st.values() if v != image_probe.OK)
                       for sid, st in current.sources.items()},
        }, ensure_ascii=False) + "\n")
    return 0


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(text)
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
