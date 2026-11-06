#!/usr/bin/env python3
"""Aufräumen als Plan: eine geprüfte Löschliste für ``jobs/*/data`` (#387).

    plan_purge.py [--status failed,cancelled] [--ended-before 2026-10-01] [--write]
    plan_purge.py --verify <plan>
    plan_purge.py --apply <plan> --sha256 <hex>      (nur aus apply_purge.sh)

Folgt aus #204. Die Löschung vom 06.11.2026 lief als Befehlsblöcke zum
Kopieren, und alle drei Fehler kamen aus der Handarbeit: der Pool-Job von
``german-medieval-v1`` stand im Vorschlag, ``sbatch`` wollte ``--nodes``, und
beim Einfügen wurden Zeilen verklebt. Hier wird die Liste erzeugt statt
getippt, und was nie gelöscht werden darf, kommt gar nicht erst hinein.

**Der Plan** (ohne ``--apply``) liest nur. Er nimmt die Jobs eines Jobs-Roots
mit einem der verlangten Status — ``failed,cancelled`` ohne Angabe;
``completed`` nur, wenn es ausdrücklich verlangt ist — und legt geschützte
Jobs mit Grund beiseite:

* jeder ``source_job`` aus ``config/heldout_eval_documents.json``: der Pool
  eines Held-out-Sets hält Split-Listen und Val-Seiten, die es sonst nirgends
  gibt;
* jeder Erbauer eines Artefakts, das ein Job übernommen hat, der selbst nicht
  im Plan steht (``progress.artefact``: ``"<key> reused, built by <job>"``) —
  ``rescore_on_shared_draw.py --heldout`` sucht das Trainingsmanifest dort;
* jeder Job, dessen Verzeichnis in ``registry/trained/*.yaml`` als
  ``local_path`` vorkommt;
* alles, dessen ``data/`` aufgelöst unter ``eval_sets/`` oder ``archive/``
  liegt oder den Jobs-Root verlässt.

Die Grösse je Eintrag kommt aus dem letzten Inventar
(``storage_inventory.sbatch``, ``jobs_data``), nicht aus einem neuen ``du``.
Mit ``--write`` wird der Plan nach ``~/ubelix/plans/purge-<zeit>.json``
geschrieben, samt sha256 über die Einträge; ``--json`` gibt ihn auf stdout.

**Die Ausführung** (``--apply``, nur im Slurm-Job von ``apply_purge.sh``)
prüft vor jedem Eintrag erneut: die Datei hat noch die sha256 von der
Einreichung, der Status ist der aus dem Plan, der Job ist nicht geschützt. Nur
dann wird ``data/`` gelöscht; ``job.json``, ``logs/`` und die Energie-Dateien
bleiben. Je Eintrag eine Zeile, ``removed`` oder ``skip`` mit Grund.

Python 3.9 und nur die Standardbibliothek: das läuft auf dem Login-Knoten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import socket
import sys
from pathlib import Path

STORAGE = Path(os.environ.get("ATR_RESEARCH_STORAGE", "/storage/research/wbkolleg_dh_1"))
OURS = STORAGE / "Textrecognition_Training"
JOBS_ROOT = OURS / "training_folder" / "jobs"
REGISTRY_ROOT = OURS / "registry"
REPO = Path(__file__).resolve().parents[1]
HELDOUT_REGISTRY = REPO / "config" / "heldout_eval_documents.json"
INVENTORY = Path(os.environ.get(
    "ATR_STORAGE_INVENTORY", Path.home() / "ubelix" / "research-storage-inventory.json"))
PLANS = Path.home() / "ubelix" / "plans"

KIND = "atr-purge-plan"
VERSION = 1
#: Wonach ``--status`` fragen darf. Ein laufender Job hat kein ``data/`` zum
#: Wegwerfen, nur eines, das er gerade schreibt.
TERMINAL = ("failed", "cancelled", "completed")
DEFAULT_STATUS = ("failed", "cancelled")
#: Verzeichnisnamen, unter denen nie gelöscht wird, wohin ``data/`` auch zeigt.
SHELTERED = ("eval_sets", "archive")
#: ``progress.artefact`` eines Jobs, der den Korpus eines anderen übernahm.
ADOPTED_RE = re.compile(r"^(?P<key>[0-9a-f]{6,64}) reused, built by (?P<builder>\S+)$")
LOCAL_PATH_RE = re.compile(r"^\s*local_path:\s*[\"']?([^\"'#]+?)[\"']?\s*(#.*)?$")
DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


class PlanError(Exception):
    """Ohne diese Angabe gibt es keinen Plan, statt eines unvollständig geprüften."""


# ── Lesen ────────────────────────────────────────────────────────────────────

def read_jobs(jobs_root: Path) -> dict:
    """``{job_id: job.json}`` für jedes Verzeichnis mit lesbarem ``job.json``."""
    out = {}
    for directory in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
        try:
            raw = json.loads((directory / "job.json").read_text())
        except (OSError, ValueError):
            continue
        if isinstance(raw, dict):
            out[directory.name] = raw
    return out


def ended(record: dict) -> str | None:
    value = record.get("finished_at") or record.get("updated_at")
    return value if isinstance(value, str) else None


def heldout_sources(path: Path) -> dict:
    """``{source_job: [set, ...]}`` aus der Held-out-Registry."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise PlanError(f"Held-out-Registry {path} nicht lesbar: {exc}") from exc
    out: dict = {}
    for entry in raw.get("sets", []):
        job = entry.get("source_job")
        if job:
            out.setdefault(job, []).append(entry.get("name", "?"))
    return out


def registered_paths(registry_root: Path) -> dict:
    """``{local_path: datei}`` aus ``registry/trained/*.yaml``.

    Zeilenweise statt mit YAML: der Login-Knoten hat kein PyYAML, und
    ``local_path`` ist ein Skalar auf einer Zeile (``registration.py`` schreibt
    ihn so).
    """
    trained = registry_root / "trained"
    if not trained.is_dir():
        raise PlanError(f"{trained} fehlt — ohne die Registry ist nicht prüfbar, "
                        "welche Jobs ein registriertes Modell halten (--registry-root)")
    out = {}
    for path in sorted(trained.glob("*.yaml")):
        try:
            text = path.read_text()
        except OSError as exc:
            raise PlanError(f"{path} nicht lesbar: {exc}") from exc
        for line in text.splitlines():
            match = LOCAL_PATH_RE.match(line)
            if match:
                out[match.group(1).strip()] = path.name
    return out


def adopted(record: dict) -> str | None:
    """Der Erbauer, dessen Artefakt dieser Job übernahm, sonst ``None``."""
    progress = record.get("progress")
    note = progress.get("artefact") if isinstance(progress, dict) else None
    match = ADOPTED_RE.match(note.strip()) if isinstance(note, str) else None
    return match.group("builder") if match else None


def sizes_from_inventory(path: Path, jobs_root: Path) -> tuple:
    """``({job_id: bytes}, gemessen um)``, wenn das Inventar diesen Jobs-Root
    gemessen hat; sonst ``({}, None)``. Ein Plan für ``runs/jobs`` auf scratch
    hat keine Grössen, und sagt das."""
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}, None
    measured = Path(record.get("root", "")) / "Textrecognition_Training" / "training_folder" / "jobs"
    try:
        same = measured.resolve() == jobs_root.resolve()
    except OSError:
        same = False
    if not same:
        return {}, None
    when = record.get("measured")
    stamp = (dt.datetime.fromtimestamp(when, dt.timezone.utc).isoformat(timespec="seconds")
             if isinstance(when, (int, float)) else None)
    return dict(record.get("jobs_data") or {}), stamp


# ── Schutz ───────────────────────────────────────────────────────────────────

def sheltered(jobs_root: Path, job_id: str) -> str | None:
    """Ein Grund, wenn ``data/`` dieses Jobs nicht dort liegt, wo es soll."""
    data = jobs_root / job_id / "data"
    try:
        real = data.resolve()
        root = jobs_root.resolve()
    except OSError as exc:
        return f"data/ nicht auflösbar: {exc}"
    hit = [part for part in real.parts if part in SHELTERED]
    if hit:
        return f"data/ liegt unter {hit[0]}/ ({real})"
    if real.parent != root / job_id:
        return f"data/ zeigt aus dem Job hinaus ({real})"
    return None


def protections(jobs_root: Path, jobs: dict, candidates: set, *, heldout: dict,
                registered: dict) -> dict:
    """``{job_id: [grund, ...]}`` für jeden Kandidaten, der nicht weg darf.

    Erbauer werden bis zum Fixpunkt geschützt: ein Kandidat, der geschützt
    bleibt, kann selbst ein Artefakt übernommen haben, und dann braucht es
    dessen Erbauer weiter.
    """
    why: dict = {}
    for job_id in candidates:
        reasons = []
        if job_id in heldout:
            reasons.append("source_job von " + ", ".join(sorted(heldout[job_id]))
                           + " (config/heldout_eval_documents.json)")
        for local, source in sorted(registered.items()):
            if job_id in Path(local).parts:
                reasons.append(f"local_path in registry/trained/{source}: {local}")
        shelter = sheltered(jobs_root, job_id)
        if shelter:
            reasons.append(shelter)
        if reasons:
            why[job_id] = reasons
    while True:
        keep = set(jobs) - (candidates - set(why))
        changed = False
        for adopter in sorted(keep):
            builder = adopted(jobs[adopter])
            if builder in candidates:
                note = f"Erbauer des Artefakts, das {adopter} übernahm"
                reasons = why.setdefault(builder, [])
                if note not in reasons:
                    reasons.append(note)
                    changed = True
        if not changed:
            return why


# ── Der Plan ─────────────────────────────────────────────────────────────────

def entries_digest(entries: list) -> str:
    canon = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def plan(jobs_root: Path, *, status: tuple = DEFAULT_STATUS, ended_before: str | None = None,
         heldout_registry: Path = HELDOUT_REGISTRY, registry_root: Path = REGISTRY_ROOT,
         inventory: Path = INVENTORY, now: dt.datetime | None = None) -> dict:
    if not jobs_root.is_dir():
        raise PlanError(f"kein Jobs-Root unter {jobs_root}")
    hit = [part for part in jobs_root.resolve().parts if part in SHELTERED]
    if hit:
        raise PlanError(f"{jobs_root} liegt unter {hit[0]}/ — dort wird nicht aufgeräumt")
    bad = [s for s in status if s not in TERMINAL]
    if bad or not status:
        raise PlanError(f"--status nimmt nur {', '.join(TERMINAL)}, nicht {', '.join(bad) or '(leer)'}")
    if ended_before is not None and not DATE_RE.match(ended_before):
        raise PlanError(f"--ended-before will YYYY-MM-DD, nicht {ended_before!r}")
    heldout = heldout_sources(heldout_registry)
    registered = registered_paths(registry_root)
    jobs = read_jobs(jobs_root)
    sizes, measured = sizes_from_inventory(inventory, jobs_root)

    candidates = set()
    for job_id, record in jobs.items():
        if record.get("status") not in status:
            continue
        end = ended(record)
        if ended_before is not None and (end is None or end[:10] >= ended_before):
            continue
        if not (jobs_root / job_id / "data").is_dir():
            continue
        candidates.add(job_id)
    why = protections(jobs_root, jobs, candidates, heldout=heldout, registered=registered)

    entries = []
    for job_id in sorted(candidates - set(why)):
        record = jobs[job_id]
        entries.append({"id": job_id, "status": record["status"], "ended": ended(record),
                        "path": str(jobs_root / job_id / "data"), "bytes": sizes.get(job_id)})
    protected = [{"id": job_id, "status": jobs[job_id].get("status"), "why": why[job_id]}
                 for job_id in sorted(why)]
    known = [e["bytes"] for e in entries if isinstance(e["bytes"], int)]
    return {
        "kind": KIND, "version": VERSION,
        "created": (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "jobs_root": str(jobs_root), "only": "data",
        "filters": {"status": list(status), "ended_before": ended_before},
        "sources": {"heldout_registry": str(heldout_registry),
                    "registry_root": str(registry_root),
                    "inventory": str(inventory), "inventory_measured": measured},
        "entries": entries,
        "entries_sha256": entries_digest(entries),
        "total_bytes": sum(known),
        "unsized": len(entries) - len(known),
        "protected": protected,
    }


def gb(n: int) -> str:
    return f"{n / 1e9:,.0f} GB".replace(",", " ")


def summary(record: dict, written: Path | None) -> str:
    lines = [f"Plan für {record['jobs_root']} — nur data/",
             f"  Status {','.join(record['filters']['status'])}"
             + (f", beendet vor {record['filters']['ended_before']}"
                if record["filters"]["ended_before"] else ""),
             f"  {len(record['entries'])} Einträge, {gb(record['total_bytes'])}"
             + (f" (+ {record['unsized']} ohne Grösse im Inventar)" if record["unsized"] else "")]
    measured = record["sources"]["inventory_measured"]
    lines.append(f"  Grössen aus dem Inventar vom {measured}" if measured
                 else "  Grössen: kein Inventar dieses Jobs-Roots")
    lines.append(f"  {len(record['protected'])} geschützt:")
    for item in record["protected"]:
        lines.append(f"    {item['id']}  ({item['status']})")
        lines.extend(f"      - {reason}" for reason in item["why"])
    lines.append(f"  sha256 der Einträge {record['entries_sha256']}")
    if written:
        lines.append(f"geschrieben: {written}")
        lines.append(f"ausführen:   ubelix/apply_purge.sh {written}")
    else:
        lines.append("nicht geschrieben (--write schreibt ihn nach ~/ubelix/plans/)")
    return "\n".join(lines)


def write_plan(record: dict, plans: Path) -> Path:
    plans.mkdir(parents=True, exist_ok=True)
    stamp = record["created"].replace("-", "").replace(":", "").replace("+0000", "Z")
    path = plans / f"purge-{stamp[:15]}Z.json"
    if path.exists():
        raise PlanError(f"{path} gibt es schon — ein Plan wird nicht überschrieben")
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: Path) -> dict:
    """Ein Plan, dessen Einträge noch ihre Prüfsumme haben — sonst ``PlanError``."""
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise PlanError(f"{path} ist kein lesbarer Plan: {exc}") from exc
    if not isinstance(record, dict) or record.get("kind") != KIND:
        raise PlanError(f"{path} ist kein {KIND}")
    if record.get("version") != VERSION:
        raise PlanError(f"{path}: Version {record.get('version')!r}, erwartet {VERSION}")
    if entries_digest(record.get("entries", [])) != record.get("entries_sha256"):
        raise PlanError(f"{path}: die Einträge passen nicht zu entries_sha256 — von Hand geändert?")
    return record


# ── Die Ausführung ───────────────────────────────────────────────────────────

def apply(path: Path, expected_sha256: str, out=sys.stdout) -> dict:
    """Löscht ``data/`` je Eintrag, nach erneuter Prüfung. Gibt die Zählung zurück.

    Fail closed: kann eine Prüfung nicht laufen (Plan geändert, Registry nicht
    lesbar), wird nichts gelöscht.
    """
    counts = {"removed": 0, "skip": 0, "bytes": 0}

    def say(line: str) -> None:
        print(line, file=out, flush=True)

    record: dict = {}
    try:
        record = load_plan(path)
        if file_sha256(path) != expected_sha256:
            raise PlanError("die Plan-Datei hat nicht mehr die sha256 von der Einreichung")
        jobs_root = Path(record["jobs_root"])
        sources = record["sources"]
        heldout = heldout_sources(Path(sources["heldout_registry"]))
        registered = registered_paths(Path(sources["registry_root"]))
        jobs = read_jobs(jobs_root)
    except (PlanError, KeyError, OSError) as exc:
        for entry in record.get("entries", []):
            say(f"skip     {entry.get('id')}  {exc}")
            counts["skip"] += 1
        say(f"!! nichts gelöscht: {exc}")
        return counts

    ids = {e["id"] for e in record["entries"]}
    why = protections(jobs_root, jobs, ids, heldout=heldout, registered=registered)
    for entry in record["entries"]:
        job_id = entry["id"]
        reason = None
        if file_sha256(path) != expected_sha256:
            reason = "Plan-Datei seit der Einreichung geändert"
        elif job_id not in jobs:
            reason = "kein lesbares job.json mehr"
        elif jobs[job_id].get("status") != entry["status"]:
            reason = f"Status jetzt {jobs[job_id].get('status')}, im Plan {entry['status']}"
        elif ended(jobs[job_id]) != entry["ended"]:
            reason = "Ende seit dem Plan geändert"
        elif job_id in why:
            reason = "geschützt: " + "; ".join(why[job_id])
        elif str(jobs_root / job_id / "data") != entry["path"]:
            reason = f"Pfad im Plan ist nicht {jobs_root / job_id / 'data'}"
        elif not (jobs_root / job_id / "data").is_dir():
            reason = "data/ gibt es nicht (mehr)"
        if reason:
            say(f"skip     {job_id}  {reason}")
            counts["skip"] += 1
            continue
        data = jobs_root / job_id / "data"
        if data.is_symlink():
            data.unlink()
        else:
            shutil.rmtree(data)
        size = entry.get("bytes")
        counts["removed"] += 1
        counts["bytes"] += size if isinstance(size, int) else 0
        say(f"removed  {job_id}  {gb(size) if isinstance(size, int) else '? GB'}")
    say(f"== {counts['removed']} gelöscht ({gb(counts['bytes'])} laut Inventar), "
        f"{counts['skip']} übersprungen")
    return counts


# ── Kommandozeile ────────────────────────────────────────────────────────────

def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--jobs-root", type=Path, default=JOBS_ROOT)
    ap.add_argument("--status", default=",".join(DEFAULT_STATUS),
                    help="Kommaliste aus failed, cancelled, completed (Vorgabe: failed,cancelled)")
    ap.add_argument("--ended-before", metavar="YYYY-MM-DD")
    ap.add_argument("--only", choices=["data"], default="data",
                    help="was gelöscht wird; job.json und logs/ bleiben immer")
    ap.add_argument("--heldout-registry", type=Path, default=HELDOUT_REGISTRY)
    ap.add_argument("--registry-root", type=Path, default=REGISTRY_ROOT)
    ap.add_argument("--inventory", type=Path, default=INVENTORY)
    ap.add_argument("--plans", type=Path, default=PLANS)
    ap.add_argument("--write", action="store_true", help="Plan nach --plans schreiben")
    ap.add_argument("--json", action="store_true", help="den Plan auf stdout statt der Zusammenfassung")
    ap.add_argument("--verify", type=Path, metavar="PLAN",
                    help="nur prüfen, ob PLAN ein unveränderter Plan ist; gibt seine sha256")
    ap.add_argument("--apply", type=Path, metavar="PLAN", help=argparse.SUPPRESS)
    ap.add_argument("--sha256", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.apply:
        if not args.sha256:
            print("--apply braucht --sha256", file=sys.stderr)
            return 2
        counts = apply(args.apply, args.sha256)
        return 0 if counts["removed"] or not counts["skip"] else 1
    if args.verify:
        try:
            record = load_plan(args.verify)
        except PlanError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        print(f"{file_sha256(args.verify)}  {len(record['entries'])} Einträge, "
              f"{gb(record['total_bytes'])}, {len(record['protected'])} geschützt")
        return 0
    try:
        status = tuple(s.strip() for s in args.status.split(",") if s.strip())
        record = plan(args.jobs_root, status=status, ended_before=args.ended_before,
                      heldout_registry=args.heldout_registry,
                      registry_root=args.registry_root, inventory=args.inventory)
        written = write_plan(record, args.plans) if args.write else None
    except PlanError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2, ensure_ascii=False) if args.json
          else summary(record, written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
