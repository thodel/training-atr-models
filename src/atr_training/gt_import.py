"""Eine registrierte Quelle einlesen, als Job hinter dem Trainer-Dienst (#200, Schritt 2).

    python -m atr_training.gt_import <job-verzeichnis>     # nur vom Dienst gestartet

Der Importweg ist ``scripts/foreign_gt_to_hf.py``. Hier wird er gestartet, nicht
nachgebaut: Lizenzprüfung, Bildweg, die vier Verweigerungen aus #187 und der
private Upload bleiben dort, und ein Dienst, der sie anders machte, wäre eine
zweite Fassung derselben Regeln.

**Was der Dienst darf, ist kleiner als das CLI:**

* nur eine **Kennung aus dem Register** (:data:`foreign_gt.SOURCES`,
  :data:`foreign_gt_zenodo.ZENODO_SOURCES`). Eine unbekannte wird mit der Liste der
  bekannten abgelehnt. Eine neue Quelle kommt per PR ins Register, nach einer
  Erhebung (:mod:`atr_training.gt_survey`) und einem Menschen, der sie gelesen hat.
* **kein** ``--public``. Öffentlich schaltet ein Mensch in der HF-Oberfläche, wenn
  die Bildrechte geklärt sind (#193). Die Kommandozeile, die hier entsteht, kann
  das Flag nicht enthalten; ein Test hält das fest.
* je Quelle ein Import zur selben Zeit.

Ein Import dauert Minuten bis Stunden (``reichsanzeiger-gt``: 101 Bildabrufe und
910 MB Upload), länger als jeder Client wartet. Darum ist er ein Job: ein
Verzeichnis mit ``job.json``, ``import.log`` und am Ende ``result.json``. Der
Runner (dieses Modul mit ``python -m``) startet das Skript, wartet, und schreibt
den Exit-Status. Er läuft in eigener Session, also übersteht ein Import den
Neustart des Dienstes, und der Status ist danach aus den Dateien lesbar.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from atr_training.foreign_gt import SOURCES, ForeignGtError, source_by_id
from atr_training.foreign_gt_zenodo import ZENODO_SOURCES, ZenodoRecord

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "foreign_gt_to_hf.py"
JOB_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-z0-9._-]{1,80}$")
PART_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
LOG_TAIL = 60
#: Zeilen des Importberichts, die in die Antwort gehören: Lizenz, Umfang mit den
#: Verwurfszahlen, Bildweg, und jede Verweigerung.
_REPORT = re.compile(r"(Lizenz:|Bildquelle:|Bildrechte:|Stand:|Teile:|Seiten, .*Zeilen|"
                     r"verworfen|Bild-URLs|Bilder:|verweigert|refused|abgelehnt|"
                     r"hochgeladen|uploaded|--dry-run)", re.IGNORECASE)
#: Exit-Status des Skripts, die eine Verweigerung sind, kein Absturz.
REFUSALS = {2: "unbekannte Quelle", 3: "öffentlich verweigert"}


class GtImportError(ValueError):
    """Ein Import, der nicht gestartet wird — mit dem Grund."""


def all_sources() -> list:
    return [*SOURCES, *ZENODO_SOURCES]


def resolve(source_id: str):
    for z in ZENODO_SOURCES:
        if z.id == source_id:
            return z
    try:
        return source_by_id(source_id)
    except ForeignGtError:
        known = ", ".join(s.id for s in all_sources())
        raise GtImportError(f"unknown source {source_id!r}; the register knows: {known}. "
                            "A new source is added to the register by a PR, after a "
                            "survey (POST /gt/survey).") from None


def describe(source) -> dict:
    """Ein Register-Eintrag, wie ``GET /gt/sources`` ihn zeigt."""
    zen = isinstance(source, ZenodoRecord)
    out = {
        "id": source.id, "kind": "zenodo" if zen else "github",
        "origin": source.record_url if zen else source.origin,
        "licence": source.licence, "licence_at": source.licence_at,
        "image_source": source.image_source, "image_rights": source.image_rights,
        "script_kind": source.script_kind, "period": source.period,
        "target": source.target,
    }
    if zen:
        out["doi"] = source.doi
        out["parts"] = [{"name": p.name, "default": p.default} for p in source.parts]
        out["same_pages_as"] = source.same_pages_as or None
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def argv_for(source, *, dry_run: bool, parts: list[str], all_parts: bool,
             python: str = sys.executable, script: Path = SCRIPT) -> list[str]:
    """Die Kommandozeile des Imports. Enthält nie ``--public``."""
    if (parts or all_parts) and not isinstance(source, ZenodoRecord):
        raise GtImportError(f"{source.id}: parts exist only for Zenodo sources")
    for name in parts:
        if not PART_RE.match(name):
            raise GtImportError(f"not a part name: {name!r}")
        try:
            source.part(name)
        except ForeignGtError as exc:
            raise GtImportError(str(exc)) from None
    argv = [python, str(script), source.id]
    if dry_run:
        argv.append("--dry-run")
    for name in parts:
        argv += ["--part", name]
    if all_parts:
        argv.append("--all-parts")
    assert "--public" not in argv
    return argv


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class GtImports:
    """Die Import-Jobs unter ``root``, je einer in einem Verzeichnis."""

    root: Path
    python: str = sys.executable
    #: None: the module's :data:`SCRIPT`, looked up when a job starts.
    script: Path | None = None

    def _dir(self, job_id: str) -> Path:
        if not JOB_RE.match(job_id):
            raise GtImportError(f"not an import job id: {job_id!r}")
        return self.root / job_id

    def running(self, source_id: str) -> str | None:
        if not self.root.is_dir():
            return None
        for d in sorted(self.root.iterdir(), reverse=True):
            try:
                rec = json.loads((d / "job.json").read_text())
            except (OSError, ValueError):
                continue
            if rec.get("source") == source_id and self.status(d.name)["status"] == "running":
                return d.name
        return None

    def start(self, source_id: str, *, dry_run: bool = True, parts: list[str] | None = None,
              all_parts: bool = False) -> dict:
        source = resolve(source_id)
        argv = argv_for(source, dry_run=dry_run, parts=list(parts or []),
                        all_parts=all_parts, python=self.python,
                        script=self.script or SCRIPT)
        busy = self.running(source.id)
        if busy:
            raise GtImportError(f"{source.id} is already being imported: {busy}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^a-z0-9._-]+", "-", source.id.lower())
        job_id = f"{stamp}-{slug}"
        d = self.root / job_id
        n = 2
        while d.exists():
            job_id = f"{stamp}-{slug}-{n}"
            d, n = self.root / job_id, n + 1
        d.mkdir(parents=True)
        record = {"id": job_id, "source": source.id, "dry_run": dry_run,
                  "parts": list(parts or []), "all_parts": all_parts,
                  "argv": argv, "created_at": _now(), "pid": None}
        (d / "job.json").write_text(json.dumps(record, indent=2))
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            [self.python, "-m", "atr_training.gt_import", str(d)],
            cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
            env={**os.environ, "PYTHONPATH": str(REPO / "src")})
        record["pid"] = proc.pid
        (d / "job.json").write_text(json.dumps(record, indent=2))
        return self.status(job_id)

    def status(self, job_id: str) -> dict:
        d = self._dir(job_id)
        try:
            rec = json.loads((d / "job.json").read_text())
        except FileNotFoundError:
            raise GtImportError(f"no such import job: {job_id}") from None
        result = None
        try:
            result = json.loads((d / "result.json").read_text())
        except (OSError, ValueError):
            pass
        if result is not None:
            code = result.get("exit")
            status = ("completed" if code == 0 else
                      "refused" if code in REFUSALS else "failed")
        elif _pid_alive(rec.get("pid")):
            status = "running"
        else:
            status = "gone"     # der Runner endete, ohne einen Exit-Status zu schreiben
        lines: list[str] = []
        try:
            lines = (d / "import.log").read_text(errors="replace").splitlines()
        except OSError:
            pass
        out = {k: rec.get(k) for k in ("id", "source", "dry_run", "parts", "all_parts",
                                         "created_at")}
        out.update({"status": status, "exit": (result or {}).get("exit"),
                    "finished_at": (result or {}).get("finished_at"),
                    "report": [ln.strip() for ln in lines if _REPORT.search(ln)][:40],
                    "log_tail": lines[-LOG_TAIL:]})
        if status == "refused":
            out["refusal"] = REFUSALS[result["exit"]]
        return out


def run(job_dir: Path) -> int:
    """Der Runner: das Skript laufen lassen, den Exit-Status schreiben."""
    rec = json.loads((job_dir / "job.json").read_text())
    argv = rec["argv"]
    if "--public" in argv:
        code = 3
    else:
        with (job_dir / "import.log").open("w") as log:
            code = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,  # noqa: S603
                                  stdin=subprocess.DEVNULL, cwd=str(REPO)).returncode
    (job_dir / "result.json").write_text(json.dumps({"exit": code, "finished_at": _now()}))
    return code


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1])))
