"""What is new in training material and base models, in one file (#112, D3).

Two discoveries run apart. The dataset half is this repository's daily timer on
asteraix (``scripts/discover_datasets.py``, #32), which keeps its memory on that
box's local disk. The model half is serving-atr-inference's weekly Action, whose
catalogue lives in that repository (``config/discovered.yaml``). Neither reached
anyone: the dataset reports sat in ``~/atr-cache/discovery/reports`` and the
model report pointed at an issue closed since September (serving#112).

This module puts both halves in one small record, written to the research share
by the timer, where the results MCP (``atr_results_mcp``) can read it with the
standard library alone and the daily report can say what is new — or that
nothing is, in one line. The judgement about what counts as news stays with the
two producers; this is the delivery, not a second opinion.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

__all__ = ["CATALOGUE_SOURCES", "FILENAME", "WINDOW_DAYS", "build", "dataset_half",
           "fetch_catalogue", "model_half", "parse_report", "publish"]

ZURICH = ZoneInfo("Europe/Zurich")
#: What the share holds under the publish directory.
FILENAME = "discoveries.json"
#: How far back "new" reaches. The daily report reads it every morning; a week
#: is long enough that a weekend without a report loses nothing.
WINDOW_DAYS = 7
#: Usable model arrivals listed by name; the rest are counted.
SHOWN = 10

_RAW = "https://raw.githubusercontent.com/thodel/serving-atr-inference/{ref}/config/discovered.yaml"
#: Where the model catalogue is read, freshest first. The weekly Action proposes
#: the catalogue on ``discovery/catalogue`` and it reaches ``main`` only when that
#: PR is merged, so the branch is what the last run found and ``main`` is what was
#: last reviewed. The branch is gone between a merge and the next run.
CATALOGUE_SOURCES = (("discovery/catalogue", _RAW.format(ref="discovery/catalogue")),
                     ("main", _RAW.format(ref="main")))
#: Machine verdicts are news only in a count, never by name (serving#112).
AUTOMATIC_BY = "discover_models.py"

#: The judgements a summary counts, by the words that name them in
#: ``atr_serving.base_models``.
WHY_NOT = (
    ("nothing identifies a backend", "nichts weist auf ein Backend hin"),
    ("vision tower but no chat template", "Vision-Tower ohne Chat-Template"),
    ("an archive", "Archiv, könnte ein .mlmodel enthalten"),
    ("weights but no .mlmodel", "Gewichte, aber kein .mlmodel"),
    ("parameter count", "Parameterzahl nicht veröffentlicht"),
    ("no public files", "keine öffentlichen Dateien"),
)

_HEADING = re.compile(r"^## `([^`]+)`\s*$")
_KIND = re.compile(r"^- \*\*([a-z]+)\*\* — (.*)$")
_PLAN = re.compile(r"\b(would (?:enter|leave|move)[^;]*)", re.IGNORECASE)


def _now() -> dt.datetime:
    return dt.datetime.now(ZURICH)


def _label(stamp: dt.datetime) -> str:
    return stamp.astimezone(ZURICH).strftime("%Y-%m-%d %H:%M %Z")


# ── the dataset half ────────────────────────────────────────────────────────
def parse_report(text: str) -> list[dict]:
    """The datasets a ``reports/YYYY-MM-DD.md`` names, with what changed in each.

    The report is :func:`atr_training.discovery.render`'s, so its shape is ours:
    ``## `name``` per dataset, ``- **kind** — what`` per change, and the plan
    consequence on an indented line below it.
    """
    out: list[dict] = []
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            out.append({"dataset": heading.group(1), "changes": [], "plan": []})
            continue
        if not out:
            continue
        kind = _KIND.match(line)
        if kind:
            out[-1]["changes"].append({"kind": kind.group(1), "what": kind.group(2)[:160]})
            continue
        plan = _PLAN.search(line)
        if plan and line.startswith("  "):
            out[-1]["plan"].append(plan.group(1).strip()[:120])
    return out


def dataset_half(state_dir: Path, today: dt.date, days: int = WINDOW_DAYS) -> dict:
    """The last run and the reports of the window, from the timer's own memory."""
    state = Path(state_dir)
    last_run = None
    runs = state / "runs.jsonl"
    if runs.is_file():
        lines = [line for line in runs.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            try:
                last_run = json.loads(lines[-1])
            except ValueError:
                last_run = {"error": "the last line of runs.jsonl is not JSON"}
    since = today - dt.timedelta(days=days)
    reports = []
    for path in sorted((state / "reports").glob("*.md")):
        try:
            day = dt.date.fromisoformat(path.stem)
        except ValueError:
            continue
        if day >= since:
            reports.append({"date": path.stem,
                            "datasets": parse_report(path.read_text(encoding="utf-8"))})
    return {"last_run": last_run, "reports": reports,
            "changed": sorted({d["dataset"] for r in reports for d in r["datasets"]})}


# ── the model half ──────────────────────────────────────────────────────────
def _why_not(judgement: str | None) -> str:
    text = judgement or ""
    for needle, label in WHY_NOT:
        if needle in text:
            return label
    return "noch nicht beurteilt" if not text else text[:60]


def fetch_catalogue(get: Callable[[str], str] | None = None) -> tuple[str | None, str | None, str | None]:
    """``(text, source, error)``: the catalogue from the first source that has it.

    A 404 on the branch is the ordinary state between a merge and the next run,
    not an error. Anything else is reported, and the next source is tried.
    """
    get = get or _http_get
    errors = []
    for source, url in CATALOGUE_SOURCES:
        try:
            return get(url), source, ("; ".join(errors) or None)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                errors.append(f"{source}: HTTP {exc.code}")
        except (OSError, ValueError) as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    return None, None, "; ".join(errors) or "no source had the catalogue"


def _http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 — fixed https URLs
        return response.read().decode("utf-8")


def model_half(text: str | None, source: str | None, today: dt.date,
               days: int = WINDOW_DAYS, error: str | None = None) -> dict:
    """New usable candidates by name, everything else as counts."""
    if text is None:
        return {"error": error or "the catalogue could not be read", "source": None}
    import yaml  # noqa: PLC0415 — the timer's venv has it; the probe never imports this

    candidates = (yaml.safe_load(text) or {}).get("candidates") or []
    since = (today - dt.timedelta(days=days)).isoformat()
    counts = Counter(c.get("verdict", "new") for c in candidates)
    fresh = [c for c in candidates if str(c.get("first_seen", "")) >= since]
    usable = sorted((c for c in fresh if c.get("verdict") == "new" and c.get("backend")),
                    key=lambda c: -(c.get("downloads") or 0))
    other = Counter(_why_not(c.get("judgement")) for c in fresh
                    if c.get("verdict") == "new" and not c.get("backend"))
    decided = [c for c in candidates if c.get("verdict") != "new" and c.get("by") != AUTOMATIC_BY
               and str(c.get("at") or "") >= since]
    return {
        "source": source,
        "error": error,
        "as_of": max((str(c.get("last_seen", "")) for c in candidates), default=None),
        "counts": {v: counts.get(v, 0) for v in ("new", "watching", "rejected", "adopted")},
        "new_usable": [{"key": f"{c.get('source')}:{c.get('id')}", "backend": c.get("backend"),
                        "judgement": c.get("judgement"), "downloads": c.get("downloads"),
                        "first_seen": c.get("first_seen")} for c in usable[:SHOWN]],
        "new_usable_more": max(0, len(usable) - SHOWN),
        "new_without_backend": dict(other.most_common()),
        "rejected_automatically": sum(1 for c in fresh if c.get("by") == AUTOMATIC_BY),
        "decided": [{"key": f"{c.get('source')}:{c.get('id')}", "verdict": c.get("verdict"),
                     "by": c.get("by"), "at": c.get("at"),
                     "reason": (c.get("reason") or "")[:80] or None} for c in decided],
        "waiting_usable": sum(1 for c in candidates if c.get("verdict") == "new"
                              and c.get("backend") and str(c.get("first_seen", "")) < since),
    }


# ── the record ──────────────────────────────────────────────────────────────
def _summary(datasets: dict, models: dict, days: int) -> str:
    parts = []
    changed = datasets.get("changed") or []
    if changed:
        parts.append(f"Trainingsmaterial: {len(changed)} Datensatz/-sätze geändert "
                     f"({', '.join(changed[:4])}{' …' if len(changed) > 4 else ''})")
    if models.get("error") and models.get("source") is None:
        parts.append(f"Modelle: Katalog nicht lesbar ({models['error']})")
    else:
        usable = len(models.get("new_usable") or []) + (models.get("new_usable_more") or 0)
        decided = len(models.get("decided") or [])
        if usable or decided:
            parts.append(f"Modelle: {usable} neue als Basis verwendbar"
                         + (f", {decided} Urteil(e)" if decided else ""))
    if not parts:
        return (f"Nichts Neues in {days} Tagen: kein Datensatz geändert, kein als Basis "
                "verwendbares Modell aufgetaucht.")
    return " · ".join(parts)


def build(state_dir: Path, *, today: dt.date | None = None, days: int = WINDOW_DAYS,
          get: Callable[[str], str] | None = None) -> dict:
    now = _now()
    today = today or now.date()
    datasets = dataset_half(state_dir, today, days)
    text, source, error = fetch_catalogue(get)
    try:
        models = model_half(text, source, today, days, error)
    except Exception as exc:  # noqa: BLE001 — a broken catalogue must not cost the dataset half
        models = {"error": f"{type(exc).__name__}: {exc}", "source": source}
    return {"generated_at": _label(now), "window_days": days,
            "summary": _summary(datasets, models, days),
            "datasets": datasets, "models": models}


def _write(path: Path, text: str) -> None:
    """Tmp file and ``os.replace`` in the same directory: atomic on the CIFS share,
    where ``copy2`` fails on the mode bits (docs/INFRASTRUCTURE.md)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(text)
    os.replace(tmp, path)


def publish(state_dir: Path, publish_dir: Path, record: dict) -> list[Path]:
    """Write the record and mirror the timer's run log and reports to the share.

    The local state stays the memory; the share copy is for readers on other
    hosts. Reports are small and few, so every one is mirrored, not only today's.
    """
    state, target = Path(state_dir), Path(publish_dir)
    written = [target / FILENAME]
    _write(target / FILENAME, json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    runs = state / "runs.jsonl"
    if runs.is_file():
        _write(target / "runs.jsonl", runs.read_text(encoding="utf-8"))
        written.append(target / "runs.jsonl")
    for report in sorted((state / "reports").glob("*.md")):
        copy = target / "reports" / report.name
        text = report.read_text(encoding="utf-8")
        if not copy.is_file() or copy.read_text(encoding="utf-8") != text:
            _write(copy, text)
            written.append(copy)
    return written
