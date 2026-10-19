"""#256 after the fact: the old ``train_lines`` taken back out, and nothing else.

The fix (``2ebebb8``) corrects ``train_lines`` at reservation time. What was
prepared before still carries the count from before the reservation — and the
copy that matters is the artefact-cache payload, because an adopting job copies
it into its own record and the step-count guard divides by it.

Every test here builds a corpus whose numbers disagree in a known way: a page
with an untranscribed line, so ``transcribed_lines`` and a ``<TextLine`` count
come apart (the trap #256's own tests fell into once).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from atr_training.train_lines_recount import (
    FIX_LANDED,
    apply_to_cache,
    apply_to_job,
    recount_cache_entry,
    recount_job,
    resolve_pages,
)

KEY = "f7188cd0b798" + "0" * 52
BEFORE_FIX = FIX_LANDED.timestamp() - 86_400


def _page(lines: int, empty: int = 1) -> str:
    """PageXML with *lines* transcribed lines and *empty* untranscribed ones."""
    body = "".join(
        f'<TextLine id="l{i}"><Coords points="0,{i*20} 100,{i*20} 100,{i*20+10} 0,{i*20+10}"/>'
        f"<TextEquiv><Unicode>zeile {i}</Unicode></TextEquiv></TextLine>"
        for i in range(lines))
    body += "".join(
        f'<TextLine id="e{i}"><Coords points="0,500 100,500 100,510 0,510"/></TextLine>'
        for i in range(empty))
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
            f'<Page imageFilename="x.jpg" imageWidth="100" imageHeight="600"><TextRegion id="r">'
            f"{body}</TextRegion></Page></PcGts>")


def _corpus(data: Path, *, train: list[int], reserved: list[int],
            absolute_from: Path | None = None) -> None:
    """Pages under ``data/pages``; manifests name them, absolutely if asked.

    *absolute_from* writes the manifests as absolute paths into a directory that
    does not exist — what a cache entry holds once its builder is gone.
    """
    pages = data / "pages"
    pages.mkdir(parents=True)
    names: dict[str, list[str]] = {"train": [], "reserved": []}
    rows = 0
    for kind, sizes in (("train", train), ("reserved", reserved)):
        for index, size in enumerate(sizes):
            name = f"{kind}_{index:04d}.xml"
            (pages / name).write_text(_page(size), encoding="utf-8")
            where = (absolute_from / "data" / "pages" / name) if absolute_from else pages / name
            names[kind].append(str(where))
            if kind == "train":
                rows += size
    (data / "pages_train.lst").write_text("\n".join(names["train"]) + "\n")
    (data / "pages_reserved.lst").write_text("\n".join(names["reserved"]) + "\n")
    (data / "train.jsonl").write_text("".join('{"text": "x"}\n' for _ in range(rows)))


def _entry(root: Path, *, before: int, reserved_pages: int = 2, built_at: float = BEFORE_FIX,
           payload: dict | None = None) -> Path:
    entry = root / KEY
    (entry / "data").mkdir(parents=True)
    meta = {"key": KEY, "built_at": built_at, "job_id": "20261013T043436Z-builder",
            "claims": {}, "payload": {"train_lines": before, "reserved_pages": reserved_pages,
                                      "job_id": "20261013T043436Z-builder", **(payload or {})}}
    (entry / "artefact.json").write_text(json.dumps(meta, indent=2))
    return entry


def _job(root: Path, job_id: str, *, before: int, reserved_pages: int = 2,
         status: str = "completed", artefact: str | None = None) -> Path:
    job = root / job_id
    (job / "data").mkdir(parents=True)
    (job / "logs").mkdir()
    record = {"id": job_id, "status": status, "metrics": {"cer": 0.12},
              "progress": {"train_lines": before, "reserved_pages": reserved_pages,
                           "artefact": artefact, "lines_written": 999}}
    (job / "job.json").write_text(json.dumps(record, indent=2))
    return job


# ── the cache payload, which is what new runs read ──────────────────────────

def test_a_payload_from_before_the_fix_loses_exactly_the_reserved_lines(tmp_path):
    entry = _entry(tmp_path, before=4 + 3 + 5 + 2)
    _corpus(entry / "data", train=[4, 3], reserved=[5, 2])

    recount = recount_cache_entry(entry)

    assert recount.verdict == "correct"
    assert recount.removed == 7, "transcribed lines only — the empty line on each page is not one"
    assert recount.after == 7
    assert recount.train_pages_lines == 7 and recount.train_jsonl_rows == 7
    assert recount.agrees is True


def test_the_independent_counts_are_reported_not_used(tmp_path):
    """A payload from the 0.9 path is an estimate: the correction is still the
    fix's own subtraction, and the disagreement is shown, not resolved."""
    entry = _entry(tmp_path, before=20)
    _corpus(entry / "data", train=[4, 3], reserved=[5, 2])

    recount = recount_cache_entry(entry)

    assert recount.after == 13, "20 − 7, exactly as runner_base would have done"
    assert recount.agrees is False


def test_pages_are_found_by_name_when_the_builder_is_gone(tmp_path):
    entry = _entry(tmp_path, before=14)
    _corpus(entry / "data", train=[4, 3], reserved=[5, 2],
            absolute_from=tmp_path / "rs_scratch" / "jobs" / "builder-gone")

    assert recount_cache_entry(entry).after == 7


def test_one_missing_reserved_page_means_no_correction(tmp_path):
    """A partial subtraction is a number nobody measured."""
    entry = _entry(tmp_path, before=14)
    _corpus(entry / "data", train=[4, 3], reserved=[5, 2])
    next((entry / "data" / "pages").glob("reserved_0001.xml")).unlink()

    recount = recount_cache_entry(entry)

    assert recount.verdict == "skip"
    assert "1 reserved page(s) not found" in recount.why


@pytest.mark.parametrize("why, kwargs", [
    ("nothing was reserved", {"reserved_pages": 0}),
    ("built after the fix", {"built_at": FIX_LANDED.timestamp() + 60}),
    ("recounted on", {"payload": {"train_lines_recount": {"at": "2026-10-19"}}}),
])
def test_a_payload_that_is_already_right_is_left_alone(tmp_path, why, kwargs):
    entry = _entry(tmp_path, before=14, **kwargs)
    _corpus(entry / "data", train=[4, 3], reserved=[5, 2])

    recount = recount_cache_entry(entry)

    assert recount.verdict == "already"
    assert why in recount.why


def test_applying_twice_subtracts_once(tmp_path):
    entry = _entry(tmp_path, before=14)
    _corpus(entry / "data", train=[4, 3], reserved=[5, 2])

    apply_to_cache(entry, recount_cache_entry(entry))
    second = recount_cache_entry(entry)

    payload = json.loads((entry / "artefact.json").read_text())["payload"]
    assert payload["train_lines"] == 7
    assert payload["train_lines_recount"]["before"] == 14
    assert payload["train_lines_recount"]["removed"] == 7
    assert second.verdict == "already"


def test_the_payload_keeps_everything_else(tmp_path):
    entry = _entry(tmp_path, before=14, payload={"lines_written": 999, "dataset_counts": [1]})
    _corpus(entry / "data", train=[4, 3], reserved=[5, 2])
    before = json.loads((entry / "artefact.json").read_text())

    apply_to_cache(entry, recount_cache_entry(entry))

    after = json.loads((entry / "artefact.json").read_text())
    for field in ("key", "built_at", "claims", "job_id"):
        assert after[field] == before[field]
    assert after["payload"]["lines_written"] == 999
    assert after["payload"]["dataset_counts"] == [1]


# ── job records ─────────────────────────────────────────────────────────────

def test_a_builder_is_recounted_from_its_own_pages(tmp_path):
    job = _job(tmp_path, "20261013T043436Z-ladder", before=14,
               artefact="f7188cd0b798 built by this job")
    _corpus(job / "data", train=[4, 3], reserved=[5, 2])

    recount = recount_job(job, {})

    assert (recount.verdict, recount.after, recount.key) == ("correct", 7, "f7188cd0b798")


def test_an_adopter_takes_its_corpus_correction(tmp_path):
    entry = _entry(tmp_path / "cache", before=14)
    _corpus(entry / "data", train=[4, 3], reserved=[5, 2])
    corpus = recount_cache_entry(entry)
    job = _job(tmp_path / "jobs", "20261014T193118Z-ladder-9b", before=14,
               artefact="f7188cd0b798 reused, built by 20261013T043436Z-ladder")

    recount = recount_job(job, {"f7188cd0b798": corpus})

    assert (recount.verdict, recount.after) == ("correct", 7)


def test_an_adopter_prepared_after_the_fix_still_carries_the_old_payload(tmp_path):
    """The case the first dry run missed (19.10.2026): a throughput run of
    17.10. adopted f7188cd0b798 *after* the fix and copied 306,569 out of the
    payload. The fix corrects the reservation step, which an adopter skips — so
    its date says nothing, and the corpus correction applies."""
    entry = _entry(tmp_path / "cache", before=14)
    _corpus(entry / "data", train=[4, 3], reserved=[5, 2])
    job = _job(tmp_path / "jobs", "20261017T114213Z-throughput", before=14,
               artefact="f7188cd0b798 reused, built by 20261013T043436Z-ladder")

    recount = recount_job(job, {"f7188cd0b798": recount_cache_entry(entry)})

    assert (recount.verdict, recount.after) == ("correct", 7)


def test_an_adopter_holding_another_value_is_not_touched(tmp_path):
    """Its number came from somewhere this tool does not know about."""
    entry = _entry(tmp_path / "cache", before=14)
    _corpus(entry / "data", train=[4, 3], reserved=[5, 2])
    job = _job(tmp_path / "jobs", "20261014T193118Z-ladder-9b", before=12,
               artefact="f7188cd0b798 reused, built by 20261013T043436Z-ladder")

    recount = recount_job(job, {"f7188cd0b798": recount_cache_entry(entry)})

    assert recount.verdict == "skip"
    assert "holds 12" in recount.why


@pytest.mark.parametrize("job_id, log, status, why", [
    ("20261017T080000Z-after", None, "completed", "prepared with the fix"),
    ("20261013T043436Z-logged", "train_lines: 14 → 7 after removing 7 held-out line(s) (#256)",
     "completed", "prepared with the fix"),
    ("20261013T043436Z-running", None, "running", "not touched"),
])
def test_a_job_that_corrected_itself_or_still_runs_is_left_alone(tmp_path, job_id, log, status, why):
    job = _job(tmp_path, job_id, before=14, status=status)
    _corpus(job / "data", train=[4, 3], reserved=[5, 2])
    if log:
        (job / "logs" / "prepare.log").write_text(log + "\n")

    recount = recount_job(job, {})

    assert recount.verdict in ("already", "skip")
    assert why in recount.why


def test_a_job_record_changes_in_one_field(tmp_path):
    job = _job(tmp_path, "20261013T043436Z-ladder", before=14)
    _corpus(job / "data", train=[4, 3], reserved=[5, 2])
    before = json.loads((job / "job.json").read_text())

    apply_to_job(job, recount_job(job, {}))

    after = json.loads((job / "job.json").read_text())
    assert after["progress"].pop("train_lines") == 7
    before["progress"].pop("train_lines")
    assert after == before


def test_applying_twice_to_a_job_subtracts_once(tmp_path):
    """The cache payload remembers its recount; a record cannot — the progress
    model drops a field it does not know on the next save — so a sidecar does,
    and it is written before the record: a crash between the two leaves a
    correction missing, never doubled."""
    job = _job(tmp_path, "20261009T150138Z-kurrent", before=14)
    _corpus(job / "data", train=[4, 3], reserved=[5, 2])

    apply_to_job(job, recount_job(job, {}))
    second = recount_job(job, {})

    assert json.loads((job / "job.json").read_text())["progress"]["train_lines"] == 7
    assert second.verdict == "already"
    sidecar = json.loads((job / "train_lines_recount.json").read_text())
    assert (sidecar["before"], sidecar["removed"]) == (14, 7)


def test_a_job_that_started_running_meanwhile_is_refused(tmp_path):
    job = _job(tmp_path, "20261013T043436Z-ladder", before=14)
    _corpus(job / "data", train=[4, 3], reserved=[5, 2])
    recount = recount_job(job, {})
    record = json.loads((job / "job.json").read_text())
    record["status"] = "running"
    (job / "job.json").write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="not touched"):
        apply_to_job(job, recount)
    assert json.loads((job / "job.json").read_text())["progress"]["train_lines"] == 14


def test_resolve_pages_counts_what_it_cannot_find(tmp_path):
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "a.xml").write_text(_page(1))
    manifest = tmp_path / "m.lst"
    manifest.write_text(f"{tmp_path / 'gone' / 'a.xml'}\n{tmp_path / 'gone' / 'b.xml'}\n")

    found, missing = resolve_pages(manifest, tmp_path)

    assert [p.name for p in found] == ["a.xml"]
    assert missing == 1


def test_the_fix_moment_is_the_merge_not_the_commit():
    """#257 merged 5b5ec6b at 21:16:50 CEST on 16.10.2026; a prepare from then on
    ran the correction. The constant is the date the evidence rests on."""
    assert time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(FIX_LANDED.timestamp())) \
        == "2026-10-16 19:16:50"
