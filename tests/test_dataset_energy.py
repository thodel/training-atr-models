"""Materialisation energy for a dataset card (#331, point 4): read from job records."""
from __future__ import annotations

import json

import pytest

from atr_training.dataset_energy import (
    MARK_CLOSE,
    MARK_OPEN,
    dataset_energy,
    load_jobs,
    markdown_section,
    splice,
)

KURRENT = "dh-unibe/image-text_kurrent-xix"
ZH = "dh-unibe/image-text_zh-regierungsratsprotokolle"


def stage(name, started, finished, model=None):
    return {"name": name, "status": "completed", "started_at": started, "finished_at": finished,
            "energy_cpu_modelled": model}


def job(job_id, *, host, counts, stages, artefact="abc123abc123 built by this job",
        workers=8, started="2026-10-12T13:57:00+00:00"):
    return {"id": job_id, "host": host, "started_at": started,
            "request": {"params": {"workers": workers}},
            "progress": {"artefact": artefact,
                         "dataset_counts": [{"hf_repo": r, "pages_written": p} for r, p in counts]},
            "stages": stages, "_store": "share"}


RECORDED = {"cpu_wh": 100.0, "ram_wh": 10.0, "cpus": 8, "w_per_core": 3.516, "cpu_model": "EPYC"}


def builder_recorded():
    return job("J-rec", host="gnode03", counts=[(KURRENT, 300), (ZH, 100)],
               stages=[stage("prepare", "2026-10-30T10:00:00+00:00", "2026-10-30T12:00:00+00:00", RECORDED),
                       stage("compile", "2026-10-30T12:00:00+00:00", "2026-10-30T12:30:00+00:00",
                             {**RECORDED, "cpu_wh": 20.0, "ram_wh": 0.0})],
               started="2026-10-30T10:00:00+00:00")


def builder_old():
    # before #331: no model on the stage, 2 h prepare + 1 h compile on asteraix, 8 workers
    return job("J-old", host="asteraix", counts=[(KURRENT, 200)],
               stages=[stage("prepare", "2026-10-12T14:00:00+00:00", "2026-10-12T16:00:00+00:00"),
                       stage("compile", "2026-10-12T16:00:00+00:00", "2026-10-12T17:00:00+00:00")])


def adopter(job_id, builder):
    return job(job_id, host="asteraix", counts=[(KURRENT, 200)], artefact=f"abc123abc123 reused, built by {builder}",
               stages=[stage("train", "2026-10-13T10:00:00+00:00", "2026-10-13T13:00:00+00:00")],
               started="2026-10-13T10:00:00+00:00")


def test_a_recorded_model_is_apportioned_by_pages():
    e = dataset_energy([builder_recorded()], KURRENT)
    (row,) = e.rows
    assert row.recorded and row.stages == ["prepare", "compile"]
    assert row.share == pytest.approx(0.75)
    # (100 + 10 + 20 + 0) Wh × 0.75
    assert row.wh_attributed == pytest.approx(97.5)
    assert row.hours == pytest.approx(2.5)
    # the other dataset gets the rest
    assert dataset_energy([builder_recorded()], ZH).rows[0].wh_attributed == pytest.approx(32.5)


def test_an_old_record_is_modelled_after_the_fact_from_wall_time_and_workers():
    e = dataset_energy([builder_old()], KURRENT)
    (row,) = e.rows
    assert not row.recorded
    assert row.w_per_core == pytest.approx(280 / 24)      # Threadripper on asteraix
    assert row.cpus == 8
    assert row.cpu_wh == pytest.approx(8 * 280 / 24 * 3, abs=0.1)
    assert row.wh_attributed == pytest.approx(row.cpu_wh, abs=0.1)   # its only dataset


def test_an_unknown_box_stays_unmodelled_and_out_of_the_sum():
    old = builder_old()
    old["host"] = "laptop"
    e = dataset_energy([old, builder_recorded()], KURRENT)
    assert [r.wh_attributed is None for r in e.rows] == [True, False]
    assert e.rows[0].why_not == "processor of laptop not in the TDP table"
    assert e.modelled_wh == pytest.approx(97.5)
    assert len(e.unmodelled) == 1


def test_a_record_without_a_host_takes_the_stores_legacy_host():
    """The share held idhefix's jobs before asteraix stamped its own; idhefix is not in the table."""
    old = builder_old()
    old["host"] = None
    (row,) = dataset_energy([old], KURRENT).rows
    assert row.host == "idhefix" and row.host_assumed
    assert row.wh_attributed is None and "idhefix" in row.why_not
    scratch = builder_old()
    scratch["host"], scratch["_store"] = None, "scratch"
    (row,) = dataset_energy([scratch], KURRENT).rows
    assert row.host == "ubelix" and row.w_per_core == pytest.approx(225 / 64)
    assert "ubelix (by store)" in markdown_section(dataset_energy([scratch], KURRENT), generated="x")


def test_a_stage_without_times_is_not_zero_watt_hours():
    old = builder_old()
    old["stages"][0]["finished_at"] = None
    (row,) = dataset_energy([old], KURRENT).rows
    assert row.wh_attributed is None and row.why_not == "no times on the prepare stage"


def test_adopters_count_as_reuses_of_their_builder_and_build_nothing():
    jobs = [builder_old(), adopter("J-a", "J-old"), adopter("J-b", "J-old"), adopter("J-c", "J-rec")]
    e = dataset_energy(jobs, KURRENT)
    assert [r.job_id for r in e.rows] == ["J-old"]
    assert e.rows[0].reused_by == ["J-a", "J-b"]
    assert e.reuses == 2


def test_a_job_that_never_touched_the_dataset_is_absent():
    assert dataset_energy([builder_old()], ZH).rows == []


def test_rows_are_in_order_of_building():
    e = dataset_energy([builder_recorded(), builder_old()], KURRENT)
    assert [r.job_id for r in e.rows] == ["J-old", "J-rec"]


def test_the_section_names_the_basis_and_the_total():
    md = markdown_section(dataset_energy([builder_old(), builder_recorded(), adopter("J-a", "J-old")], KURRENT),
                          generated="2026-11-01")
    assert md.startswith(MARK_OPEN) and md.endswith(MARK_CLOSE)
    assert "recorded by the stage" in md and "after the fact: 8 cores × 11.7 W" in md
    assert "modelled, not measured" in md
    assert "reused 1 time" in md and "| 1 |" in md
    assert "2026-10-12 | `J-old` | asteraix | 200 | 100 % | 3.00 |" in md


def test_splice_appends_once_and_then_replaces():
    readme = "---\nlicense: mit\n---\n\n# Dataset Card for x\n\nSummary.\n"
    first = splice(readme, f"{MARK_OPEN}\nv1\n{MARK_CLOSE}")
    assert first.startswith("---\nlicense: mit\n---\n\n# Dataset Card for x")
    assert first.count(MARK_OPEN) == 1 and "v1" in first
    second = splice(first, f"{MARK_OPEN}\nv2\n{MARK_CLOSE}")
    assert "v1" not in second and "v2" in second and second.count(MARK_OPEN) == 1
    assert second.split(MARK_OPEN)[0] == first.split(MARK_OPEN)[0], "nothing above the section moved"


def test_load_jobs_reads_a_store_and_tags_the_rows(tmp_path):
    (tmp_path / "J-1").mkdir()
    (tmp_path / "J-1" / "job.json").write_text(json.dumps(builder_old()))
    (tmp_path / "J-2").mkdir()
    (tmp_path / "J-2" / "job.json").write_text("not json")
    jobs = load_jobs(tmp_path, store="scratch")
    assert [j["id"] for j in jobs] == ["J-old"] and jobs[0]["_store"] == "scratch"
    assert load_jobs(tmp_path / "missing", store="x") == []
