"""A job gets a card; two jobs get two cards (#12).

asteraix has two A40s at 46 GB over NVLink, and until now the second one was
unreachable for three reasons the epic names. Two of them had already gone by
the time this was written: `preflight.check_vram(gpu, …)` takes a card index, so
the VRAM check was never global, and the units stopped pinning
`CUDA_VISIBLE_DEVICES` (`deploy/systemd/atr-train.service` says so in as many
words — the service never touches a card). What was left is the third and the
first: every job was given `settings.gpu`, and `max_concurrent` was a flat 1.

The rule the whole file turns on is that a card a running job **holds** is not a
candidate, whatever nvidia-smi says about it. A job between its claim and its
first allocation has nothing resident yet, so asking the card would cheerfully
let a second run onto it — which is the failure of 15.09.2026 one layer up: the
gateway started a model during a run's `prepare`, and the run died three minutes
into `train` with 841 MiB free (serving-atr-inference#129).

The three test names below are the ones #12 asks for.
"""

from pathlib import Path

import pytest

from atr_training.preflight import GpuInfo, PreflightError
from atr_training.settings import TrainerSettings

from tests.test_train_svc_api import (  # noqa: F401 — fixtures
    BODY, app, busy_gpu, client, free_gpu, settings, spawn, store_of, venvs,
)
import engines.kraken_train_svc.app as app_module


# ── the settings ────────────────────────────────────────────────────────────
def test_one_card_is_the_default_and_nothing_changed():
    """The point of `gpus: None`: every deployment keeps exactly what it had, and
    opening the second card is one setting."""
    plain = TrainerSettings()

    assert plain.usable_gpus == (plain.gpu,)
    assert plain.concurrency_limit() == 1
    assert plain.env_for_child()["CUDA_VISIBLE_DEVICES"] == str(plain.gpu)


def test_two_cards_raise_the_limit_without_a_second_setting():
    """#12: "max_concurrent wird eine Funktion der freien Karten"."""
    both = TrainerSettings(gpus=[0, 1])

    assert both.usable_gpus == (0, 1)
    assert both.concurrency_limit() == 2


def test_an_explicit_cap_still_holds():
    """An operator may want one at a time on a two-card box — for a run that
    wants both cards, say."""
    assert TrainerSettings(gpus=[0, 1], max_concurrent=1).concurrency_limit() == 1


def test_the_limit_never_exceeds_the_cards():
    """Two trainings on one card is the OOM everything here guards against, so a
    counter above the card count is a misconfiguration, not a bigger limit."""
    assert TrainerSettings(gpus=[1], max_concurrent=4).concurrency_limit() == 1


def test_the_cards_are_ordered_and_deduplicated():
    """It decides which card a job lands on, and "whichever the set iterated
    first" is not a decision."""
    assert TrainerSettings(gpus=[1, 0, 1]).usable_gpus == (0, 1)


def test_the_child_sees_the_card_it_was_given():
    """And sees it as `cuda:0`, which is what every `device: "cuda:0"` in
    contracts.py means."""
    both = TrainerSettings(gpus=[0, 1])

    assert both.env_for_child(0)["CUDA_VISIBLE_DEVICES"] == "0"
    assert both.env_for_child(1)["CUDA_VISIBLE_DEVICES"] == "1"


def test_a_job_without_a_card_falls_back_to_the_default():
    """A hand-run script, or a record from before cards were allocated."""
    assert TrainerSettings(gpu=1).env_for_child(None)["CUDA_VISIBLE_DEVICES"] == "1"


# ── the three the epic names ────────────────────────────────────────────────
@pytest.fixture
def two_cards(client, settings):  # noqa: F811
    """The client's own settings, with the second card opened.

    Installed on ``app.state`` rather than passed to ``schedule_once`` alone:
    POSTing a job runs a scheduling pass of its own, so a two-card test whose
    submit ran under one-card settings has already put its first job somewhere
    this test did not choose.
    """
    both = settings.model_copy(update={"gpus": [0, 1]})
    client.app.state.settings = both
    return both


def _submit(client, model_id: str) -> str:  # noqa: F811
    response = client.post("/jobs", json={**BODY, "model_id": model_id})
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def test_two_jobs_run_on_two_cards(client, two_cards, spawn):  # noqa: F811
    """The epic's first test. One card each, and neither is the other's."""
    store = store_of(client)
    first, second = _submit(client, "m1"), _submit(client, "m2")

    for _ in range(2):
        app_module.schedule_once(store, two_cards, spawn=spawn, vram_check=free_gpu)

    assert {store.load(first).gpu, store.load(second).gpu} == {0, 1}


def test_the_preflight_checks_the_card_it_will_use(client, two_cards, spawn):  # noqa: F811
    """The epic's second test. Not a box-wide number: the card this job is about
    to be put on is the one that has to have room."""
    asked: list[int] = []

    def only_card_one_is_free(gpu, min_free_mb):
        asked.append(gpu)
        if gpu != 1:
            raise PreflightError(f"GPU {gpu} has 900 MB free, need {min_free_mb} MB")
        return GpuInfo(index=gpu, free_mb=40000, total_mb=46068)

    client.app.state.vram_check = only_card_one_is_free
    store = store_of(client)
    job_id = _submit(client, "m1")

    assert 0 in asked and 1 in asked, asked
    assert store.load(job_id).gpu == 1


def test_a_job_never_lands_on_a_card_another_job_holds(client, two_cards, spawn):  # noqa: F811
    """The epic's third test, and the rule the rest rests on.

    The card is reported free — the first job has claimed it and has nothing
    resident yet. Asking nvidia-smi would put the second job on it. The record
    is what decides.
    """
    store = store_of(client)
    first, second = _submit(client, "m1"), _submit(client, "m2")

    app_module.schedule_once(store, two_cards, spawn=spawn, vram_check=free_gpu)
    held = store.load(first).gpu
    app_module.schedule_once(store, two_cards, spawn=spawn, vram_check=free_gpu)

    assert store.load(second).gpu != held
    assert held is not None


# ── the refusals ────────────────────────────────────────────────────────────
def test_a_third_job_waits_when_both_cards_are_held(client, two_cards, spawn):  # noqa: F811
    store = store_of(client)
    _submit(client, "m1"), _submit(client, "m2")
    third = _submit(client, "m3")

    for _ in range(3):
        app_module.schedule_once(store, two_cards, spawn=spawn, vram_check=free_gpu)

    assert len(spawn.calls) == 2
    assert store.load(third).status == "queued"


def test_the_hold_names_every_card_that_refused(client, two_cards, spawn):  # noqa: F811
    """On a two-card box "GPU 1 has too little" names half the problem and sends
    the reader to the wrong card."""
    client.app.state.vram_check = busy_gpu
    store = store_of(client)
    job_id = _submit(client, "m1")

    assert app_module.schedule_once(store, two_cards, spawn=spawn,
                                    vram_check=busy_gpu) is None

    reason = store.load(job_id).queued_reason
    assert "GPU 0" in reason and "GPU 1" in reason, reason


def test_the_hold_says_when_a_counter_exceeds_the_cards(client, settings, spawn):  # noqa: F811
    """Otherwise a box with max_concurrent=2 and one card looks broken rather
    than misconfigured."""
    capped = settings.model_copy(update={"gpus": [1], "max_concurrent": 2})
    store = store_of(client)
    _submit(client, "m1")
    second = _submit(client, "m2")

    app_module.schedule_once(store, capped, spawn=spawn, vram_check=free_gpu)
    app_module.schedule_once(store, capped, spawn=spawn, vram_check=free_gpu)

    reason = store.load(second).queued_reason or ""
    assert "ATR_TRAIN_GPUS" in reason, reason


# ── the record ──────────────────────────────────────────────────────────────
def test_a_job_that_never_started_holds_no_card(client):  # noqa: F811
    """Refused at the door, so it never got as far as a card."""
    client.app.state.vram_check = busy_gpu

    assert store_of(client).load(_submit(client, "m1")).gpu is None


def test_an_unallocated_record_is_not_taken_to_hold_the_default_card(
        client, two_cards, spawn):  # noqa: F811
    """A record from before cards were allocated has `gpu: None`. Reading that as
    "holds `settings.gpu`" would be a guess; reading it as "holds nothing" can at
    worst put a second job on a card an old run uses, which the VRAM check then
    catches. Neither is free, and the code says which it chose."""
    store = store_of(client)
    first = _submit(client, "m1")
    running = store.load(first)
    running.gpu = None                      # a record from before cards existed
    store.save(running)
    store.advance(store.load(first), "preparing")

    second = _submit(client, "m2")
    app_module.schedule_once(store, two_cards, spawn=spawn, vram_check=free_gpu)

    assert store.load(second).gpu in (0, 1)


def test_the_health_body_says_which_cards_it_trains_on(client):  # noqa: F811
    """It already reports `gpu` and `gpus` — the second being what nvidia-smi
    sees. Which cards the box may *train* on is policy, and a reader needs both
    to tell "the card is busy" from "we never use that card"."""
    body = client.get("/health").json()

    assert "training_gpus" in body
    assert body["concurrency_limit"] >= 1


# ── the comments the epic lists ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", [
    "src/atr_training/contracts.py",
    "src/atr_training/ketos_cmd.py",
])
def test_no_file_still_says_the_unit_pins_the_card(path):
    """#12 lists six comments that become wrong. `deploy/systemd/atr-train.service`
    sets no CUDA_VISIBLE_DEVICES at all — it says so — so "the unit sets
    CUDA_VISIBLE_DEVICES=1" was already wrong before this change and would have
    been read as the reason a job cannot pick a card."""
    text = (ROOT / path).read_text(encoding="utf-8")

    assert "unit sets CUDA_VISIBLE_DEVICES" not in text
    assert "unit sets ``CUDA_VISIBLE_DEVICES" not in text


def test_the_service_unit_really_does_not_pin_a_card():
    """The claim the comments above are corrected against."""
    unit = (ROOT / "deploy" / "systemd" / "atr-train.service").read_text(encoding="utf-8")
    directives = [line for line in unit.splitlines()
                  if line.strip().startswith("Environment") and "CUDA_VISIBLE" in line]

    assert directives == [], directives
