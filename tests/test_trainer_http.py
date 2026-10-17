"""HttpTrainer turns the trainer's 409 for a live model_id into SubmitConflict."""

import httpx
import pytest

from atr_training.sweep_driver import SubmitConflict
from atr_training.trainer_http import HttpTrainer


def client_answering(status, body):
    t = HttpTrainer("http://trainer")
    t._client = httpx.Client(base_url="http://trainer", transport=httpx.MockTransport(
        lambda request: httpx.Response(status, json=body)))
    return t


def test_a_live_model_id_names_the_job():
    t = client_answering(409, {"detail": "job 20261017T111646Z-x is already training for "
                                         "model_id 'x'. Two live jobs …"})
    with pytest.raises(SubmitConflict) as info:
        t.submit({"model_id": "x"})
    assert info.value.job_id == "20261017T111646Z-x"


def test_the_curated_id_409_is_not_a_live_job():
    t = client_answering(409, {"detail": "model_id 'x' is a curated id in …"})
    with pytest.raises(httpx.HTTPStatusError):
        t.submit({"model_id": "x"})


def test_a_200_is_returned():
    assert client_answering(200, {"job_id": "j"}).submit({}) == {"job_id": "j"}
