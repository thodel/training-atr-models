"""The trainer over HTTP: the calls the sweep driver and the noise-floor
measurement make, and nothing the service does not already offer.

It lived in ``scripts/run_sweep.py`` until the VLM noise floor needed the same
client from a second script (#138); a copy would have been a second client to
keep in step with the API.
"""

from __future__ import annotations

import re

import httpx

__all__ = ["HttpTrainer"]


def _json(response: httpx.Response) -> dict | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


class HttpTrainer:
    """The trainer over HTTP. Five calls, all of them already in its API."""

    def __init__(self, base_url: str, api_key: str | None = None,
                 timeout: float = 30.0) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers,
                                    timeout=timeout)

    def verify(self, request: dict) -> dict:
        response = self._client.post("/jobs/verify", json=request)
        response.raise_for_status()
        return response.json()

    def submit(self, request: dict) -> dict:
        response = self._client.post("/jobs", json=request)
        if response.status_code == 409:
            # "job <id> is already <status> for model_id …" — the one 409 that
            # names a live job; the curated-id 409 names none and is raised.
            detail = str((_json(response) or {}).get("detail") or "")
            live = re.match(r"job (\S+) is already ", detail)
            if live:
                from atr_training.sweep_driver import SubmitConflict
                raise SubmitConflict(live.group(1), detail)
        response.raise_for_status()
        return response.json()

    def job(self, job_id: str) -> dict:
        response = self._client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def jobs(self) -> list[dict]:
        """Every job the trainer knows, in the shape `requested_jobs` needs.

        ``fields=summary`` because the whole record of every job is 807 KB for
        42 of them (serving-atr-inference#107) and this is polled while waiting.
        """
        response = self._client.get("/jobs", params={"fields": "summary"})
        response.raise_for_status()
        return response.json().get("jobs", [])

    def cancel(self, job_id: str) -> dict:
        response = self._client.post(f"/jobs/{job_id}/cancel")
        response.raise_for_status()
        return response.json()
