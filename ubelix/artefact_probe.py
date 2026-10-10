#!/usr/bin/env python3
"""Would this spec's prepare only ADOPT a compiled corpus, or build one? (#212)

    artefact_probe.py <spec.json>     -> prints ARTEFACT=cached | ARTEFACT=build

`ubelix/submit.sh` asks this to pick a walltime, and that is the whole purpose.
A prepare that adopts an artefact is done in seconds — the four olmOCR page preps
of 09.10.2026 took 7 s each — while one that builds from scratch has taken
11 h 15 (job 17795292). One default had to cover the build, and 8 CPUs x 20 h =
9,600 CPU-minutes is 83 % of what `job_gratis` allows (11,520, counted as CPUs x
*remaining* walltime of RUNNING jobs). So the prepare sat PENDING on
`MaxCpuRunMinsPerUser` behind any other running job of ours, and was unblocked by
hand three times.

**Every uncertainty answers `build`.** A wrong `cached` books a half-hour wall for
an eleven-hour job, which is a TIMEOUT and costs the corpus; a wrong `build` books
twelve hours for a job that takes seconds, which is merely the behaviour this
replaces. The asymmetry is the design, not caution — see :func:`verdict`.

**The key comes from the engine's own ``_cache_key``, never from a copy of it.**
The backends fold different options in: kraken the chunk size, the VLM backend the
granularity and the two character caps. A second implementation here would drift
from the one the job actually looks up and then answer confidently about the wrong
digest — which is worse than not answering.

Runs INSIDE the container: it needs pydantic, `atr_training` and the engine
package. `preflight.py`, which runs on the login node's python 3.9 with the
standard library only, is the other half of the same checking step.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

#: Where `prepare.sbatch` points the artefact cache. Declared here as well as
#: there because this runs on the login node, before the batch file's own
#: `export` has happened — and pinned to it by
#: `tests/test_ubelix_prepare_walltime.py`, so the two cannot drift apart. That
#: is the same remedy `backends.supports_chunked_prepare` uses for the same
#: shape of problem.
DEFAULT_CACHE_ROOT = (Path("/scratch/network/users") / (os.environ.get("USER") or "")
                      / "expA" / "artefacts")

#: How much of an unpinned entry's life must be left before `cached` is safe.
#:
#: The entry is checked now and adopted when the job starts, and a prepare that
#: waits in the queue can cross the 7-day deadline in between — after which it
#: rebuilds, under a wall chosen for a job that would not. A pinned entry cannot
#: expire at all and is exempt.
MARGIN_DAYS = 1.0


def cache_root() -> Path:
    """The cache this submission's job will read. The env wins, as in the job."""
    env = os.environ.get("ATR_TRAIN_ARTEFACT_CACHE_ROOT")
    return Path(env) if env else DEFAULT_CACHE_ROOT


def pipeline_class(engine: str) -> Any:
    """The backend class whose ``_cache_key`` the job will call."""
    from atr_training.backends import backend_for

    return importlib.import_module(backend_for(engine).runner_module).Pipeline


def cache_key(request: Any, settings: Any) -> Any:
    """The digest the job's backend will look up, or None if it does not cache.

    The pipeline is built as the runner builds it. Its store and page source are
    inert at construction — no directory is created, nothing is fetched — so a
    throwaway store is enough to reach ``_cache_key``, and reaching the real one
    is the point.
    """
    from atr_training.contracts import TrainJob
    from atr_training.jobstore import JobStore

    with tempfile.TemporaryDirectory(prefix="artefact-probe-") as tmp:
        pipeline = pipeline_class(request.engine)(JobStore(Path(tmp)), settings)
        # Private on purpose: only the backend knows what compiling means, and
        # this asks it rather than deciding for it.
        return pipeline._cache_key(TrainJob(id="artefact-probe", request=request))


def verdict(spec_path: str, *, root: Optional[Path] = None,
            settings: Any = None) -> Tuple[str, str]:
    """``("cached"|"build", why)`` for the spec at ``spec_path``.

    ``build`` whenever the answer is not a plain, comfortable hit: no key, no
    entry, an expired entry, or one close enough to its deadline that the queue
    wait could carry it past (:data:`MARGIN_DAYS`).
    """
    from atr_training.artefact_cache import ArtefactCache
    from atr_training.contracts import TrainRequest
    from atr_training.settings import TrainerSettings

    settings = settings or TrainerSettings()
    request = TrainRequest.model_validate(
        json.loads(Path(spec_path).read_text(encoding="utf-8")))
    key = cache_key(request, settings)
    if key is None:
        return "build", "the %s backend does not reuse artefacts" % request.engine

    root = Path(root) if root is not None else cache_root()
    cache = ArtefactCache(root)
    # `entry`, not `lookup`: a probe must not stamp `last_used`. That field is the
    # only evidence eviction has that a run still depends on an artefact, and a
    # submission that never starts would be forging it.
    entry = cache.entry(key)
    if entry is None:
        return "build", "%s is not in %s — this prepare builds the corpus" % (key, root)
    ok, why = entry.usable(cache.max_age_days)
    if not ok:
        return "build", "%s: %s — this prepare rebuilds it" % (key, why)
    if not entry.pinned and entry.age_days + MARGIN_DAYS > cache.max_age_days:
        return "build", (
            "%s: %s, within %.0f day of the %.0f-day limit — it could expire while "
            "this prepare waits in the queue" % (key, why, MARGIN_DAYS, cache.max_age_days))
    return "cached", "%s: %s, %.1f GiB — this prepare only adopts it" % (
        key, why, entry.bytes_ / 2 ** 30)


def main(argv: list) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    answer, why = verdict(argv[0])
    print("artefact: %s" % why, file=sys.stderr)
    print("ARTEFACT=%s" % answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
