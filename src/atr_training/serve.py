"""Start atr-train — the only supported way, and the one the unit uses (#13).

    python -m atr_training.serve --host 0.0.0.0 --port 8204
    python -m atr_training.serve --check --host 0.0.0.0     # judge, do not start

A loopback bind starts as it always did. Any other bind — ``0.0.0.0`` and ``::``
included — is refused with exit code 2 unless the settings hold all three of
``require_auth``, an ``api_key`` of at least 32 characters, and a non-empty
``allowed_clients``. The refusal names each missing piece and never the key.

Why the launcher and not only the middleware: asteraix's ufw does not filter
high ports and nobody there has sudo, so a bind beyond loopback is reachable
from the whole university network the moment it exists. A service that starts
in that state and refuses requests is already one mistake from open; one that
does not start is not. The middleware checks the same list per request
(:meth:`TrainerSettings.remote_access_problems`) for the case where somebody
starts uvicorn by hand.

Exit code 2 is also what the unit's ``RestartPreventExitStatus=`` names: a
refused bind is a configuration error that no restart fixes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from pydantic import ValidationError

from atr_training.access import is_loopback_host
from atr_training.settings import TrainerSettings, get_settings

APP = "kraken_train_svc.app:app"
#: Configuration refused — distinct from uvicorn's own failures (exit 1), which
#: a restart may well cure (a port still held by the previous process).
EXIT_REFUSED = 2


def bind_problems(host: str, settings: TrainerSettings) -> list[str]:
    """Why ``host`` must not be bound with these settings; [] when it may."""
    if is_loopback_host(host):
        return []
    return settings.remote_access_problems()


def main(argv: list[str] | None = None, *, settings: TrainerSettings | None = None,
         run: Callable[..., object] | None = None) -> int:
    """Parse, judge, start. ``settings`` and ``run`` are test seams."""
    if settings is None:
        try:
            settings = get_settings()
        except ValidationError as exc:
            # A bad ATR_TRAIN_ALLOWED_CLIENTS entry lands here. The message shows
            # the offending input; no validator exists on the key, so it is never
            # among them.
            print(f"atr-train: refusing to start, the settings do not validate:\n{exc}",
                  file=sys.stderr)
            return EXIT_REFUSED

    parser = argparse.ArgumentParser(prog="python -m atr_training.serve",
                                     description=__doc__.split("\n", 1)[0])
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--check", action="store_true",
                        help="judge the bind against the settings and exit")
    args = parser.parse_args(argv)

    problems = bind_problems(args.host, settings)
    if problems:
        print(f"atr-train: refusing to bind {args.host}:{args.port}. A bind beyond "
              "loopback is reachable from the university network (asteraix's ufw does "
              "not filter high ports), so .env must hold all of ATR_TRAIN_REQUIRE_AUTH, "
              "ATR_TRAIN_API_KEY and ATR_TRAIN_ALLOWED_CLIENTS. Missing:\n  - "
              + "\n  - ".join(problems), file=sys.stderr)
        return EXIT_REFUSED
    if args.check:
        print(f"atr-train: {args.host}:{args.port} may be bound with these settings")
        return 0

    if run is None:
        import uvicorn

        run = uvicorn.run
    # proxy_headers off: uvicorn otherwise rewrites the client address from
    # X-Forwarded-For for connections from 127.0.0.1. Nothing proxies this
    # service, and the allowlist must judge the socket peer, not a header.
    run(APP, host=args.host, port=args.port, proxy_headers=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
