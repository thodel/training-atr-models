"""Who may talk to atr-train, decided before any route runs (#13).

Until #13 the service had no authentication at all and relied on its loopback
bind. It is now reached from idhefix, and the bind had to open — onto a box whose
ufw does not filter high ports (a listener on :8299 was reached from idhefix
*and* from a VPN client, 16.09.2026) and where nobody has sudo to add a source
rule. From that bind on, this module is the only barrier, so every check fails
closed.

One ASGI middleware rather than a dependency per route: a route added later is
covered without anyone remembering to. Per request, in this order:

1. **Source.** The client address comes from the ASGI scope — the socket peer —
   and never from a header; the launcher also turns off uvicorn's
   ``X-Forwarded-For`` rewriting, since no proxy stands in front of this service.
   Loopback always passes; anything else must be inside
   ``ATR_TRAIN_ALLOWED_CLIENTS``, and an empty list admits no one else. A scope
   without a usable IP — no ``client``, or Starlette's test client, which reports
   the host ``"testclient"`` — is refused like a stranger: tests present a
   loopback address instead of production code trusting a test transport.
2. ``GET``/``HEAD`` on exactly ``/health`` needs no key: liveness must not need
   a secret. ``/health/`` and every other spelling are ordinary paths.
3. No key configured (with ``require_auth`` on): **503**, for everything but
   ``/health`` — an unconfigured trainer serves nothing.
4. A non-loopback caller while the settings would not pass the launcher (key too
   short, ``require_auth`` off): **503**. This is what stops ``python -m uvicorn
   --host 0.0.0.0`` from skipping the launcher's check.
5. ``X-API-Key`` missing, repeated, or wrong: **401**.

Unknown paths get the same treatment, so an unauthenticated caller cannot tell a
route from a typo. ``/docs`` and ``/redoc`` are switched off in the app —
Swagger UI loads the schema without the header, so behind a key they could only
ever show an error — and ``/openapi.json`` needs the key like any route.

The key is compared with :func:`hmac.compare_digest` and appears in no log line,
no response and no exception message; refusals name the setting to fix.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
from collections.abc import Callable
from typing import Any

from loguru import logger

#: The one route a caller without the key may use, and the methods that read it.
OPEN_PATH = "/health"
OPEN_METHODS = frozenset({"GET", "HEAD"})
KEY_HEADER = b"x-api-key"

#: The gateway (serving-atr-inference#137) turns 401 and 403 into a 502 naming
#: ATR_TRAIN_API_KEY or ATR_TRAIN_ALLOWED_CLIENTS, and passes 503 through, so
#: these texts reach whoever operates the gateway. Worded in the contract.
UNCONFIGURED = "atr-train has no ATR_TRAIN_API_KEY configured; set it in .env and restart"
BAD_KEY = "missing or invalid X-API-Key"


def _address(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """``text`` as an address, with an IPv4-mapped IPv6 address read as IPv4.

    A dual-stack socket reports an IPv4 peer as ``::ffff:130.92.59.240``; the
    allowlist names ``130.92.59.240``, and the two must compare equal.
    """
    try:
        address = ipaddress.ip_address(text.strip().strip("[]"))
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def is_loopback_host(host: str) -> bool:
    """True for 127.0.0.0/8, ::1 and ``localhost``; False for everything else.

    ``0.0.0.0`` and ``::`` bind every interface and are *not* loopback. A name
    other than ``localhost`` is not resolved: whatever it resolves to today, the
    launcher treats it as the wider bind.
    """
    if host.strip().lower() == "localhost":
        return True
    address = _address(host)
    return address is not None and address.is_loopback


def client_address(scope: dict) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The peer of this request, from the ASGI scope alone; None if it has none."""
    client = scope.get("client")
    if not client:
        return None
    return _address(str(client[0]))


def _presented_keys(scope: dict) -> list[bytes]:
    return [value for name, value in scope.get("headers") or () if name.lower() == KEY_HEADER]


def _key_matches(scope: dict, api_key: str) -> bool:
    # Exactly one header. With two, which one a framework reads is an accident of
    # its implementation, and "the other one was right" is not an answer to give.
    presented = _presented_keys(scope)
    return len(presented) == 1 and hmac.compare_digest(presented[0], api_key.encode("utf-8"))


def refusal(scope: dict, settings: Any) -> tuple[int, str] | None:
    """``(status, detail)`` when this request must not reach a route, else None."""
    address = client_address(scope)
    if address is None:
        client = scope.get("client")
        shown = repr(client[0]) if client else "(none)"
        return 403, (f"client {shown} is not an IP address; atr-train serves only "
                     "callers it can place")
    loopback = address.is_loopback
    if not loopback and not any(address in net for net in settings.allowed_networks()):
        return 403, f"client {address} is not in ATR_TRAIN_ALLOWED_CLIENTS"

    if scope.get("path") == OPEN_PATH and scope.get("method") in OPEN_METHODS:
        return None
    return key_refusal(scope, settings)


def key_refusal(scope: dict, settings: Any) -> tuple[int, str] | None:
    """The key half of :func:`refusal`, for a route that is open by path but has
    a mode that is not: ``/health?deep=1`` calls the gateway with the gateway's
    key, so it asks for the trainer's key like every other route (#48)."""
    loopback = bool((address := client_address(scope)) and address.is_loopback)
    if settings.require_auth and not settings.api_key:
        return 503, UNCONFIGURED
    if not loopback:
        problems = settings.remote_access_problems()
        if problems:
            return 503, ("atr-train is not configured to serve remote callers: "
                         + "; ".join(problems))
    if settings.require_auth and not _key_matches(scope, settings.api_key):
        return 401, BAD_KEY
    return None


async def _answer(send: Callable, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii"))]})
    await send({"type": "http.response.body", "body": body})


class AccessGuard:
    """The middleware. ``settings`` is a callable so tests can swap the settings
    object on ``app.state`` between requests, as the routes already allow."""

    def __init__(self, app: Callable, settings: Callable[[], Any]) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        kind = scope["type"]
        if kind == "http":
            refused = refusal(scope, self.settings())
            if refused is None:
                await self.app(scope, receive, send)
                return
            status, detail = refused
            # Path as repr: it is the caller's text, and a newline in it must not
            # forge a line in the journal. The key's value is never at hand here.
            logger.warning("refused {} {!r} from {} with {}: {}", scope.get("method"),
                           scope.get("path"), (scope.get("client") or ("(none)",))[0],
                           status, detail)
            await _answer(send, status, detail)
            return
        if kind == "websocket":
            # No websocket route exists. Closing before accept is a 403 to the peer.
            await send({"type": "websocket.close", "code": 1008})
            return
        await self.app(scope, receive, send)  # lifespan
