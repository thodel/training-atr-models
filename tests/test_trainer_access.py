"""The trainer's door (#13): who may call, with which key, and how it is started.

Until #13 the service had no authentication and relied on a loopback bind. The
bind is now 0.0.0.0 on a box whose ufw does not filter high ports — a listener
on :8299 was reached from idhefix and from a VPN client on 16.09.2026 — so these
tests are the whole argument that opening it was safe. They run against the
real app and its real middleware; only the peer address is chosen per test.
"""

from __future__ import annotations

import configparser
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from loguru import logger
from pydantic import ValidationError, model_validator
from starlette.routing import Route

from atr_training import access, serve
from atr_training.jobstore import JobStore
from atr_training.preflight import GpuInfo
from atr_training.settings import MIN_API_KEY_LENGTH, TrainerSettings
from kraken_train_svc import app as app_module

REPO = Path(__file__).resolve().parents[1]
UNIT = REPO / "deploy" / "systemd" / "atr-train.service"

LOOPBACK = ("127.0.0.1", 50000)
IDHEFIX = "130.92.59.240"
#: The VPN client that reached a test listener on asteraix:8299.
VPN_CLIENT = "130.92.212.96"
#: A key that would pass a naive "is it set?" check and nothing more.
SHORT_KEY = "short-but-still-a-secret"


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    """A client of the real app at any peer address, with settings of the test's choosing.

    No ``with``: the lifespan (scheduler, reconcile) is not what these tests are
    about, and the routes they reach need only a store and the settings.
    """
    app = app_module.app
    monkeypatch.setattr(app_module, "query_gpus", lambda: [GpuInfo(0, 20986, 46068)])

    def make(address=LOOPBACK, headers=None, **overrides) -> TestClient:
        settings = TrainerSettings(**{"jobs_root": tmp_path / "jobs",
                                      "trained_root": tmp_path / "trained",
                                      "venvs_root": tmp_path / "venvs", **overrides})
        app.state.settings = settings
        app.state.store = JobStore(settings.jobs_root, host_id=settings.host_id)
        return TestClient(app, client=address, headers=headers or {})

    yield make
    for attr in ("settings", "store"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


# ── the key ─────────────────────────────────────────────────────────────────
def test_the_trainer_refuses_a_request_without_a_key(make_client, trainer_key):
    """The service used to answer anyone who could reach it; now nobody without the key."""
    c = make_client()
    missing = c.get("/jobs")
    assert missing.status_code == 401
    assert missing.json() == {"detail": access.BAD_KEY}
    assert c.get("/jobs", headers={"X-API-Key": "not-the-key"}).status_code == 401
    assert c.post("/jobs", json={}).status_code == 401, "refused before validation"
    assert c.get("/jobs", headers={"X-API-Key": trainer_key}).status_code == 200


def test_a_second_key_header_is_refused_even_if_one_is_right(make_client, trainer_key):
    """Which of two headers a framework reads is an implementation accident."""
    c = make_client()
    both = [("X-API-Key", trainer_key), ("X-API-Key", "wrong")]
    assert c.get("/jobs", headers=both).status_code == 401
    assert c.get("/jobs", headers=[("x-api-key", trainer_key)]).status_code == 200


def test_every_route_but_health_needs_the_key(make_client, trainer_key):
    """Enumerated from the app, so a route added later is checked, not listed."""
    c = make_client()
    checked = set()
    for route in app_module.app.routes:
        if not isinstance(route, Route):  # APIRoute is one; mounts would not be
            continue
        path = re.sub(r"\{[^}]+\}", "20260916T061502Z-x", route.path)
        for method in sorted(route.methods or ()):
            status = c.request(method, path).status_code
            checked.add((method, route.path))
            if route.path == "/health":
                assert status == 200, (method, route.path, status)
            else:
                assert status == 401, (method, route.path, status)
    # Not vacuous: the routes #13 names are among the ones just checked.
    assert {("POST", "/jobs"), ("POST", "/jobs/verify"), ("GET", "/jobs"),
            ("GET", "/jobs/{job_id}"), ("GET", "/jobs/{job_id}/log"),
            ("GET", "/jobs/{job_id}/curve"), ("POST", "/jobs/{job_id}/cancel"),
            ("DELETE", "/jobs/{job_id}"), ("GET", "/gpu-claim"), ("GET", "/gpu"),
            ("GET", "/openapi.json"), ("GET", "/health"), ("HEAD", "/health")} <= checked


@pytest.mark.parametrize("method,path", [
    ("GET", "/health/"), ("GET", "//health"), ("GET", "/Health"), ("GET", "/health/x"),
    ("POST", "/health"), ("DELETE", "/health"),
    ("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json"),
    ("GET", "/no-such-route"), ("GET", "/"),
])
def test_only_the_exact_health_route_is_open(make_client, method, path):
    """No other spelling is exempt, and a stranger cannot map the API surface:
    an unknown path and a real one answer the same 401."""
    assert make_client().request(method, path).status_code == 401


def test_the_docs_are_off_and_the_schema_needs_the_key(make_client, trainer_key):
    """Swagger UI would fetch the schema without the header and show an error."""
    c = make_client(headers={"X-API-Key": trainer_key})
    assert c.get("/docs").status_code == 404
    assert c.get("/redoc").status_code == 404
    assert "/jobs" in c.get("/openapi.json").json()["paths"]
    assert "/gpu" in c.get("/openapi.json").json()["paths"]


def test_head_health_is_treated_like_get_health(make_client):
    c = make_client()
    assert c.head("/health").status_code == 200
    assert c.get("/health").json()["status"] == "ok"


def test_an_unconfigured_trainer_serves_only_health(make_client, trainer_key):
    """Fail closed: no key configured means nothing but liveness, whoever asks."""
    c = make_client(api_key="", headers={"X-API-Key": trainer_key})
    assert c.get("/health").status_code == 200
    for method, path in [("GET", "/jobs"), ("POST", "/jobs"), ("GET", "/gpu"),
                         ("GET", "/gpu-claim"), ("GET", "/openapi.json"),
                         ("GET", "/no-such-route")]:
        answer = c.request(method, path)
        assert answer.status_code == 503, (method, path)
        assert answer.json() == {"detail": access.UNCONFIGURED}


def test_require_auth_off_waives_the_key_for_loopback_only(make_client, trainer_key):
    """The development switch. A remote caller is refused regardless — the
    launcher would not have bound for it, and uvicorn started by hand must not
    make it reachable."""
    local = make_client(require_auth=False, allowed_clients=IDHEFIX)
    assert local.get("/jobs").status_code == 200

    remote = make_client(address=(IDHEFIX, 40000), headers={"X-API-Key": trainer_key},
                         require_auth=False, allowed_clients=IDHEFIX)
    answer = remote.get("/jobs")
    assert answer.status_code == 503
    assert "ATR_TRAIN_REQUIRE_AUTH" in answer.json()["detail"]
    assert remote.get("/health").status_code == 200


def test_a_short_key_serves_loopback_but_no_remote_caller(make_client):
    """A hand-started `uvicorn --host 0.0.0.0` meets the launcher's rule at the door."""
    local = make_client(api_key=SHORT_KEY, headers={"X-API-Key": SHORT_KEY})
    assert local.get("/jobs").status_code == 200

    remote = make_client(address=(IDHEFIX, 40000), headers={"X-API-Key": SHORT_KEY},
                         api_key=SHORT_KEY, allowed_clients=IDHEFIX)
    answer = remote.get("/jobs")
    assert answer.status_code == 503
    assert f"shorter than {MIN_API_KEY_LENGTH}" in answer.json()["detail"]
    assert SHORT_KEY not in answer.text


# ── the source ──────────────────────────────────────────────────────────────
def test_a_client_outside_the_allowlist_is_refused(make_client, trainer_key):
    """The VPN client that reached :8299 — with the right key, and lying in
    every forwarding header there is. The socket peer decides."""
    spoofed = {"X-API-Key": trainer_key, "X-Forwarded-For": f"{IDHEFIX}, 127.0.0.1",
               "X-Real-IP": IDHEFIX, "Forwarded": f"for={IDHEFIX}"}
    c = make_client(address=(VPN_CLIENT, 40000), headers=spoofed, allowed_clients=IDHEFIX)
    for path in ("/health", "/jobs", "/gpu", "/no-such-route"):
        answer = c.get(path)
        assert answer.status_code == 403, path
        assert answer.json() == {
            "detail": f"client {VPN_CLIENT} is not in ATR_TRAIN_ALLOWED_CLIENTS"}

    gateway = make_client(address=(IDHEFIX, 40000), headers={"X-API-Key": trainer_key},
                          allowed_clients=IDHEFIX)
    assert gateway.get("/jobs").status_code == 200
    assert gateway.get("/jobs", headers={"X-API-Key": "wrong"}).status_code == 401


def test_an_empty_allowlist_admits_loopback_only(make_client, trainer_key):
    """"No restriction" would be the wrong reading: an empty list on a 0.0.0.0
    bind is exactly the state the launcher refuses."""
    stranger = make_client(address=(IDHEFIX, 40000), headers={"X-API-Key": trainer_key})
    assert stranger.get("/health").status_code == 403
    assert make_client(headers={"X-API-Key": trainer_key}).get("/jobs").status_code == 200


def test_an_ipv4_mapped_address_matches_the_allowlist(make_client, trainer_key):
    """A dual-stack socket reports idhefix as ::ffff:130.92.59.240."""
    mapped = make_client(address=(f"::ffff:{IDHEFIX}", 40000),
                         headers={"X-API-Key": trainer_key}, allowed_clients=IDHEFIX)
    assert mapped.get("/jobs").status_code == 200

    loop = make_client(address=("::ffff:127.0.0.1", 40000))
    assert loop.get("/health").status_code == 200

    other = make_client(address=(f"::ffff:{VPN_CLIENT}", 40000),
                        headers={"X-API-Key": trainer_key}, allowed_clients=IDHEFIX)
    answer = other.get("/jobs")
    assert answer.status_code == 403
    assert VPN_CLIENT in answer.json()["detail"]


@pytest.mark.parametrize("client,allowed,refused", [
    (("127.0.0.1", 1), "", False),
    (("127.8.9.10", 1), "", False),
    (("::1", 1), "", False),
    (("130.92.59.241", 1), "130.92.59.240/30", False),
    (("130.92.59.244", 1), "130.92.59.240/30", True),
    (("2001:db8::7", 1), "2001:db8::/64, 130.92.59.240", False),
    (("130.92.59.240", 1), "2001:db8::/64", True),
    # Starlette's test client, and a transport that gives no peer at all.
    (("testclient", 50000), "130.92.59.240", True),
    (None, "130.92.59.240", True),
    (("", 0), "130.92.59.240", True),
])
def test_the_source_is_judged_from_the_scope(client, allowed, refused, trainer_key):
    settings = TrainerSettings(allowed_clients=allowed)
    scope = {"type": "http", "method": "GET", "path": "/health", "client": client,
             "headers": [(b"x-api-key", trainer_key.encode())]}
    result = access.refusal(scope, settings)
    assert (result is not None and result[0] == 403) is refused, result


@pytest.mark.parametrize("entry", [
    "130.92.59.240/24",     # host bits set: a typo must not widen the rule to a /24
    "idhefix",              # names are not resolved
    "130.92.59.300",
    "0.0.0.0/0",            # restricts nothing while looking like a rule
    "::/0",
])
def test_a_bad_allowlist_entry_fails_at_startup(entry):
    with pytest.raises(ValidationError, match="ATR_TRAIN_ALLOWED_CLIENTS"):
        TrainerSettings(allowed_clients=f"{IDHEFIX}, {entry}")


def test_a_good_allowlist_parses(monkeypatch):
    monkeypatch.setenv("ATR_TRAIN_ALLOWED_CLIENTS", f" {IDHEFIX}, 10.0.0.0/8 ,")
    assert [str(n) for n in TrainerSettings().allowed_networks()] == [
        f"{IDHEFIX}/32", "10.0.0.0/8"]


# ── the launcher ────────────────────────────────────────────────────────────
ENV_OF = {"api_key": "ATR_TRAIN_API_KEY", "allowed_clients": "ATR_TRAIN_ALLOWED_CLIENTS",
          "require_auth": "ATR_TRAIN_REQUIRE_AUTH"}


class FakeRun:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, app, **kwargs):
        self.calls.append((app, kwargs))


@pytest.mark.parametrize("host,env,starts,missing", [
    ("127.0.0.1", {"api_key": "", "allowed_clients": ""}, True, []),
    ("127.3.2.1", {"api_key": "", "allowed_clients": ""}, True, []),
    ("localhost", {"api_key": "", "allowed_clients": ""}, True, []),
    ("::1", {"api_key": "", "allowed_clients": ""}, True, []),
    ("0.0.0.0", {"api_key": "", "allowed_clients": IDHEFIX}, False, ["ATR_TRAIN_API_KEY"]),
    ("0.0.0.0", {"api_key": SHORT_KEY, "allowed_clients": IDHEFIX}, False,
     ["ATR_TRAIN_API_KEY"]),
    ("0.0.0.0", {"allowed_clients": ""}, False, ["ATR_TRAIN_ALLOWED_CLIENTS"]),
    ("0.0.0.0", {"allowed_clients": IDHEFIX, "require_auth": "false"}, False,
     ["ATR_TRAIN_REQUIRE_AUTH"]),
    ("0.0.0.0", {"allowed_clients": IDHEFIX}, True, []),
    ("::", {"api_key": "", "allowed_clients": ""}, False,
     ["ATR_TRAIN_API_KEY", "ATR_TRAIN_ALLOWED_CLIENTS"]),
    ("::", {"allowed_clients": IDHEFIX}, True, []),
    ("130.92.59.242", {"allowed_clients": ""}, False, ["ATR_TRAIN_ALLOWED_CLIENTS"]),
    ("asteraix", {"allowed_clients": ""}, False, ["ATR_TRAIN_ALLOWED_CLIENTS"]),
])
@pytest.mark.parametrize("check", [False, True], ids=["start", "check"])
def test_the_launcher_refuses_an_unsafe_bind(host, env, starts, missing, check, monkeypatch,
                                             capsys, trainer_key):
    """Settings through their real environment names, as the unit's .env gives them.

    ``--check`` (install_user_unit.sh) must judge exactly as a start does: the
    process test below relies on it to never start a server.
    """
    for field, value in env.items():
        monkeypatch.setenv(ENV_OF[field], value)
    run = FakeRun()
    argv = ["--host", host, "--port", "8204", *(["--check"] if check else [])]
    code = serve.main(argv, settings=TrainerSettings(), run=run)
    out, err = capsys.readouterr()
    # Both streams, on every path: the starting ones are the ones every deploy takes.
    for secret in (trainer_key, SHORT_KEY):
        assert secret not in out + err

    if starts:
        assert code == 0 and err == ""
        if check:
            assert run.calls == [] and "may be bound" in out
        else:
            assert run.calls == [(serve.APP, {"host": host, "port": 8204,
                                              "proxy_headers": False})]
        return
    assert code == serve.EXIT_REFUSED == 2
    assert run.calls == [] and out == ""
    named = err.split("Missing:", 1)[1]
    for name in ENV_OF.values():
        assert (name in named) is (name in missing), (name, named)


def test_the_launcher_refuses_an_unsafe_bind_as_a_process(tmp_path):
    """The module the unit runs, from a clean interpreter: exit code and output.

    Always with ``--check``, which is decided after the bind is judged (the test
    above pins that). Without it, the regression this test exists to catch — a
    bad key let through — would not fail fast: uvicorn runs the app's lifespan
    before it binds, so the child would create the job directories, reconcile
    every record there and start the scheduler, then listen on 0.0.0.0:8204
    until the timeout. A guarded probe on 16.09.2026 did exactly that. And in
    case ``--check`` itself regresses, the child's home and every root of the
    store point into tmp_path: its defaults are ``~/atr-cache`` and this
    checkout's ``.venvs``, which on asteraix are the real ones.
    """
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO / "engines")]),
           "HOME": str(home),
           "ATR_TRAIN_JOBS_ROOT": str(tmp_path / "jobs"),
           "ATR_TRAIN_TRAINED_ROOT": str(tmp_path / "trained"),
           "ATR_TRAIN_CHECKPOINT_ROOT": str(tmp_path / "checkpoints"),
           "ATR_TRAIN_VENVS_ROOT": str(tmp_path / "venvs"),
           "ATR_TRAIN_API_KEY": SHORT_KEY, "ATR_TRAIN_ALLOWED_CLIENTS": IDHEFIX}
    argv = [sys.executable, "-m", "atr_training.serve", "--check",
            "--host", "0.0.0.0", "--port", "8204"]

    def launch(**overrides) -> subprocess.CompletedProcess:
        return subprocess.run(argv, cwd=tmp_path, env={**env, **overrides},
                              capture_output=True, text=True, timeout=20)

    refused = launch()
    assert refused.returncode == 2, refused.stderr
    assert f"ATR_TRAIN_API_KEY is shorter than {MIN_API_KEY_LENGTH}" in refused.stderr
    assert SHORT_KEY not in refused.stdout + refused.stderr

    # A long, distinctive key: pydantic truncates a printed input in the middle,
    # so a whole-key check would miss the prefix that shows.
    key = "process-test-key-" + "0123456789abcdef" * 2
    bad_list = launch(ATR_TRAIN_API_KEY=key, ATR_TRAIN_ALLOWED_CLIENTS="130.92.59.240/24")
    assert bad_list.returncode == 2
    assert "host bits set" in bad_list.stderr and "Traceback" not in bad_list.stderr
    for secret in (key, key[:8], key[-8:]):
        assert secret not in bad_list.stdout + bad_list.stderr

    # What install_user_unit.sh asks: may the unit's bind start with this .env?
    ok = launch(ATR_TRAIN_API_KEY=key)
    assert ok.returncode == 0, ok.stderr
    assert "may be bound" in ok.stdout
    assert key[:8] not in ok.stdout + ok.stderr
    assert list(home.iterdir()) == [] and not (tmp_path / "jobs").exists(), \
        "--check started the service"


def _unit() -> configparser.SectionProxy:
    parser = configparser.ConfigParser(strict=False, interpolation=None, delimiters=("=",))
    parser.optionxform = str  # systemd keys are case-sensitive
    parser.read_string(UNIT.read_text(encoding="utf-8"))
    return parser["Service"]


def test_the_unit_starts_the_launcher_not_uvicorn(trainer_key):
    """uvicorn started directly would bind whatever it is told."""
    service = _unit()
    argv = shlex.split(service["ExecStart"])
    assert argv[0].endswith("/.venvs/kraken-train/bin/python")
    assert argv[1:3] == ["-m", "atr_training.serve"]
    assert not any("uvicorn" in arg for arg in argv)
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    assert argv[argv.index("--port") + 1] == "8204"

    # The launcher imports atr_training from src/, and uvicorn imports the app
    # from the working directory.
    assert "PYTHONPATH=%h/Repo/training-atr-models/src" in service["Environment"]
    assert service["WorkingDirectory"].endswith("/training-atr-models/engines")
    # A refused bind is not retried every 5 s, and a restart still spares the run.
    assert service["RestartPreventExitStatus"].split() == [str(serve.EXIT_REFUSED)]
    assert service["KillMode"] == "process"

    # And the launcher accepts exactly that command line, given the three settings.
    run = FakeRun()
    good = TrainerSettings(api_key=trainer_key, allowed_clients=IDHEFIX)
    assert serve.main(argv[3:], settings=good, run=run) == 0
    assert run.calls == [(serve.APP, {"host": "0.0.0.0", "port": 8204,
                                      "proxy_headers": False})]


def test_the_app_module_starts_through_the_launcher():
    """`python -m kraken_train_svc.app` used to call uvicorn.run with no check."""
    source = Path(app_module.__file__).read_text(encoding="utf-8")
    main_block = source.split('if __name__ == "__main__"', 1)[1]
    assert "atr_training.serve" in main_block
    assert "uvicorn" not in main_block


# ── the key stays secret ────────────────────────────────────────────────────
def test_the_key_is_never_logged(make_client, trainer_key, capsys):
    """Not in a log line, a response, a repr, or the launcher's refusal —
    and a wrong key a caller presented is not echoed either."""
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG",
                      format="{level} {message} {extra} {exception}")
    presented = "presented-by-a-caller-0123456789abcdef"
    callers = [
        (LOOPBACK, None, {}),
        (LOOPBACK, presented, {}),
        (LOOPBACK, trainer_key, {}),
        (LOOPBACK, trainer_key, {"api_key": ""}),
        ((VPN_CLIENT, 1), trainer_key, {"allowed_clients": IDHEFIX}),
        ((IDHEFIX, 1), presented, {"allowed_clients": IDHEFIX}),
        ((IDHEFIX, 1), SHORT_KEY, {"api_key": SHORT_KEY, "allowed_clients": IDHEFIX}),
    ]
    try:
        bodies = []
        for address, key, overrides in callers:
            # One at a time: the settings live on app.state, so the next client's
            # would apply to this one's requests too.
            c = make_client(address=address, headers={"X-API-Key": key} if key else None,
                            **overrides)
            for path in ("/jobs", "/gpu-claim", "/health", "/no-such-route"):
                bodies.append(c.get(path).text)
        for key in ("", SHORT_KEY):
            serve.main(["--host", "0.0.0.0"], run=FakeRun(),
                       settings=TrainerSettings(api_key=key, require_auth=False))
        # And the paths every deploy takes: a start, and install_user_unit.sh's check.
        good = TrainerSettings(api_key=trainer_key, allowed_clients=IDHEFIX)
        for extra in ([], ["--check"]):
            run = FakeRun()
            assert serve.main(["--host", "0.0.0.0", "--port", "8204", *extra],
                              settings=good, run=run) == 0
            assert len(run.calls) == (0 if extra else 1)
        settings = TrainerSettings(api_key=trainer_key, gateway_api_key=presented)
        shown = [repr(settings), str(settings)]
    finally:
        logger.remove(sink)
    printed = capsys.readouterr()

    assert any("refused" in line for line in lines), "nothing was logged; the test is vacuous"
    everything = "\n".join([*lines, *bodies, *shown, printed.out, printed.err])
    for secret in (trainer_key, presented, SHORT_KEY):
        assert secret not in everything


def test_a_settings_error_never_prints_the_key(monkeypatch, capsys, trainer_key):
    """For an error raised by a model-level validator, pydantic's own message
    carries the whole input as ``input_value`` — the key's first and last
    characters, truncated in the middle. Only field validators raise today;
    the launcher's refusal must not depend on that staying true."""
    class Refusing(TrainerSettings):
        @classmethod
        def settings_customise_sources(cls, settings_cls, init_settings, **sources):
            # The key as the only input, so pydantic's truncation cannot hide it.
            return (init_settings,)

        @model_validator(mode="after")
        def _refuse(self) -> "Refusing":
            raise ValueError("a model-level check failed")

    monkeypatch.setattr(serve, "get_settings", lambda: Refusing(api_key=trainer_key))
    run = FakeRun()
    assert serve.main(["--check", "--host", "0.0.0.0"], run=run) == serve.EXIT_REFUSED
    out, err = capsys.readouterr()
    assert "settings: Value error, a model-level check failed" in err
    assert run.calls == []
    for secret in (trainer_key, trainer_key[:8], trainer_key[-8:]):
        assert secret not in out + err


def test_the_example_env_would_not_open_the_service_by_accident():
    """An unreplaced placeholder long enough to pass the launcher would be a
    known key on a network-facing service. Empty is refused; the allowlist names
    idhefix alone; and the trainer reads no ATR_API_KEY, so none is offered."""
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    values = dict(line.split("=", 1) for line in text.splitlines()
                  if line and not line.startswith("#") and "=" in line)
    assert len(values["ATR_TRAIN_API_KEY"]) < MIN_API_KEY_LENGTH
    assert values["ATR_TRAIN_ALLOWED_CLIENTS"] == IDHEFIX
    assert "ATR_API_KEY" not in values
    assert "ATR_TRAIN_HOST" not in values, "the bind belongs to the unit"
    assert "ATR_TRAIN_REQUIRE_AUTH" not in values, "leave the dev switch at its default"
    assert serve.bind_problems("0.0.0.0", TrainerSettings(
        api_key=values["ATR_TRAIN_API_KEY"],
        allowed_clients=values["ATR_TRAIN_ALLOWED_CLIENTS"])) == ["ATR_TRAIN_API_KEY is empty"]
