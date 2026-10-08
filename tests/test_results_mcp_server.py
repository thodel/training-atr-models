"""#174: the server half — stable tool names, one probe call per tool, and the ssh
transport that ships the probe. The server tests need the mcp SDK (the `mcp`
extra); CI installs it, a developer without it sees skips, not failures."""
from __future__ import annotations

import asyncio
import json
import subprocess
from types import SimpleNamespace

import pytest

from atr_results_mcp import remote
from atr_results_mcp.remote import SshTransport, TransportError, strip_banner

mcp = pytest.importorskip("mcp")
from atr_results_mcp.server import TOOL_NAMES, build_server  # noqa: E402


class Recording:
    def __init__(self, answer=None, fail: str | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.answer = answer or {"ok": True}
        self.fail = fail

    def call(self, cmd, args):
        self.calls.append((cmd, args))
        if self.fail:
            raise TransportError(self.fail)
        return dict(self.answer, cmd=cmd)


def tools_of(server):
    return asyncio.run(server.list_tools())


def call(server, name, **args):
    result = asyncio.run(server.call_tool(name, args))
    return json.loads(result.content[0].text)


# ── names are the interface ─────────────────────────────────────────────────
def test_the_tool_names_are_the_ones_an_allow_rule_names():
    """mcp__atr-results__<name> is what settings.local.json allows, once. A rename
    is a new permission prompt that no unattended run can answer (#174)."""
    names = [t.name for t in tools_of(build_server(Recording()))]
    assert names == list(TOOL_NAMES)
    assert set(names) == {"queue", "finished", "job", "results", "draw", "prepared",
                          "deadlines", "log", "report", "slurm_job", "checkout"}


def test_every_tool_has_a_description_a_reader_can_act_on():
    for tool in tools_of(build_server(Recording())):
        assert len(tool.description or "") > 60, tool.name


# ── each tool is one probe call ─────────────────────────────────────────────
@pytest.mark.parametrize("name, args, sent", [
    ("queue", {}, {}),
    ("finished", {"days": 3}, {"days": 3}),
    ("job", {"job_id": "x"}, {"job_id": "x"}),
    ("results", {"granularity": "page"}, {"granularity": "page"}),
    ("draw", {"job_id": "x"}, {"job_id": "x"}),
    ("prepared", {}, {}),
    ("deadlines", {}, {}),
    ("log", {"slurm_job_id": "17545199", "lines": 6}, {"slurm_job_id": "17545199", "lines": 6}),
    ("report", {"tag": "gemma4-e4b-xix"}, {"evalset": "evalset-federal-minutes",
                                           "tag": "gemma4-e4b-xix"}),
    ("slurm_job", {"slurm_job_id": "1"}, {"slurm_job_id": "1"}),
    ("checkout", {}, {}),
])
def test_a_tool_sends_its_command_and_only_the_arguments_given(name, args, sent):
    link = Recording()
    answer = call(build_server(link), name, **args)
    assert link.calls == [(name, sent)]
    assert answer["cmd"] == name


def test_a_transport_failure_is_an_answer_not_a_crash():
    answer = call(build_server(Recording(fail="no host answered")), "queue")
    assert answer == {"error": "no host answered", "cmd": "queue"}


# ── the transport ───────────────────────────────────────────────────────────
def test_the_probe_travels_on_stdin_and_the_json_is_quoted_for_the_remote_shell():
    seen = {}

    def runner(argv, input, capture_output, text, timeout):
        seen.update(argv=argv, stdin=input)
        return SimpleNamespace(returncode=0, stdout='{"jobs": []}', stderr="")

    link = SshTransport(hosts=["ubelix"], probe_source="PROBE", runner=runner)
    answer = link.call("finished", {"days": 2})
    assert seen["stdin"] == "PROBE"
    assert seen["argv"][:5] == ["ssh", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes"]
    assert seen["argv"][5:8] == ["ubelix", "python3", "-"]
    # ssh re-parses the remote command line: unquoted, {"days": 2} arrives as {days: 2}.
    assert seen["argv"][-1] == "'{\"days\": 2}'"
    assert answer == {"jobs": [], "_via": "ubelix"}


def test_the_next_host_is_tried_when_ssh_itself_fails():
    attempts = []

    def runner(argv, **kw):
        host = argv[5]
        attempts.append(host)
        if host == "ubelix":
            return SimpleNamespace(returncode=255, stdout="",
                                   stderr="** post-quantum …\nkex_exchange_identification: "
                                          "Connection timed out during banner exchange")
        return SimpleNamespace(returncode=0, stdout='{"ok": 1}', stderr="")

    link = SshTransport(hosts=["ubelix", "ubelix-direct"], probe_source="P", runner=runner)
    assert link.call("queue", {}) == {"ok": 1, "_via": "ubelix-direct"}
    assert attempts == ["ubelix", "ubelix-direct"]


def test_no_host_names_the_vpn_and_every_failure():
    def runner(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)

    link = SshTransport(hosts=["a", "b"], probe_source="P", runner=runner, call_timeout=1)
    with pytest.raises(TransportError, match="VPN missing.*a: no answer.*b: no answer"):
        link.call("queue", {})


def test_a_probe_error_passes_through_with_its_stderr():
    def runner(argv, **kw):
        return SimpleNamespace(returncode=1, stdout='{"error": "sacct: exit 1"}',
                               stderr="** post-quantum warning\nreal detail")

    answer = SshTransport(hosts=["h"], probe_source="P", runner=runner).call("finished", {})
    assert answer["error"] == "sacct: exit 1"
    assert answer["_stderr"] == "real detail"


def test_non_json_from_the_probe_is_a_transport_error():
    def runner(argv, **kw):
        return SimpleNamespace(returncode=0, stdout="Traceback …", stderr="")

    with pytest.raises(TransportError, match="did not answer with JSON"):
        SshTransport(hosts=["h"], probe_source="P", runner=runner).call("queue", {})


def test_the_banner_is_not_an_error():
    assert strip_banner("** WARNING: connection is not using a post-quantum key exchange\n"
                        "** See https://openssh.com/pq.html\nPermission denied") \
        == "Permission denied"


def test_hosts_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ATR_RESULTS_HOSTS", "ubelix-direct, ubelix")
    assert remote.default_hosts() == ["ubelix-direct", "ubelix"]
    monkeypatch.delenv("ATR_RESULTS_HOSTS")
    assert remote.default_hosts() == ["ubelix", "ubelix-direct"]


def test_the_shipped_probe_is_the_file_in_this_package():
    link = SshTransport(hosts=["h"], runner=lambda *a, **k: None)
    assert link.probe_source == remote.PROBE_PATH.read_text(encoding="utf-8")
    assert "def dispatch" in link.probe_source
