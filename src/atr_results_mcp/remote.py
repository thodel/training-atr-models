"""How the server reaches the cluster: ``ssh <host> python3 - <cmd> <json>``.

The probe's source travels on stdin with every call. That is the whole deployment
story: no package on UBELIX, no checkout to keep current, no version skew between
the tool and what answers it. A call costs one SSH round trip (about two seconds
through the jump host).

Hosts are tried in order. ``ubelix`` goes through ``srv-train`` (idhefix) and has
hung in banner exchange while the VPN was fine; ``ubelix-direct`` reaches submit02
without the jump. The answer says which one served it (``_via``).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Protocol

PROBE_PATH = Path(__file__).with_name("probe.py")

#: The OpenSSH client's own warning about the server's key exchange; it is not an
#: error and not from the probe.
_BANNER_MARKERS = ("post-quantum", "store now, decrypt later", "openssh.com/pq.html")


class TransportError(RuntimeError):
    """No host answered, or the probe did not return JSON."""


class Transport(Protocol):
    def call(self, cmd: str, args: dict[str, Any]) -> dict[str, Any]: ...


def default_hosts() -> list[str]:
    raw = os.environ.get("ATR_RESULTS_HOSTS", "ubelix,ubelix-direct")
    return [h.strip() for h in raw.split(",") if h.strip()]


def strip_banner(stderr: str) -> str:
    return "\n".join(line for line in stderr.splitlines()
                     if not any(marker in line for marker in _BANNER_MARKERS)).strip()


class SshTransport:
    """Runs the probe on the first host that answers."""

    def __init__(self, hosts: list[str] | None = None, connect_timeout: int = 25,
                 call_timeout: float = 180.0, probe_source: str | None = None,
                 runner=subprocess.run) -> None:
        self.hosts = hosts or default_hosts()
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout
        self.probe_source = probe_source or PROBE_PATH.read_text(encoding="utf-8")
        self._run = runner

    def argv(self, host: str, cmd: str, args: dict[str, Any]) -> list[str]:
        # ssh joins its remote arguments into ONE command line that the remote
        # shell parses again, so the JSON must be quoted for that shell or its
        # double quotes vanish and the probe sees {days:2}.
        return ["ssh", "-o", "ConnectTimeout=%d" % self.connect_timeout, "-o", "BatchMode=yes",
                host, "python3", "-", shlex.quote(cmd), shlex.quote(json.dumps(args))]

    def call(self, cmd: str, args: dict[str, Any]) -> dict[str, Any]:
        failures: list[str] = []
        for host in self.hosts:
            try:
                proc = self._run(self.argv(host, cmd, args), input=self.probe_source,
                                 capture_output=True, text=True, timeout=self.call_timeout)
            except subprocess.TimeoutExpired:
                failures.append("%s: no answer within %.0f s" % (host, self.call_timeout))
                continue
            stderr = strip_banner(proc.stderr or "")
            if proc.returncode == 255 or not (proc.stdout or "").strip():
                # 255 is ssh itself (no route, banner timeout, refused key); an empty
                # stdout means the probe never ran. Both are reasons to try the next host.
                failures.append("%s: ssh exit %d: %s" % (host, proc.returncode, stderr[-300:]))
                continue
            try:
                result = json.loads(proc.stdout)
            except ValueError:
                raise TransportError("%s: the probe did not answer with JSON: %s"
                                     % (host, (proc.stdout or "")[-300:]))
            if isinstance(result, dict):
                result["_via"] = host
                if stderr and "error" in result:
                    result["_stderr"] = stderr[-300:]
            return result
        raise TransportError("no host answered; VPN missing is the usual cause. "
                             + " | ".join(failures))
