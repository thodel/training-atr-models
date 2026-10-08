"""The MCP server: eleven reading tools over the probe (#156, #174).

Tool names are an interface. An unattended session allows them once, by name
(``mcp__atr-results__queue``), and a rename is a new permission prompt that
nobody is there to answer. Add tools; do not rename them.

The server holds no logic of its own: every tool is one probe call, and the probe
is what the suite tests. ``build_server`` takes the transport so a test can pass a
fake and see which command each tool sends.
"""
from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from .remote import SshTransport, Transport, TransportError

TOOL_NAMES = ("queue", "finished", "job", "results", "draw", "prepared", "deadlines",
              "log", "report", "slurm_job", "checkout")

INSTRUCTIONS = (
    "Read-only view of the ATR training programme on UBELIX (Slurm) for the daily "
    "report. Times are Europe/Zurich; fields ending in _cest were converted from the "
    "trainer's UTC. Nothing here submits, cancels or deletes. Start with queue, "
    "finished and results; use draw before putting two CER values side by side; "
    "deadlines warns about artefact expiry and the scratch purge."
)


def build_server(transport: Transport | None = None) -> MCPServer:
    link = transport or SshTransport()

    def ask(cmd: str, **args: Any) -> dict[str, Any]:
        try:
            return link.call(cmd, {k: v for k, v in args.items() if v is not None})
        except TransportError as exc:
            return {"error": str(exc), "cmd": cmd}

    server = MCPServer(name="atr-results", version="1.0.0", instructions=INSTRUCTIONS)

    @server.tool()
    def queue() -> dict:
        """Running and pending Slurm jobs of the training user, with the reason a
        pending job waits (explained), and for a running job its last progress
        counter and a verdict whether it fits its wall (fits / at risk / will hit
        the wall; at risk means within 15 % of the limit)."""
        return ask("queue")

    @server.tool()
    def finished(days: int = 2) -> dict:
        """Slurm jobs that ended in the last `days` days (sacct). A FAILED, TIMEOUT
        or OUT_OF_MEMORY job carries the cause line pulled from its log; a job under
        10 s is marked short_run: a requeue that found the record finished, not a
        crash. code_drift lists stages that ran on another commit than the one the
        job was created with."""
        return ask("finished", days=days)

    @server.tool()
    def job(job_id: str) -> dict:
        """One training job record (job.json): status, base model, granularity,
        4-bit flag, metrics, stages with their commits, code drift, artefact,
        registration note, Slurm job ids, and the draw fingerprint."""
        return ask("job", job_id=job_id)

    @server.tool()
    def results(status: str | None = None, granularity: str | None = None,
                base_model: str | None = None) -> dict:
        """The metrics table over all job records: CER, WER, length_ratio,
        truncated_cer, samples, draw_md5. Filter by status (completed, training,
        failed), granularity (line, page, block, mixed) or a substring of the base
        model. Two rows compare only when draw_md5 matches."""
        return ask("results", status=status, granularity=granularity, base_model=base_model)

    @server.tool()
    def draw(job_id: str) -> dict:
        """Fingerprint (md5, lines) of a job's evaluation draw and every other job
        that shares it. The answer to "are these two CER values comparable?"."""
        return ask("draw", job_id=job_id)

    @server.tool()
    def prepared() -> dict:
        """Jobs whose corpus is built but whose GPU stage never finished (status
        preparing/training/queued), with the scratch purge date of their data and
        whether a completed job of the same model_id supersedes them."""
        return ask("prepared")

    @server.tool()
    def deadlines() -> dict:
        """Artefact cache entries with built/expiry times (7 days unless pinned),
        their claimants' status and an at_risk flag; the 30-day scratch purge date
        of prepared jobs; home filesystem usage."""
        return ask("deadlines")

    @server.tool()
    def log(slurm_job_id: str, lines: int = 40) -> dict:
        """Head and tail of a Slurm job's log without DEBUG and progress-bar noise,
        plus notes: the code pin line, commit drift lines, the cause line of a
        failure, the last progress counter."""
        return ask("log", slurm_job_id=slurm_job_id, lines=lines)

    @server.tool()
    def report(evalset: str = "evalset-federal-minutes", tag: str | None = None) -> dict:
        """Benchmark reports on a shared evaluation set, sorted by CER, with
        load_in_4bit so quantisation is visible. With tag, one report's scalar
        fields."""
        return ask("report", evalset=evalset, tag=tag)

    @server.tool()
    def slurm_job(slurm_job_id: str) -> dict:
        """scontrol show job for one Slurm id: command, TRES, limits, reason (with
        the state of any node named as unavailable), and the log's notes. Falls
        back to accounting once Slurm has forgotten the job."""
        return ask("slurm_job", slurm_job_id=slurm_job_id)

    @server.tool()
    def checkout() -> dict:
        """The training-atr-models checkout on UBELIX: HEAD, origin/main, ahead and
        behind counts, dirty files. submit.sh refuses a checkout that is behind or
        dirty, and a job runs the code of this HEAD."""
        return ask("checkout")

    return server
