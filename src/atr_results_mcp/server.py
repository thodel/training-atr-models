"""The MCP server: thirteen reading tools over the probe (#156, #174).

Tool names are an interface. An unattended session allows them once, by name
(``mcp__atr-results__queue``), and a rename is a new permission prompt that
nobody is there to answer. Add tools; do not rename them.

The server holds no logic of its own: every tool is one probe call, and the probe
is what the suite tests. ``build_server`` takes the transport so a test can pass a
fake and see which command each tool sends.
"""
from __future__ import annotations

from typing import Any

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server
    SERVER_GENERATION = 2
except ImportError:  # mcp 1.x: the same decorator API under its old name
    from mcp.server.fastmcp import FastMCP as _Server
    SERVER_GENERATION = 1

from .remote import SshTransport, Transport, TransportError

TOOL_NAMES = ("queue", "live", "finished", "job", "results", "draw", "prepared", "deadlines",
              "log", "report", "slurm_job", "storage", "checkout")

INSTRUCTIONS = (
    "Read-only view of the ATR training programme on UBELIX (Slurm) and asteraix "
    "(the atr-train service) for the daily report. The record tools (results, job, "
    "draw, prepared, live) read both job stores and name the host on every row; the "
    "Slurm tools (queue, finished, slurm_job, log) are UBELIX only, and live is what "
    "says what runs on asteraix. Times are Europe/Zurich; fields ending in _cest were "
    "converted from the trainer's UTC. Nothing here submits, cancels or deletes. Start "
    "with queue, live, finished and results; use draw before putting two CER values "
    "side by side; deadlines warns about artefact expiry and the scratch purge."
)


def transport_security(public_host: str | None) -> Any:
    """The Host headers the streamable HTTP transport accepts.

    The SDK's DNS-rebinding protection allows only loopback Hosts by default.
    Behind nginx the header says ``tei.dh.unibe.ch``, and every proxied request
    came back ``421 Misdirected Request`` (11.10.2026, #214 A4). Naming the
    public host keeps the protection on and lets that one name through;
    without a public host the answer is None and the SDK default stands.
    """
    if not public_host:
        return None
    from mcp.server.transport_security import TransportSecuritySettings

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[public_host, f"{public_host}:*", "127.0.0.1:*", "localhost:*"],
        allowed_origins=[f"https://{public_host}", f"http://{public_host}",
                         "http://127.0.0.1:*", "http://localhost:*"],
    )


def build_server(transport: Transport | None = None, public_host: str | None = None) -> Any:
    link = transport or SshTransport()

    def ask(cmd: str, **args: Any) -> dict[str, Any]:
        try:
            return link.call(cmd, {k: v for k, v in args.items() if v is not None})
        except TransportError as exc:
            return {"error": str(exc), "cmd": cmd}

    # Why 1.x is what the laptop runs: mcp 2.x negotiates the 2026-07-28 protocol
    # revision in a form Claude Code 2.1.274 refuses over stdio ("connection is
    # serving the 2026-07-28 protocol; the initialize handshake is not accepted",
    # 07.10.2026). The 1.x line negotiates a 2025 revision and connects. The tool
    # API is the same under both names, so the server is written once.
    # 2.x takes the transport security in run(), not here: __main__ passes it.
    if SERVER_GENERATION == 2:
        server = _Server(name="atr-results", version="1.0.0", instructions=INSTRUCTIONS)
    else:
        server = _Server(name="atr-results", instructions=INSTRUCTIONS,
                         transport_security=transport_security(public_host))

    @server.tool()
    def queue() -> dict:
        """Running and pending Slurm jobs of the training user, with the reason a
        pending job waits (explained), and for a running job its last progress
        counter (from the Slurm stdout, else from the record's stage log; its
        source and state are named, and a counter from a known false source such
        as a retry count is `unattributable` and never projected) and a verdict
        whether it fits its wall (fits / at risk / will hit the wall / unknown;
        at risk means within 15 % of the limit). `wall.why` writes the sum out:
        the rest of the running stage from tqdm's estimate or the ratio, plus
        the stages still ahead (test after train) measured on a completed
        sibling named in `following`; without one, `uncounted_stages` says so.
        `wall.on_timeout` says, from the batch file's own header, what the wall
        does to the job (train.sbatch: SIGTERM 120 s before, the trap requeues,
        the trainer resumes; a scoring job: TIMEOUT ends it), and `attempt` is
        the attempt number from the log, so "attempt 2 running" is read, not
        guessed."""
        return ask("queue")

    @server.tool()
    def live(host: str | None = None) -> dict:
        """Job records that have a runner or wait for one (queued, preparing,
        compiling, training, testing, registering), on every host, read from the
        record and the stage log: status, stage, pid and cards, epoch and steps,
        peak GPU memory, the last progress counter and when the log last moved.
        This is how asteraix's state is read, since it has no Slurm; filter with
        host="asteraix", "ubelix" or "idhefix". A dead runner leaves its record
        live until the service (asteraix) or close_job (UBELIX) closes it, so
        judge a row by updated_cest and log_modified_cest."""
        return ask("live", host=host)

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
        """One training job record (job.json) from whichever store holds it, the
        UBELIX scratch store or the shared store asteraix writes: host, status,
        base model, granularity, 4-bit flag, metrics, stages with their commits,
        code drift, artefact, registration note, Slurm job ids, and the draw
        fingerprint. For a UBELIX record in `training` or `queued`, `resubmit`
        holds the one command that puts it back in the queue, written out and
        never run from here: pinned to the commit its last attempt ran (so the
        resume carries no drift), with the container that attempt printed, the
        checkpoint it resumes from, every Slurm attempt with its end state, and
        `command: null` with `why` while an attempt is still RUNNING or PENDING
        or the last one FAILED with a cause. `checkout_behind` warns that
        submit.sh will refuse until someone pulls."""
        return ask("job", job_id=job_id)

    @server.tool()
    def results(status: str | None = None, granularity: str | None = None,
                base_model: str | None = None) -> dict:
        """The metrics table over all job records of both stores (UBELIX scratch
        and the shared store asteraix writes), each row with its host: CER, WER,
        length_ratio, truncated_cer, samples, draw_md5, reserved_pages. Filter by
        status (completed, training, failed), granularity (line, page, block,
        mixed) or a substring of the base model.

        Two rows compare only when draw_md5 **and** reserved_pages match: the
        draw says they were scored on the same lines, reserved_pages that they
        were trained on the same corpus. The medieval ladder's September rows
        carry 0 or None and its October rows 3173 — 28 % fewer training lines
        from an identical spec (#98). None means nobody looked, which is not 0.

        `stores` says whether each store was readable, so an absent share is not
        mistaken for zero asteraix rows."""
        return ask("results", status=status, granularity=granularity, base_model=base_model)

    @server.tool()
    def draw(job_id: str) -> dict:
        """Fingerprint (md5, lines) of a job's evaluation draw and every other job
        in either store that shares it, with its host. The answer to "are these
        two CER values comparable?" — and it needs two conditions, not one.

        A shared draw says the rows were **scored** on the same lines. It says
        nothing about what they were **trained** on, so `heldout.reserved_pages`
        comes back for this job and for every sharer, and `not_comparable` names
        the sharers whose corpus differs. The medieval ladder read across that
        gap for three weeks: September rows reserved 0 pages, October rows 3173
        — 28 % fewer training lines from an identical spec (#98). A
        `reserved_pages` of None means nobody looked, which is not 0.
        """
        return ask("draw", job_id=job_id)

    @server.tool()
    def prepared() -> dict:
        """Jobs whose corpus is built but whose GPU stage never finished (status
        preparing/training/queued), in either store, with the scratch purge date
        of their data (null on the share, which never purges) and whether a
        completed job of the same model_id supersedes them."""
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
    def storage() -> dict:
        """Belegung des research-storage (`/storage/research/wbkolleg_dh_1`) und,
        wenn weniger als 10 % frei sind, Vorschläge was sich woanders besser
        aufhebt — je Posten mit Grösse, Klasse und Begründung.

        Die Klassen unterscheiden, was eine Löschung wirklich kostet: eine
        zweite lokale Kopie, die einzige lokale Kopie (aber auf dem Hub, also
        ein Download), oder abgeleitete Arbeitsdaten (wiederherstellbar nur
        über Stunden prepare). Vorschläge gibt es ausschliesslich innerhalb von
        `Textrecognition_Training`; die übrigen 38 Verzeichnisse des Bereichs
        gehören anderen Projekten und werden ohne Empfehlung berichtet.

        Lesend. Löscht nichts, und schlägt nichts vor, was nicht aus dem
        Inventar belegt ist — fehlt das Inventar oder ist es älter als 14 Tage,
        sagt die Antwort das."""
        return ask("storage")

    @server.tool()
    def checkout() -> dict:
        """The training-atr-models checkout on UBELIX: HEAD, origin/main, ahead and
        behind counts, dirty files. submit.sh refuses a checkout that is behind or
        dirty, and a job runs the code of this HEAD."""
        return ask("checkout")

    return server
