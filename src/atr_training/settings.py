"""Trainer-service configuration (env prefix ``ATR_TRAIN_``).

Kept separate from the gateway's :class:`atr_serving.config.Settings`: the
trainer owns paths and guards the gateway has no business knowing about, and it
runs in its own venv.

Both classes use ``extra="ignore"``, which matters because the prefixes overlap —
the gateway reads ``ATR_TRAIN_URL`` as its ``train_url`` (#35) and this class
would otherwise see it as an unknown ``url``.

One instance of this service supervises **every** training backend; which
interpreter and runner module a job gets is looked up per engine in
:mod:`atr_training.backends`.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from atr_training.backends import runner_python
from atr_training.jobstore import LEGACY_JOB_HOST, SLURM_HOST

REPO_ROOT = Path(__file__).resolve().parents[2]
#: ``<registry_root>/models.yaml`` — the gateway's ``CURATED_FILENAME``.
CURATED_FILENAME = "models.yaml"

#: Shortest ``ATR_TRAIN_API_KEY`` a non-loopback bind is started with.
#: ``secrets.token_urlsafe(24)`` is 32 characters; anything shorter is a
#: placeholder or a password somebody typed.
MIN_API_KEY_LENGTH = 32

#: What a host id may look like: it is written into job.json and claim files.
_HOST_ID_RE = re.compile(r"[A-Za-z0-9._-]+")


@lru_cache(maxsize=16)
def parse_allowed_clients(value: str) -> tuple:
    """``ATR_TRAIN_ALLOWED_CLIENTS`` as a tuple of networks. Raises on any bad entry.

    ``strict`` parsing on purpose: ``130.92.59.240/24`` is refused ("host bits
    set") rather than quietly read as the whole /24 — a typo in the only source
    restriction this box has must not widen it. ``/0`` is refused for the same
    reason: it restricts nothing while looking like a rule. Empty items (a
    trailing comma) are skipped. Cached because the middleware asks per request.
    """
    networks = []
    for entry in (item.strip() for item in value.split(",")):
        if not entry:
            continue
        try:
            network = ipaddress.ip_network(entry)
        except ValueError as exc:
            raise ValueError(
                f"ATR_TRAIN_ALLOWED_CLIENTS entry {entry!r} is not an IP address or "
                f"network: {exc}") from None
        if network.prefixlen == 0:
            raise ValueError(
                f"ATR_TRAIN_ALLOWED_CLIENTS entry {entry!r} admits every address; "
                "name the hosts that may call (the gateway: 130.92.59.240)")
        networks.append(network)
    return tuple(networks)


class TrainerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATR_TRAIN_", env_file=".env", extra="ignore")

    #: Defaults for ``python -m atr_training.serve`` started by hand. The unit
    #: passes both on its command line, so the deployed bind is in git.
    host: str = "127.0.0.1"
    port: int = 8204

    # ── access (#13) ──────────────────────────────────────────────────────
    #: The key a caller presents as ``X-API-Key``. **Shared with idhefix under the
    #: same name**: the gateway reads its own ``ATR_TRAIN_API_KEY`` and sends it on
    #: every ``/train/*`` call. Not the gateway's ``ATR_API_KEY`` — #9 split control
    #: of this box from inference on that one, so a leaked caller key cannot start
    #: training runs. Empty: this service answers nothing but ``/health``.
    #: ``repr=False`` keeps it out of every repr of this object, and so out of logs.
    api_key: str = Field(default="", repr=False)
    #: A development switch, and only on loopback: a non-loopback client is never
    #: served without the key, whatever this says (:mod:`atr_training.access`),
    #: and the launcher refuses a non-loopback bind with it off.
    require_auth: bool = True
    #: Comma-separated IPs or CIDRs served besides loopback, which always is.
    #: The only source restriction asteraix has: its ufw does not filter high
    #: ports — a listener on :8299 was reached from idhefix and from a VPN client
    #: on 16.09.2026 — and nobody there has sudo to add a rule. Empty admits
    #: loopback callers only. A bad entry fails at startup, not at the first call.
    allowed_clients: str = ""

    @field_validator("allowed_clients")
    @classmethod
    def _allowed_clients_parse(cls, value: str) -> str:
        parse_allowed_clients(value)
        return value

    def allowed_networks(self) -> tuple:
        return parse_allowed_clients(self.allowed_clients)

    def remote_access_problems(self) -> list[str]:
        """What stops this service from serving a non-loopback caller; [] if nothing.

        One list for two doors: the launcher refuses a non-loopback *bind* with
        it, and the middleware refuses a non-loopback *client* with it — so
        ``python -m uvicorn ... --host 0.0.0.0``, which skips the launcher, meets
        the same three conditions at the first request. Names settings, never the
        key's value or length.
        """
        problems = []
        if not self.require_auth:
            problems.append("ATR_TRAIN_REQUIRE_AUTH is false (allowed on a loopback bind only)")
        if not self.api_key:
            problems.append("ATR_TRAIN_API_KEY is empty")
        elif len(self.api_key) < MIN_API_KEY_LENGTH:
            problems.append(f"ATR_TRAIN_API_KEY is shorter than {MIN_API_KEY_LENGTH} characters")
        if not self.allowed_networks():
            problems.append("ATR_TRAIN_ALLOWED_CLIENTS is empty (name the gateway, "
                            "130.92.59.240)")
        return problems

    # ── which host this is (#15) ──────────────────────────────────────────
    #: The name this service stamps on every job it accepts, and the only jobs it
    #: will spawn, reconcile, cancel or count on its card. A pid means something
    #: only on the machine that assigned it: with two trainers on one store, each
    #: read the other's pids against its own /proc and wrote ``failed`` into runs
    #: that were training on the other machine.
    #:
    #: A setting, not ``gethostname()``, because the hostnames are no identity:
    #: idhefix reports ``srv`` and asteraix ``dhserver03`` (measured 16.09.2026).
    #: The hostname stays the default so a box nobody configured still has a
    #: stable name of its own.
    host_id: str = Field(default_factory=socket.gethostname, validate_default=True)
    #: The host a record WITHOUT ``host`` belongs to. Every such record in the
    #: shared store — all 48 on 16.09.2026 — was written by the old trainer on
    #: idhefix, which predates the field and will never write it.
    legacy_job_host: str = LEGACY_JOB_HOST

    @field_validator("host_id", "legacy_job_host")
    @classmethod
    def _host_names_are_plain(cls, value: str, info: ValidationInfo) -> str:
        # The name goes into job records, claim files and log lines, and is
        # compared byte for byte: a stray space would be a host nobody owns.
        if not _HOST_ID_RE.fullmatch(value or ""):
            env = f"ATR_TRAIN_{info.field_name.upper()}"
            raise ValueError(f"{env} must be non-empty and match {_HOST_ID_RE.pattern}, "
                             f"got {value!r}")
        if info.field_name == "host_id" and value == SLURM_HOST:
            # Slurm supervises those jobs. A trainer calling itself this would
            # spawn every job UBELIX has queued, onto its own card.
            raise ValueError(f"ATR_TRAIN_HOST_ID={SLURM_HOST!r} is reserved for jobs "
                             "Slurm runs on UBELIX; no trainer may take it")
        return value

    # ── layout ────────────────────────────────────────────────────────────
    #: One directory per job; all job state lives here (see jobstore).
    jobs_root: Path = Path.home() / "atr-cache" / "training"
    #: Promoted weights, one directory per trained model. Every registration's
    #: ``local_path`` is under here, and the engines on idhefix open it as
    #: written — so on a trainer this must be on the share, next to
    #: ``registry_root``. The default is local disk, fine for a box that
    #: registers nowhere; a registration from it is refused (runner_base
    #: ``_write_registration``) rather than served as a path idhefix cannot open.
    #: Absolute, for the reason ``registry_root`` is.
    trained_root: Path = Path.home() / "atr-cache" / "trained"
    #: A weights directory under ``trained_root`` without ``metadata.json`` is
    #: removed only once nothing in it has changed for this many hours. A
    #: registration writes ``metadata.json`` LAST, so a directory in the middle of
    #: one looks exactly like an orphan — and ``trained_root`` is shared, so the
    #: registration can be the other machine's, for a job this store has never
    #: seen (#15). At least an hour: a VLM registration copies gigabytes over SMB.
    orphan_weights_min_age_h: float = Field(default=24.0, ge=1.0)
    #: The shared registry directory (#5, #14) — MUST equal the gateway's
    #: ``ATR_REGISTRY_ROOT`` on idhefix. The gateway publishes the curated
    #: ``models.yaml`` here, and every trained model is registered here as
    #: ``trained/<id>.yaml`` (:mod:`atr_training.registration`).
    #:
    #: It replaced ``overlay_path`` (``<repo>/config/models.local.yaml``). After
    #: the split that file was in THIS checkout, on the other machine from the
    #: gateway, and all three runners registered into it without a word (#14).
    #: Absolute, like the gateway's: a relative root is a different registry for
    #: every working directory, which is the problem the share exists to remove.
    registry_root: Path = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/registry")
    #: The curated registry, read only to resolve a ``base_model`` given as a
    #: registry id rather than a Zenodo DOI (#76). Never written to. Only kraken
    #: ever resolves an id — 12 of 26 kraken jobs as of 16.09.2026 — so vllm and
    #: trocr never read this.
    #:
    #: Derived from ``registry_root`` unless set explicitly
    #: (ATR_TRAIN_MODELS_CONFIG), for the reason ``ketos`` follows ``venvs_root``
    #: below: two fields naming one place drift apart, and then base models are
    #: read from one registry while models are registered into another. Empty
    #: counts as unset.
    models_config: Path | None = None
    #: Checkpoints go to LOCAL disk, not the job directory on the share. Lightning
    #: saves them via a temp file + rename; with the target on CIFS and the temp
    #: local that rename is cross-device, and the fsspec version datasets<4 pins
    #: (2025.3.0) cannot fall back to a copy — "Upgrade fsspec to enable
    #: cross-device local checkpoints". Local is also plainly right: kraken keeps
    #: the top 10 checkpoints and rewrites them every epoch, which is a lot of
    #: traffic to push over SMB for files we discard once the best is converted.
    checkpoint_root: Path = Path.home() / "atr-cache" / "checkpoints"
    #: Cache the downloaded ground truth in the standard HF cache, or stream it.
    #:
    #: **False (default) — stream from the hub, keeping nothing.** This is the
    #: right default for the selections this trainer exists for: a ~1 TB page
    #: selection's Arrow generation cache is not something this box wants to
    #: materialise, and in cached mode ``datasets`` downloads and converts the
    #: *entire* selection before yielding the first row — 11½ hours with zero
    #: pages written and no progress reported, on the run that exposed it (#60).
    #: Streaming passes rows straight into the kraken page format, so pages
    #: appear within minutes and the page count is a real progress signal.
    #:
    #: True — download and convert once, reuse across runs. Correct at project
    #: scale (a 116 MB dataset fetched repeatedly is waste), and the reasoning
    #: that made it the old default. It inverts entirely at terabyte scale, which
    #: is why the default moved rather than the option disappearing.
    cache_datasets: bool = False

    #: Reuse a compiled corpus across jobs (#109). Between 24 August and
    #: 5 September the same four-dataset German corpus was compiled eight times —
    #: 12,300 pages, ~41 GB of arrow, ~2.5 hours each — five of those differing
    #: from their predecessor only in a hyperparameter the *train* stage reads.
    #: The cache is keyed on the dataset selection (see
    #: :mod:`atr_training.artefact_cache`), so those five are one compile.
    artefact_cache: bool = True
    #: Deliberately **not** under ``jobs_root``: the job directory is the wrong
    #: home for something meant to outlive the job, and cleaning up finished jobs
    #: — which is how 221 GB of dead arrows were removed on 2026-09-08 — must not
    #: take the cache with it.
    artefact_cache_root: Path = Path.home() / "atr-cache" / "artefacts"
    #: Size budget, in GB. 0 disables eviction by size (expired unpinned entries
    #: are still dropped). Two corpus-scale artefacts is the intent: the cache
    #: sits on the box's system disk (457 GB free), not on the 12 TB share, so
    #: this is a real constraint rather than a formality.
    artefact_cache_max_gb: int = 100

    # ── executables ───────────────────────────────────────────────────────
    #: Where the per-engine venvs live. Each job is spawned with *its own*
    #: engine's interpreter (see runner_python) — this service never imports an
    #: engine package, so it does not matter which venv it happens to run in.
    venvs_root: Path = REPO_ROOT / ".venvs"
    #: ``ketos`` from the kraken-train venv. Derived from ``venvs_root`` unless set
    #: explicitly (ATR_TRAIN_KETOS).
    #:
    #: It used to be ``REPO_ROOT / ".venvs" / …`` — a field of its own that never
    #: followed ``venvs_root``. Pointing ATR_TRAIN_VENVS_ROOT elsewhere (to reuse
    #: venvs already built) then gave a runner from the new tree and a ketos from
    #: the old one, and the compile stage failed long after submit. It went
    #: unnoticed on idhefix only because ``<repo>/.venvs`` always existed. Found
    #: by the independent check of the move (#3); it predates the move.
    ketos: Path | None = None

    @model_validator(mode="after")
    def _ketos_follows_the_venvs(self) -> "TrainerSettings":
        if self.ketos is None:
            self.ketos = self.venvs_root / "kraken-train" / "bin" / "ketos"
        return self

    @field_validator("registry_root", "trained_root", mode="before")
    @classmethod
    def _shared_paths_are_absolute(cls, value, info: ValidationInfo):
        # Before, not after: pydantic would turn "" into Path("."), and the
        # message should show what was actually configured. A relative
        # trained_root used to pass here and be refused only by the registration
        # validator, after the whole training run — in a message whose
        # register-by-hand command carried the same relative path.
        if value is None or not str(value).strip() or not Path(str(value)).is_absolute():
            what = ("the gateway's ATR_REGISTRY_ROOT" if info.field_name == "registry_root"
                    else "on the share, at the path idhefix mounts it too")
            raise ValueError(f"{info.field_name} must be an absolute path ({what}), "
                             f"got {value!r}")
        return value

    @field_validator("models_config", mode="before")
    @classmethod
    def _empty_models_config_is_unset(cls, value):
        # `ATR_TRAIN_MODELS_CONFIG=` in .env means "derive it", not the cwd.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _models_config_follows_the_registry(self) -> "TrainerSettings":
        if self.models_config is None:
            self.models_config = self.registry_root / CURATED_FILENAME
        return self

    def runner_python(self, engine: str) -> Path:
        """Interpreter for ``engine``'s detached runner."""
        return runner_python(engine, self.venvs_root)

    # ── the promotion gate (#36) ──────────────────────────────────────────
    #: The gate posts one held-out page here. Through the gateway, not straight
    #: to the engine: "can the serving box serve it" is a question about the path
    #: real clients take.
    #:
    #: The default below is the loopback of the shared box and is wrong on
    #: asteraix, where the gateway is a second machine. ``.env`` sets
    #: ``ATR_TRAIN_GATEWAY_URL=http://130.92.59.240:8200``; without it the gate
    #: simply fails and the model stays registered-but-disabled.
    gateway_url: str = "http://127.0.0.1:8200"
    #: The gateway's own ``ATR_API_KEY`` on idhefix, under this name here — not
    #: ``api_key`` above, which guards the other direction (#9). Empty disables
    #: the gate, which leaves models registered-but-disabled rather than wrongly
    #: advertised.
    gateway_api_key: str = Field(default="", repr=False)
    #: How long the gate keeps asking while the gateway answers ``404 unknown
    #: model``, and how far apart. The gateway reads ``trained/`` at most once per
    #: its ``registry_reload_interval_s`` (5 s), in a thread, and answers the
    #: request that started the read from what it knew before; the CIFS
    #: attribute cache (``actimeo``) lags on top. So the first request after a
    #: registration is always a 404 (#14 review, against the gateway's own app),
    #: and 10 s apart is wide enough that each retry finds the previous one's read
    #: finished. 90 s is nine retries: a gateway that has not seen the file by
    #: then is looking somewhere else.
    gateway_registry_wait_s: float = Field(default=90.0, ge=0)
    gateway_registry_retry_s: float = Field(default=10.0, gt=0)

    # ── guards (serving-atr-inference/docs/TRAINING_PLAN.md §5) ─────────────────────────────────
    #: PHYSICAL GPU index. Both of asteraix's cards are free for training and a
    #: job takes this one by convention (#12 allocates a card per job);
    #: nvidia-smi enumerates physically and ignores CUDA_VISIBLE_DEVICES, so this
    #: is the number preflight queries. The child gets CUDA_VISIBLE_DEVICES=<gpu>,
    #: which makes it cuda:0 inside the process.
    gpu: int = 1
    #: Headroom a kraken run needs (batch 256 through 3× Lbx256).
    min_free_vram_mb: int = 12000
    #: A QLoRA fine-tune of an 8B Qwen3-VL is a different order of appetite: ~6 GB
    #: of 4-bit weights, plus activations for a 4 k-token page sample and paged
    #: optimizer state. Checked instead of ``min_free_vram_mb`` for vllm jobs, so a
    #: VLM job queues rather than OOMing on a card that would have fit a kraken run.
    vlm_min_free_vram_mb: int = 24000
    #: Headroom demanded in the job store — on asteraix that is the research
    #: share, not local disk. Never materialize a dataset into the last of it.
    min_free_disk_gb: int = 50
    #: Pages materialized before a chunk is compiled and deleted (#39). 0 = off,
    #: which materializes the whole selection first — right for the 238-page test
    #: case, impossible for the full corpus: 548,322 pages is ~6.96 TB of pages on
    #: top of a ~6.6 TB hub cache, on a share with ~1.3 TB free (16.09.2026). With chunking on,
    #: peak page-disk is one chunk instead of the whole selection.
    chunk_pages: int = 0

    #: Publish a finished model to the Hub when its character accuracy reaches
    #: this, in percent. **0 disables it**, and that is the default: uploading is
    #: outward-facing and the hub keeps history, so it is opted into deliberately.
    #: Repos are created **private** — `publish.py` takes no `--public` from
    #: automation, and no licence is ever invented (#88).
    auto_publish_min_accuracy: float = 0.0
    #: Where auto-published models go. Ignored when the threshold is 0.
    auto_publish_org: str = "dh-unibe"

    #: Training and inference do not share the card politely; one job at a time.
    max_concurrent: int = 1
    #: How often the scheduler reconciles jobs and starts a queued one.
    poll_interval_s: int = 10
    #: Lines of a stage log kept on a failed job record.
    log_tail_lines: int = 50
    #: Passed to every spawned training process. Empty leaves the allocator alone.
    cuda_alloc_conf: str = "expandable_segments:True"

    def min_free_vram_for(self, engine: str) -> int:
        """VRAM a job of this engine must find free before it may start."""
        return self.vlm_min_free_vram_mb if engine == "vllm" else self.min_free_vram_mb

    def env_for_child(self) -> dict[str, str]:
        """Environment overrides for a spawned training process.

        ``expandable_segments`` because the allocator's fixed-size segments
        fragment badly under this workload: at the OOM that killed
        `20260908T101611Z-qwen3vl-german-pages-v1`, **5.72 GiB were reserved by
        PyTorch but unallocated** — most of the 8.16 GiB the run then could not
        find. It is not a fix for that failure (one 8 GiB allocation is one
        allocation however the heap is arranged, which is what #110's compile-time
        cap addresses), but it is free, and the hand-run sweep on this box had
        already been setting it.
        """
        return {
            "CUDA_VISIBLE_DEVICES": str(self.gpu),
            "PYTORCH_CUDA_ALLOC_CONF": self.cuda_alloc_conf,
        }


_settings: TrainerSettings | None = None


def get_settings() -> TrainerSettings:
    global _settings
    if _settings is None:
        _settings = TrainerSettings()
    return _settings
