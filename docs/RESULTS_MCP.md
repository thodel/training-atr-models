# The results MCP — reading UBELIX and asteraix without a shell

`src/atr_results_mcp/` answers the questions the daily training report asks
(#156, #174), as MCP tools. Read-only: nothing in it submits, cancels, deletes
or registers. It reads **both job stores**: the one UBELIX runs keep under
`/scratch`, and the one the `atr-train` service on asteraix writes on the
research share, which the login node mounts too. Every row says which `host`
trained it.

## Why it exists

The report used to be `ssh` plus hand-written one-liners, and it never ran
unattended: a session in permission mode `default` waits for a click on every
Bash line no allow rule names, and an allow rule names a command *line*. One
changed flag, one `| grep -v` against the SSH banner, and the morning run sat
for eight hours (#174 has the table). An MCP tool is allowed **once, by name**
(`mcp__atr-results__queue`), whatever its arguments.

## Two halves

| half | runs where | Python | job |
|---|---|---|---|
| `probe.py` | on the login node (submit02), sent over `ssh host python3 -` with its source on stdin | **3.9**, standard library only; the suite parses it with `feature_version=(3, 9)` | run `squeue`, `sacct`, `scontrol show`, `sinfo`, `df`, `git`; read `job.json`, logs, `artefact.json` in both stores; print JSON |
| `server.py` | wherever the MCP client is: the laptop today, asteraix behind tei later | 3.11+, the `mcp` extra | one tool per probe question, nothing else |

Shipping the probe on stdin is the deployment story: no package on UBELIX, no
checkout to keep current, no version skew between a tool and what answers it.
`remote.py` tries `ubelix` first and `ubelix-direct` second (`ATR_RESULTS_HOSTS`
overrides); the answer carries `_via`. One call is one SSH round trip, about two
seconds through the jump host.

## The two job stores

| store | root on the login node | written by | records carry |
|---|---|---|---|
| `scratch` | `/scratch/network/users/$USER/runs/jobs` | `ubelix/submit_job.py` and `fanout.py` | `host: ubelix`; purged after 30 days |
| `share` | `/storage/research/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs` (`ATR_RESULTS_SHARE` overrides the share root) | the `atr-train` service on asteraix | `host: asteraix`; a record without `host` is the retired idhefix trainer's (#15); never purged (#183) |

#156 measured `find ~/atr-cache -name job.json` on asteraix, found nothing and
concluded that asteraix had no job store. It looked in the wrong place: every
job the service accepts has a record in the shared store, in the same format
as a UBELIX record. What has no record are the hand-run measurements
(`scripts/measure_*.py`), and those still answer nothing here.

Every record-reading answer carries `stores`: per store its root, whether it
was found, and how many records it holds. A share that is not mounted reads as
*not found*, never as zero asteraix rows (#165).

## The tools

| tool | answers |
|---|---|
| `queue()` | **UBELIX:** running and pending Slurm jobs, every reason explained (`MaxCpuRunMinsPerUser`, `ReqNodeNotAvail` with the named node's state and GRES), and for a running job its last `k/n` counter — from the Slurm stdout, else from the record's stage log (`stage_log`, see *Progress* below) — and a verdict: `fits`, `at risk` (within 15 % of the wall), `will hit the wall`, `unknown` — with the sum written out in `why`, the stages still ahead added from a measured sibling (`following`, see *The wall* below) |
| `live(host)` | **both stores:** records with a runner or waiting for one (`queued` … `registering`), from the record and the stage log: status, stage, pid, cards, epoch and steps, `val_accuracy`, peak GPU memory, the last `k/n` counter in `logs/<stage>.log` (else `runner.log`), when the log last moved. asteraix's `queue`, since it has no Slurm; `host="asteraix"` filters. A dead runner leaves its record live until the service (asteraix) or `close_job` (UBELIX) closes it: read `updated_cest` and `log_modified_cest` |
| `finished(days)` | **UBELIX:** `sacct` since `now-<days>days`; a failed job carries the cause line from the END of its log, a job under 10 s is `short_run` (a requeue that found the record finished, #153), `code_drift` lists stages that ran on another commit than the job was created with |
| `job(job_id)` | one `job.json` from whichever store holds it, summarised, with `host`, times in Europe/Zurich, the draw fingerprint and the Slurm ids from the registration note |
| `results(status, granularity, base_model)` | the metrics table over every record of both stores, with `host` and `engine` per row: CER, WER, `length_ratio`, `truncated_cer`, samples, `draw_md5` |
| `draw(job_id)` | md5 and size of `data/val_eval.jsonl`, and every job in either store sharing it, with its host — the answer to "comparable?" (#108) |
| `prepared()` | corpora built, GPU stage never finished, in either store; the scratch purge date of their data (null on the share); `superseded_by` when a completed job of the same `model_id` exists |
| `deadlines()` | artefacts with expiry (7 days after `built_at` unless pinned), claimant status, `at_risk`; the 30-day scratch purge; home usage |
| `log(slurm_job_id, lines)` | head and tail without DEBUG and progress-bar noise, `last_raw`, and notes: code pin, drift lines, cause, last progress (with the stage-log fallback and `job_id` from the attempt line) |
| `report(evalset, tag)` | benchmark reports on a shared set, by CER, with `load_in_4bit` visible |
| `slurm_job(slurm_job_id)` | `scontrol show job`, the state of any node named unavailable, the log's notes; falls back to accounting once Slurm forgot the job |
| `checkout()` | HEAD against `origin/main`, ahead/behind, dirty files of the UBELIX checkout |

Names are an interface: add tools, never rename one.

Times: Slurm prints local time and is passed through. `job.json` carries UTC;
every converted field ends in `_cest` and carries its label (`CEST`/`CET`).

## Progress

A `progress` field is in one of three states (#203):

- `null` — no `k/n` counter in the log at all.
- `state: found` — a counter that is the job's progress: `done`, `total`,
  the `line` it came from and its `source`. The only state a wall projection
  uses.
- `state: unattributable` — the only counters come from a known false
  source: the retry count of a hub request (`Retry 2/5`), a model load
  (`Loading weights: 644/677`), the worktree checkout. The `line` and a `why`
  are returned, `done` is deliberately absent, and the wall verdict is
  `unknown` with that `why`. "Found something that looks like a counter" is
  not "measured": on 09.10.2026 a retry count projected to "fits, 30.1 % of
  the wall". The list of false sources is in `probe._NOT_PROGRESS`; it grows.

`source` says where the counter was read: `slurm` for the job's stdout under
`~/ubelix/logs/`, else the stage log's name (`train.log`, `runner.log`). A VLM
run writes its step counter only to `<jobs_root>/<job_id>/logs/train.log`;
the attempt line in the stdout (`== attempt N — job <id> (<status>)`) names the
record, the record names its stage, and the probe reads that file when the
stdout carries no attributable counter. The answer then also carries
`job_id`, `stage_log` and `stage_log_modified_cest`.

## The wall

`wall.why` writes the sum out, so a verdict is checkable without the box
(#216):

```
train: 56577 s + rest 18052 s (tqdm 10321/13704) + test ≈ 420 s + register ≈ 1 s (from 20260925T045500Z-ladder-med-qwen35-27b) = 75050 s of 86400
```

- **The rest of the running stage** is tqdm's own `<remaining` when the
  counter line carries one (`method: tqdm`), else `elapsed × (total − done) /
  done` (`method: ratio`). After a resume the HF trainer's bar starts at the
  checkpoint's `global_step`: `done` holds earlier attempts' steps, `elapsed`
  only this one, and the ratio comes out too small. tqdm's estimate is from
  this attempt's rate and is the one to trust.
- **The stages still ahead in the same wall** (`test` and `register` after
  `train`, `compile` after `prepare`) are added from a measured sibling: the
  most recently completed job of the same granularity in the same store, of
  the same base model when one exists, with the durations from its
  `stages[]`. `following` names that job (`from`, `match`, `seconds`). Nothing
  is estimated: without a sibling, `uncounted_stages` and `why` say which
  stages the sum is short of, and a sum above 70 % of the wall is `at risk`,
  not `fits`. The TIMEOUT of 05.10.2026 fell in a page job's test stage, which
  takes about an hour; a `fits` from the `train` counter alone was wrong.

## Running it

Locally, as the MCP client's stdio server (what the daily report uses):

```bash
claude mcp add -s user atr-results -- uv run --directory /Users/TH_1/Documents/Repo/training-atr-models --extra mcp python -m atr_results_mcp
```

One answer without an MCP client, for debugging:

```bash
uv run --extra mcp python -m atr_results_mcp --call queue
uv run --extra mcp python -m atr_results_mcp --call job '{"job_id": "20260930T170315Z-ladder-xix-gemma4-12b"}'
```

Behind tei (#214), as streamable HTTP mounted at its public path. The process
runs on asteraix as the user unit `atr-results-mcp.service`, from its own
checkout `~/Repo/atr-results-mcp` and venv `~/.venvs/atr-results-mcp` (the
training checkout is never pulled under a running job):

```bash
python -m atr_results_mcp --http --host 127.0.0.1 --port 8013 --path /mcp/atr-results/mcp --public-host tei.dh.unibe.ch
```

`--public-host` names the Host header nginx forwards. The SDK's DNS-rebinding
protection allows only loopback Hosts by default, and without the flag every
proxied request answers `421 Misdirected Request`; the flag keeps the
protection on and lets that one name through. Port 8013 because 8012 is taken
on tei and the tunnel maps the same number on both boxes.

The chain: nginx `location /mcp/atr-results/mcp` (`auth_basic`, user
`atr-results`, key in `/etc/nginx/.htpasswd-atr-results`) → `127.0.0.1:8013`
on tei → `mcp-tunnel-asteraix@8013` (dh user unit, key restricted on asteraix
to `permitopen="127.0.0.1:8013"` with `command="/bin/false"`) → the unit on
asteraix → `ssh ubelix-direct` with `~/.ssh/id_ed25519_ubelix`, whose
`authorized_keys` line on UBELIX forces `~/ubelix/atr_results_probe_only.sh`:
only `python3 - <cmd> <json>` passes, everything else is refused and logged.

Update on asteraix: `git -C ~/Repo/atr-results-mcp pull --ff-only &&
systemctl --user restart atr-results-mcp`. Nothing on UBELIX needs updating.

## What it does not do

- **asteraix's hand runs.** `scripts/measure_*.py` write hand-named JSON under
  `~/atr-cache` on asteraix's local disk, which no other host mounts. Nothing
  here reads them; note such a number by hand wherever it is quoted. The
  service's jobs are covered (above).
- **asteraix's cards.** `GET /gpu` on the trainer, proxied by the gateway as
  `/train/gpu`, is the only reader of what the cards hold; `live` reads the
  record's `gpus` and `peak_gpu_mib`, not `nvidia-smi`.
- **GitHub.** Issues and PRs stay with `gh`, whose prefix rules
  (`Bash(gh issue *)`, `Bash(gh pr *)`) hold as long as the command stands alone.
- **Writes.** `git fetch` in `checkout()` touches `.git`, not the tree. That is
  the only thing it changes anywhere.
