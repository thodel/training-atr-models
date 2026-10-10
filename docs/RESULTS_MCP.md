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
| `queue()` | **UBELIX:** running and pending Slurm jobs, every reason explained (`MaxCpuRunMinsPerUser`, `ReqNodeNotAvail` with the named node's state and GRES), and for a running job its last `k/n` counter and a verdict: `fits`, `at risk` (within 15 % of the wall), `will hit the wall` |
| `live(host)` | **both stores:** records with a runner or waiting for one (`queued` … `registering`), from the record and the stage log: status, stage, pid, cards, epoch and steps, `val_accuracy`, peak GPU memory, the last `k/n` counter in `logs/<stage>.log` (else `runner.log`), when the log last moved. asteraix's `queue`, since it has no Slurm; `host="asteraix"` filters. A dead runner leaves its record live until the service (asteraix) or `close_job` (UBELIX) closes it: read `updated_cest` and `log_modified_cest` |
| `finished(days)` | **UBELIX:** `sacct` since `now-<days>days`; a failed job carries the cause line from the END of its log, a job under 10 s is `short_run` (a requeue that found the record finished, #153), `code_drift` lists stages that ran on another commit than the job was created with |
| `job(job_id)` | one `job.json` from whichever store holds it, summarised, with `host`, times in Europe/Zurich, the draw fingerprint and the Slurm ids from the registration note |
| `results(status, granularity, base_model)` | the metrics table over every record of both stores, with `host` and `engine` per row: CER, WER, `length_ratio`, `truncated_cer`, samples, `draw_md5` |
| `draw(job_id)` | md5 and size of `data/val_eval.jsonl`, and every job in either store sharing it, with its host — the answer to "comparable?" (#108) |
| `prepared()` | corpora built, GPU stage never finished, in either store; the scratch purge date of their data (null on the share); `superseded_by` when a completed job of the same `model_id` exists |
| `deadlines()` | artefacts with expiry (7 days after `built_at` unless pinned), claimant status, `at_risk`; the 30-day scratch purge; home usage |
| `log(slurm_job_id, lines)` | head and tail without DEBUG and progress-bar noise, `last_raw`, and notes: code pin, drift lines, cause, last progress |
| `report(evalset, tag)` | benchmark reports on a shared set, by CER, with `load_in_4bit` visible |
| `slurm_job(slurm_job_id)` | `scontrol show job`, the state of any node named unavailable, the log's notes; falls back to accounting once Slurm forgot the job |
| `checkout()` | HEAD against `origin/main`, ahead/behind, dirty files of the UBELIX checkout |

Names are an interface: add tools, never rename one.

Times: Slurm prints local time and is passed through. `job.json` carries UTC;
every converted field ends in `_cest` and carries its label (`CEST`/`CET`).

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

On tei behind nginx (#156), as streamable HTTP mounted at its public path:

```bash
python -m atr_results_mcp --http --port 8012 --path /mcp/atr-results/mcp
```

That deployment needs an SSH key from the serving box to submit02, which does
not exist yet (#156 lists it); until then the laptop runs it.

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
