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
| `queue()` | **UBELIX:** running and pending Slurm jobs, every reason explained (`MaxCpuRunMinsPerUser`, `ReqNodeNotAvail` with the named node's state and GRES), and for a running job its last `k/n` counter — from the Slurm stdout, else from the record's stage log (`stage_log`, see *Progress* below) — and a verdict: `fits`, `at risk` (within 15 % of the wall), `will hit the wall`, `unknown` — with the sum written out in `why`, the stages still ahead added from a measured sibling (`following`, see *The wall* below), `wall.on_timeout` read from the batch file (what the wall does to this job), and `attempt` from the log's attempt line |
| `live(host)` | **both stores:** records with a runner or waiting for one (`queued` … `registering`), from the record and the stage log: status, stage, pid, cards, epoch and steps, `val_accuracy`, peak GPU memory, the last `k/n` counter in `logs/<stage>.log` (else `runner.log`), when the log last moved. asteraix's `queue`, since it has no Slurm; `host="asteraix"` filters. A dead runner leaves its record live until the service (asteraix) or `close_job` (UBELIX) closes it: read `updated_cest` and `log_modified_cest` |
| `finished(days)` | **UBELIX:** `sacct` since `now-<days>days`; a failed job carries the cause line from the END of its log, a job under 10 s is `short_run` (a requeue that found the record finished, #153), `code_drift` lists stages that ran on another commit than the job was created with |
| `job(job_id)` | one `job.json` from whichever store holds it, summarised, with `host`, times in Europe/Zurich, the draw fingerprint and the Slurm ids from the registration note; for a UBELIX record in `training`/`queued`, `resubmit` — the pinned command that puts it back in the queue, or `command: null` with `why` (see *The resubmission* below) |
| `results(status, granularity, base_model)` | the metrics table over every record of both stores, with `host` and `engine` per row: CER, WER, `length_ratio`, `truncated_cer`, samples, `draw_md5`, and `runtime` — the image and transformers the number was measured in, from the test (else train) stage's `runtime` field; `unknown` on every record from before the field (#206), which is not `no container` |
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
  this attempt's rate and is the one to trust. A counter without a tqdm
  estimate on a resumed attempt (`attempt` > 1) is marked `lower_bound`: the
  sum is still written out, but a `fits` from it is reported as `at risk`,
  because a floor cannot prove a fit.
- **The stages still ahead in the same wall** (`test` and `register` after
  `train`, `compile` after `prepare`) are added from a measured sibling: the
  most recently completed job of the same granularity in the same store, of
  the same base model when one exists, with the durations from its
  `stages[]`. `following` names that job (`from`, `match`, `seconds`). Nothing
  is estimated: without a sibling, `uncounted_stages` and `why` say which
  stages the sum is short of, and a sum above 70 % of the wall is `at risk`,
  not `fits`. The TIMEOUT of 05.10.2026 fell in a page job's test stage, which
  takes about an hour; a `fits` from the `train` counter alone was wrong.
- **What the wall does** is `wall.on_timeout`, read from the batch file that
  carries the job's name (`#SBATCH --job-name`), never assumed (#226):
  `train.sbatch` asks for `--signal=B:TERM@120` and its trap runs `scontrol
  requeue`, so the same Slurm job goes back in the queue and the trainer
  resumes from its newest checkpoint; `--requeue` alone covers a preemption
  and not a wall; the scoring and rescoring files have neither, and for them
  `TIMEOUT ends the job`. `attempt` (from `== attempt N — job …`) says which
  attempt is running, so a requeue is read, not inferred. Measured on
  12.10.2026, `17890142`: SIGTERM at 13:16:10 CEST (120 s before the wall),
  `== SIGTERM … — requeueing`, attempt 2 on the same Slurm id at 13:20:07 —
  the wall requeues; the trap's second firing printed `(requeue refused)`,
  which is the already-requeued job, not a failure. At 14:42 the same record
  was submitted again by hand, and two attempts ran on one checkpoint
  directory: the case `resubmit` answers with `command: null`.

## The resubmission

The one command the daily report could not run on 12.10.2026 (#226): a
training that the wall, a preemption that did not requeue, a node or a hand
stopped, and that nothing in Slurm is still carrying. `job(job_id).resubmit`
writes it out for a UBELIX record in `training` or `queued`:

```
cd /storage/homefs/th19c587/training-atr-models && ATR_CODE_COMMIT=<commit of the last train stage> JOB_ID=<id> ubelix/submit.sh ubelix/train.sbatch -- --export=ALL,SIF=<the container the last attempt printed>
```

- **Pinned to the commit the last attempt ran**, not to HEAD. `submit.sh` pins
  to HEAD at submission time, and every drift line in the records so far
  (`stage train: running X, but the job was created with Y`) came from a
  resubmission by hand that let it. Dropping `ATR_CODE_COMMIT` to pick up a fix
  is a decision, and `why` says so.
- **The attempts** are found the other way round from the registration note,
  which exists only once a job completed (#17): every `~/ubelix/logs/*.out`
  written since the record was created whose attempt line names it, with the
  state from one `sacct -j`. `command` is `null` while one is RUNNING or
  PENDING (two attempts on one checkpoint directory), and after a FAILED or
  OUT_OF_MEMORY attempt, which carries its cause and needs a person.
- **`kind`** is `resume` when `checkpoint-N` exists under the record's
  `checkpoint_dir` (`resume_from` names it), `first train submission` for a
  corpus that never ran on Slurm, `restart without checkpoint` otherwise.
- **`unattributed_pending`** lists PENDING jobs without a log yet: a fresh
  submission has no attempt line, so it cannot be tied to a record, and it is
  said rather than assumed away (#165).
- **`checkout_behind`** is the count against `origin/main` as of the last
  fetch (`checkout()` fetches); `submit.sh` refuses a checkout that is behind,
  and pulling is a write that stays with a person.

Nothing here submits. The field is the command, checked, for the person or
the chat session with VPN who runs it; the daily report prints it.

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

**Not from a claude.ai cloud session.** The connector dialog there speaks OAuth
only and cannot send the `auth_basic` header, so a cloud session cannot add this
server. For that case the ATR operations MCP on tei carries three read-only
tools, `ubelix_queue`, `ubelix_job` and `ubelix_record` (agentic_historian#674,
since 21.10.2026), with their own key and forced command on UBELIX
(`~/bin/atr-mcp-ro`). They answer queued, running, preempted and the log tail;
the wall verdicts, `live`, `finished`, `deadlines` and the share store stay here.
An OAuth door on this server would make those three redundant.

## The third state: a failed look is never a zero (#165)

In the night of 06.10.2026 two watchers printed measurements that were none:
`0 of 16 settled` from a `grep -c` over the empty output of a failed `ssh`, and
an exit 0 with no line about a job that had finished. Both turned "could not
look" into "nothing there". Every answer here keeps the two apart:

| where a look can fail | what the answer says |
|---|---|
| the ssh, or the probe itself (`squeue`, `sacct` not answering) | `{"error": …, "cmd": …}` — never `jobs: []`; `main()` exits 1 |
| a job store that is not mounted | `stores.<name>.mounted: false` with a note; its rows are absent, not zero (`results`, `live`, `prepared`) |
| `sinfo` about a node a pending job names | the node reads `state: "unknown"` with `why`; the queue is still the queue |
| a `~/ubelix/logs/*.out` that cannot be read | `resubmit.unreadable_logs` names it and `command` is withheld: it may be the live attempt |
| an id `sacct` does not list | `state: null` with `state_why` — unknown, not "did not run" |
| `git rev-list` in `resubmit` | `checkout_behind: null` with `checkout_behind_why` |
| `df` in `deadlines` | `home: null` with `home_why` |
| a `ubelix/*.sbatch` that cannot be read | `wall.on_timeout.unreadable` names it; `meaning` says unknown, not "none is named" |
| a counter that is not progress | `progress.state: "unattributable"`, verdict `unknown` (#203) |
| no GPU stage carries a runtime | `runtime.known: false` with `why` (#206), which is not "no container" |

A reader that sums or counts over these answers has to read the field beside
the number first. `None` is "nobody looked"; `0` is a count.

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
