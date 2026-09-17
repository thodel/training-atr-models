# Operating atr-train on asteraix

Procedures for the training machine. The reasons behind them are in
[INFRASTRUCTURE.md](INFRASTRUCTURE.md). The gateway and engines on idhefix are
operated from serving-atr-inference; see its
[docs/INFRASTRUCTURE.md](https://github.com/thodel/serving-atr-inference/blob/main/docs/INFRASTRUCTURE.md).

Every command below runs **on asteraix** unless it says otherwise. None of them
prints a key. Keep it that way: never paste a key into a command line, an issue
or a chat. The commands read keys into a shell variable straight from `.env`.

| What | Where |
|---|---|
| Checkout | `~/Repo/training-atr-models` |
| Settings | `~/Repo/training-atr-models/.env` (mode 600, never committed) |
| Unit | `~/.config/systemd/user/atr-train.service`, copied from `deploy/systemd/` |
| Job store | `/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs` (called `$JOBS` below) |
| Registry | `/mnt/wbkolleg_dh_1/Textrecognition_Training/registry` |
| Checkpoints, TMPDIR, corpus cache | `~/atr-cache/{checkpoints,tmp,artefacts}` |
| `.env` backups | `~/atr-cache/env-backups/` |

```bash
cd ~/Repo/training-atr-models
JOBS=/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs
KEY=$(grep '^ATR_TRAIN_API_KEY=' .env | cut -d= -f2-)    # for every route but /health
```

## Health

```bash
systemctl --user status atr-train
curl -s localhost:8204/health        # no key: engines, venvs, cards, job counts
```

The job counts in `/health` cover the whole store, including other hosts'
records.

**The gateway's `/health` does not test the key or the allowlist.** On
idhefix, `curl -s localhost:8200/health` lists the trainer as `reachable` for
any answer below 500. `/health` here needs no key, and a caller outside the
allowlist gets a 403, which is below 500. A trainer that refuses every
`/train/*` call can therefore still read as reachable. Only an authenticated
call through the gateway tests the whole path. Run it on idhefix:

```bash
GW=$(grep '^ATR_API_KEY=' ~/Repo/serving-atr-inference/.env | cut -d= -f2-)
curl -sS -w '\nHTTP %{http_code}\n' -H "X-API-Key: $GW" localhost:8200/train/gpu
```

**The bot says the trainer is unreachable.** It has two messages, and they
mean different things:

| The bot says | What it means | Check |
|---|---|---|
| `/atr_gpu`: "Trainer nicht erreichbar — nichts ist einem Job zugeordnet" | the trainer **did** answer, but it could not read the job store, so no process on the cards is attributed to a job (`job_attribution_available: false`) | here: `ls "$JOBS"`; the journal has "job store unreadable for /gpu" |
| the watcher: "Der Trainingsserver antwortet seit … nicht" | its calls to `/train/jobs` and `/train/gpu` through the gateway have failed for 30 min. The gateway may be the part that is down | on idhefix: `systemctl --user status atr-gateway`, then the authenticated call above |

What the authenticated call answers:

| Answer | Cause | Look at |
|---|---|---|
| JSON with `cards`, `HTTP 200` | the path works | |
| no connection to `localhost:8200` | the gateway is down | `journalctl --user -u atr-gateway` on idhefix |
| 502 "training service unreachable at URL" | the connection was refused: `atr-train` is down, or idhefix's `ATR_TRAIN_URL` names the wrong port | `systemctl --user status atr-train` here; `ATR_TRAIN_URL` in idhefix's `.env` |
| 502 naming `ATR_TRAIN_API_KEY` or `ATR_TRAIN_ALLOWED_CLIENTS` | the two `.env` files disagree | here, `journalctl --user -u atr-train` has "refused … with 401" or "with 403"; compare the keys as in [Editing .env](#editing-env) |
| 504 "could not connect within 5s" | nothing answered the connection: asteraix is down or off the network, or `ATR_TRAIN_URL` names the wrong host | whether asteraix is up (`ssh asteraix` from the laptop) |
| 504 "did not answer within 20s" | the trainer is slower than the gateway waits, for example on a slow share | the journal here; `ls "$JOBS"` |
| a text that starts "training service at URL:", any status | the trainer's own error, passed on with its status. A 503 means its settings are incomplete, and the text names what is missing | the text |

## Deploy

1. **Is one of this host's jobs running?** This reads the share, no key
   needed. It prints `[]` when nothing of this host is live:

   ```bash
   python3 -c 'import glob,json,sys; print([(j["id"], j["status"]) for p in sorted(glob.glob(sys.argv[1]+"/*/job.json")) if (j:=json.load(open(p))).get("host")=="asteraix" and j["status"] not in ("completed","failed","cancelled")])' "$JOBS"
   ```

2. **If a job runs**, the restart will not stop it (`KillMode=process`). But the
   runner reads code from disk later: each train/eval step starts as a new
   process (`python -m vlm_train_svc.train_qlora`, `…evaluate_qlora`,
   `trocr_train_svc.train_trocr`, `…evaluate_trocr`), and imports inside
   functions (auto-publish) happen when they run. Before pulling, look at what
   would change under the run:

   ```bash
   git rev-parse HEAD                  # the commit the run started from, if nothing was pulled since
   git fetch origin
   git diff --stat HEAD origin/main -- engines/ src/atr_training/
   ```

   If the diff touches a stage the run has not reached yet, wait for the run,
   or read the diff and decide that the old run and the new code fit together.

3. **Pull, install, restart:**

   ```bash
   git pull --ff-only
   bash scripts/install_user_unit.sh --no-start && systemctl --user restart atr-train
   ```

   `install_user_unit.sh` refuses to install if the checkout is not at
   `~/Repo/training-atr-models`, if `.env` or the kraken-train venv is
   missing, or if the launcher would refuse the unit's bind. Keep the `&&`.
   Without it, a refused check is followed by a restart with settings the
   launcher refuses: the unit exits 2, `RestartPreventExitStatus=2` keeps it
   down, and nothing new is scheduled (running jobs go on). If the check
   fails, fix `.env` (or restore the backup) before restarting.

4. **If a `requirements.txt` changed:** `bash scripts/make_venvs.sh <venv>`,
   then `bash scripts/check_venvs.sh`. The script installs into the existing
   venv in place, so do not rebuild the venv of a job that is running.

5. **Check:** `systemctl --user status atr-train` and
   `curl -s localhost:8204/health`. Then, on idhefix, run the authenticated
   call from [Health](#health). It must answer `HTTP 200`. The gateway's
   `/health` is not enough: it reads a trainer that refuses the gateway as
   reachable.

What a restart, a reboot or a `.env` edit does to running jobs is listed in
[INFRASTRUCTURE.md](INFRASTRUCTURE.md#deploying-and-restarting).

## Cancel, resubmit, resume

**Cancel** this host's job, here or through the gateway
(`POST /train/jobs/<job_id>/cancel` on idhefix:8200):

```bash
curl -s -X POST -H "X-API-Key: $KEY" localhost:8204/jobs/<job_id>/cancel
```

- If the job has not started, it becomes `cancelled` at once. The answer is 409
  "being started right now" if a scheduler is starting it at that moment; ask
  again a few seconds later.
- If the job is running, its process group gets SIGTERM, and the runner writes
  `cancelled`. The response can still show the old status; read the record
  again a moment later.
- If the job belongs to another host, the answer is 409 naming that host. See
  [below](#closing-another-hosts-stuck-record).

`DELETE /jobs/<job_id>` works on terminal jobs only, of any host. It keeps
`job.json`, removes the job's checkpoints only if the job ran on this host
(another host's are on that host's disk), never touches the registered model,
and removes orphaned weights under the conditions in
[INFRASTRUCTURE.md](INFRASTRUCTURE.md#the-life-of-a-job).

**Resubmit.** A `failed` or `cancelled` job never runs again. Submit its
request as a new job:

```bash
python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["request"]))' \
  "$JOBS/<job_id>/job.json" > /tmp/request.json
curl -s -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d @/tmp/request.json 'localhost:8204/jobs?verify_only=true'    # a dry run first
curl -s -X POST -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d @/tmp/request.json localhost:8204/jobs
```

The `model_id` may be reused once no live job of any host carries it. The old
registration is disabled before its weights are replaced.

**Resume.** There is no resume on this machine, only what makes a resubmit
cheaper:

- kraken and vllm: with an identical dataset selection, the corpus cache
  (#109) hands the compiled corpus straight to `train`, and prepare and compile
  are skipped. A corpus built without a pinned `revision` is reused for 7 days
  at most. trocr jobs do not use the cache.
- kraken: `base_model` also accepts a local weights file, so a new job can
  start from saved weights.
- vllm: the failure text of a train stage names what survived under
  `~/atr-cache/checkpoints/<job_id>`: a resumable checkpoint, a recovery
  snapshot, or the best adapter so far. A resubmit gets a new job id and a new
  checkpoint directory, so it does not pick that up by itself.
- The code's own resume path (a runner started again on a job still in
  `training`) is used on UBELIX. Here the next scheduler tick marks such a
  job `failed` once its runner is gone.

## Editing .env

```bash
cd ~/Repo/training-atr-models
install -m 600 .env ~/atr-cache/env-backups/training-atr-models.env.$(date +%Y%m%dT%H%M%S)
${EDITOR:-nano} .env
stat -c %a .env                                   # must print 600
# The launcher judges the new settings; restart only if it accepts them.
bash scripts/install_user_unit.sh --no-start && systemctl --user restart atr-train
```

If the check fails, fix `.env` or restore the backup before restarting. A
restart with refused settings leaves the service down ([Deploy](#deploy),
step 3).

- Keep backups in `~/atr-cache/env-backups/`, never in the checkout. On idhefix
  on 16.09.2026, a backup `.env.bak-<date>` sat in the checkout of a public repo,
  untracked and not ignored. `.env.*` is ignored now, but a backup has no place
  in the checkout.
- The service reads `.env` only when it starts. Running jobs keep the
  environment they were started with.
- A `>>> SHARED <<<` value must change on idhefix in the same step. See the
  [table](INFRASTRUCTURE.md#values-shared-with-idhefix). To compare a key
  across the machines without printing it, run on both hosts:

  ```bash
  grep '^ATR_TRAIN_API_KEY=' .env | cut -d= -f2- | sha256sum
  ```

  For the gate's key, hash `ATR_TRAIN_GATEWAY_API_KEY` here and `ATR_API_KEY`
  on idhefix.

  The file on idhefix is `~/Repo/serving-atr-inference/.env`. A new key comes
  from `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`. On
  16.09.2026 it was generated on asteraix and piped to idhefix without being
  shown, and the two hashes were compared as above.

## Logs

| What | Where |
|---|---|
| the service: scheduler, spawns, reconcile, orphan cleanup | `journalctl --user -u atr-train -f` |
| a runner's stdout and stderr | `$JOBS/<job_id>/logs/runner.out` |
| the runner's own log: prepare, the VLM and TrOCR compile, the gate | `$JOBS/<job_id>/logs/runner.log` |
| stages that run a subprocess | `$JOBS/<job_id>/logs/train.log`, `test.log`, and for kraken `compile.log` |
| why a job failed | `error` and `log_tail` (the last 50 lines) in `$JOBS/<job_id>/job.json` |
| through the gateway | `GET /train/jobs/<job_id>/log?stage=train&lines=N`, `GET /train/jobs/<job_id>/curve` |

## Registering by hand

A job that trained but could not register **fails**, and keeps its weights
(#14). Its `error` reads
`StageFailed in register: the model is trained but NOT registered: …`. It says
why and where the weights are, and it ends with a command:

```bash
python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["error"])' "$JOBS/<job_id>/job.json"
```

The command in that text has this shape. It uses the runner's interpreter and
`src/`, so it can be pasted as it stands:

```text
PYTHONPATH=…/training-atr-models/src …/.venvs/<venv>/bin/python -m atr_training.registration --root <registry root> <<'EOF'
id: <model_id>
engine: <engine>
local_path: <weights directory>
enabled: false
…
EOF
```

1. **Fix the cause first.** The text names it: the share is not mounted,
   `ATR_TRAIN_REGISTRY_ROOT` is not the gateway's `ATR_REGISTRY_ROOT`, or the
   weights are not on the registry's filesystem. In the last case, move them to
   the share and change `local_path` in the command.
2. **Paste the command.** It validates the entry like the runner does and
   writes `registry/trained/<model_id>.yaml` atomically. It prints the path.
3. **The model is now registered but disabled.** The promotion gate did not
   run. Test the model the way the gate does, with the header that lets the
   gateway serve a disabled registration to this request only. A vllm adapter
   must be merged first ([Promoting by hand](#promoting-by-hand), step 1).

   ```bash
   GKEY=$(grep '^ATR_TRAIN_GATEWAY_API_KEY=' .env | cut -d= -f2-)
   head -3 "$JOBS/<job_id>/data/pages_val.lst"     # held-out pages; each image is the same path with .jpg
   curl -s -H "X-API-Key: $GKEY" -H 'X-ATR-Promotion-Gate: 1' \
     -F image=@<page>.jpg -F model=<model_id> http://130.92.59.240:8200/ocr
   ```

   The gateway reads `trained/` at most every 5 s, so a first `404 unknown
   model` is normal: ask again after 10 s. If the answer has non-empty `text`,
   enable the model as in [Promoting by hand](#promoting-by-hand).

A job refused because its `model_id` is a **curated id** registered nothing.
Its text names where the trained weights still are (until the job is deleted)
and says to copy them to a directory named after a new id under the trained
root, then register that id with
`python -m atr_training.registration --root <registry root>`, with the entry as
YAML on stdin.

## Promoting by hand

A registered model that is still `enabled: false` is not in `/models`. That is
where every trocr and vllm model ends, and a kraken model whose gate failed
(`promoted: false`; `promotion_reason` on the record says why). A completed job
has no registration command in its text, and none is needed: the file on the
share is the complete entry, and only `enabled` changes.

1. **vllm only:** merge the adapter first, on idhefix, with
   `scripts/merge_loras.py` from the serving repo. vLLM 0.11 cannot serve the
   adapter before that.
2. **Test it** through the gateway with the gate's header, as in
   [Registering by hand](#registering-by-hand), step 3. The page comes from
   `$JOBS/<job_id>/data/pages_val.lst`. Go on only if `text` is not empty.
3. **Enable it.** This reads the registration as it is, changes only the
   `enabled` line, and writes it back through the same validation and atomic
   write as the trainer. It prints the path:

   ```bash
   REG=/mnt/wbkolleg_dh_1/Textrecognition_Training/registry
   sed 's/^enabled: false$/enabled: true/' "$REG/trained/<model_id>.yaml" \
     | PYTHONPATH=src .venvs/kraken-train/bin/python -m atr_training.registration --root "$REG"
   ```

   The gateway notices the change at its next look at `trained/`. A request
   starts that look, at most every 5 s, so the first `GET /models` may not list
   the model yet; ask again a few seconds later. The comment lines at the top
   of the file are written anew. Do not edit the file in place with an editor:
   the gateway may read it half-written.

## Closing another host's stuck record

This host never starts, judges or signals another host's job (#15). For a
live one, cancel and `DELETE` answer 409, and so does a resubmit of the same
`model_id`. If the owning host no longer runs the job, a person closes the
record: a legacy idhefix record (idhefix's trainer is retired), or a UBELIX
record whose Slurm job is gone. A scancelled preemptable job also stays live,
because its runner took the SIGTERM for a preemption.

The UBELIX case applies only to a record in the shared store. Today, UBELIX
runs keep their records under `/scratch` on UBELIX, which asteraix does not
mount, and on 16.09.2026 the shared store held no `ubelix` record
([INFRASTRUCTURE.md](INFRASTRUCTURE.md#ubelix)).

```bash
cd ~/Repo/training-atr-models     # the host id comes from this .env
PYTHONPATH=src .venvs/kraken-train/bin/python -m atr_training.close_job <job_id> \
  --reason 'ps -p <pid> on idhefix: no such process'    # shows what it would write
PYTHONPATH=src .venvs/kraken-train/bin/python -m atr_training.close_job <job_id> \
  --reason 'ps -p <pid> on idhefix: no such process' --yes
```

- It sends **no signal**: the pid in the record belongs to the other machine.
- A job that never started becomes `cancelled`, anything else `failed`. The
  error says who closed it, on which host, and why.
- It refuses a job of this host (cancel that through the service), a run
  without `ATR_TRAIN_HOST_ID`, a queued job that a scheduler has claimed, and
  a record that changed while the command ran.
- For a UBELIX record, check first with `squeue`/`sacct` on UBELIX that the
  Slurm job is really gone.
