# Eval harness

Runs the gateway's `/recognize` over a folder of images and reports **CER/WER per
model**. Ports the `os-vlm-tester` result schema (`outputs/<model>/<image>.json`
+ `outputs/index.jsonl`) but calls the live ATR API instead of loading models
locally — so it measures the **deployed** system end to end: segmentation,
recognition, and whatever the gateway does in between.

It came here from `serving-atr-inference` with #11. It has **no** import from the
serving half — its input is a folder of images and its output is JSON — so the
seam it crosses is one HTTP route (`/recognize`) and one key, the third of the
three edges in [`docs/INFRASTRUCTURE.md`](../docs/INFRASTRUCTURE.md).

## Usage

The harness needs `httpx` and nothing else. On asteraix that means
`kraken-train`, the service venv — it is the only one carrying `httpx`, and no
venv has this repo installed, so `eval/metrics.py` puts `src/` on the path
itself.

```bash
.venvs/kraken-train/bin/python eval/run_eval.py \
    --images-dir data/test \
    --models kraken-catmus_medieval,party,qwen3vl-8b-hebrew \
    --gt-dir data/test/gt
```

- `--models` (comma-separated) or `--models-file` (one id per line).
- `--gateway` defaults to `$ATR_TRAIN_GATEWAY_URL`, the same variable the
  promotion gate reads; `.env` sets it to `http://130.92.59.240:8200` (idhefix).
  **There is no loopback fallback.** On the shared box there was one, and it was
  right; here it would post to asteraix's own `:8200`, where nothing listens, and
  report a page of connection errors as if the models had failed. Unset is
  answered with the variable's name.
- `--api-key` defaults to `$ATR_TRAIN_GATEWAY_API_KEY`, else `$ATR_API_KEY` —
  the gateway's own key, which is the one that opens `/recognize` (#9 keeps the
  two directions apart).
- `--recursive`, `--max-images N`, `--out-dir` (default `eval/outputs`).

## Ground truth (optional, enables CER/WER)

For each `image.png`, the harness looks (in `--gt-dir`, else alongside the image) for:
`image.txt`, `image.gt.txt`, or `image.xml` (PAGE-XML — line text is extracted in
document order). Without ground truth it still records transcriptions + timing.

## Output

- `outputs/<model>/<image>.json` — per-image record (text, engine, timings, cer/wer).
- `outputs/index.jsonl` — one record per line.
- `outputs/summary.json` + a printed table — per-model CER/WER and latency.

`CER` in that table is the **corpus** rate, total errors over total reference
characters — the definition `ketos test` uses, so a kraken CER and a VLM CER are
the same kind of number. `~CER` beside it is the mean of per-sample rates, kept
only for spotting outlier pages: a 5-character line read wrong scores 1.0 there
and outweighs a 200-character page read almost perfectly.

CER and WER themselves live in
[`src/atr_training/textmetrics.py`](../src/atr_training/textmetrics.py);
`eval/metrics.py` re-exports them and adds the ground-truth loading. One
implementation, so a CER printed here and a CER stored on a training job are the
same number computed the same way. `textmetrics` had no other consumer on the
serving side, which is why moving `eval/` settled the question of a shared
metrics package by removing it.

## What this harness answers, and what it does not

It is the only thing here that measures **through `/recognize`**: many model ids
against one folder of images, over the path a real caller takes. That makes it
the comparison between a freshly promoted model and the ones already served —
`--models kraken-thun-v1,kraken-catmus_medieval` is the whole experiment.

It does not answer the granularity question. A model trained on line crops can
read lines well and whole pages not at all (`qwen3vl-medieval-german-v3`: CER
0.111 on held-out lines, every one of 14 pages read as two characters). That
needs the same pages asked four ways, which is
[`scripts/eval_granularity.py`](../scripts/eval_granularity.py), and it is the
tool #55 and #61 are written against — not this one.

**Never point either at the gateway while it serves callers.** Card 1 of idhefix
holds the engines and one VLM; asking for a second model evicts the one a caller
is using.
