# TrOCR on German handwriting, XVI–XIX (#37)

What the TrOCR line reads, measured against the VLM line arm on lines neither
saw in training. Status 28.10.2026.

## The loss fix came first (#306)

Every TrOCR run before 26.10.2026 trained on the token after next:
`ForCausalLMLoss` shifts the labels, and our collator had shifted them already.
On the same 500 validation lines that took the CER from 0.53 to 0.09–0.11. No
number from before the fix says anything about TrOCR.

## The runs

All on asteraix, one A40 each, batch 8, bf16, `lrate` 5e-5, evaluated with
beam 4, `length_penalty` 1.0 and a trigram block (#305). Times are CET.

| run | base | data | epochs | train lines | steps | training | energy | own val CER (500 lines) |
|---|---|---|---:|---:|---:|---|---:|---:|
| `20261026T071211Z-trocr-german-kurrent-xvi-xix-v2` | `microsoft/trocr-base-handwritten` | ≤ 100 pages per dataset | 3 | 53,008 | 19,878 | 88 min | 431 Wh | 0.1100 |
| `20261026T064540Z-trocr-german-kurrent-xvi-xix-from-kurrent-v2` | `dh-unibe/trocr-kurrent-XVI-XVII` | ≤ 100 pages per dataset | 3 | 53,008 | 19,878 | 86 min | 443 Wh | 0.0889 |
| `20261027T051612Z-trocr-german-kurrent-xvi-xix-full-v1` | `dh-unibe/trocr-kurrent-XVI-XVII` | the VLM line arm's corpus, ≤ 2,000 pages per dataset | 1 | 473,549 | 59,194 | 3 h 29 | 1,205 Wh | 0.1038 |

The "own val CER" columns are three different draws and do not compare across
rows. The full run spent 1 h 28 in `prepare` and 6 h 02 in `compile` before it
trained; the energy column is the whole job.

The full run has the same pages and the same split as
`20261009T150138Z-qwen3.5-4b-german-kurrent-xvi-xix-v1`, the VLM line arm
(Qwen3.5-4B): 13,606 / 1,709 / 1,318 pages train / val / reserved, with the
same text fingerprint on every page (md5 of the sorted, whitespace-normalised
line texts). Both arms saw the same 473,549 lines once.

## T4: both lines on one draw

**The draw** is the validation lines of both runs, less the 51 validation pages
whose text also stands on a training page (duplicates in the corpus). Pages were
shuffled with seed 42 per dataset and taken whole while a dataset stayed at or
under 400 lines; where no page fit, the shortest one was taken. That gives
**5,952 lines on 214 pages from 18 of the 20 datasets**;
`koenigsfelden-adhr-colmar` and `sg-missiven` have no validation page without a
duplicate. Every arm read the same line crops against the same references.

The draw, the raw readings of every arm, the reports and the scripts that built
and scored them are kept on the share under
`eval_sets/t4-trocr-vs-vlm-xvi-xix/` (`t4b_draw.jsonl`, md5 `564a6189`;
`SHA256SUMS` beside them). The draw's image paths resolve under the full TrOCR
run's job directory. TrOCR was read with `trocr_train_svc.evaluate_trocr
--max-samples 100000`; the VLM with the call `rescore_on_shared_draw.py` makes
(`evaluate_cmd`), adapter md5 `6aacb99d025d`, bf16.

| CER | all (214 pages) | 14 clean datasets (164) | without Königsfelden (189) | Königsfelden (25) |
|---|---:|---:|---:|---:|
| VLM `qwen3.5-4b-…-v1` (4 B) | 0.0936 | 0.0983 | 0.0899 | **0.1153** |
| **TrOCR, full corpus** (334 M) | **0.0910** | **0.0786** | **0.0710** | 0.2075 |
| TrOCR, 100 pages per dataset | 0.1251 | 0.0950 | 0.0895 | 0.3319 |
| `trocr-kurrent-XVI-XVII`, untrained | 0.2612 | 0.2240 | 0.2166 | 0.5204 |

"Clean" leaves out the four datasets in the draw that `trocr-kurrent-XVI-XVII`
may have seen in its own pretraining: the two Königsfelden sets, kurrent-xix
and the Zurich Regierungsrat protocols. The 100-page arm trained on 27 of the
draw's pages; its row above includes them, its paired comparison below does not.

**Paired, difference of the pooled CER, 95 % bootstrap over pages** (2,000
resamples, seed 42):

| comparison | Δ CER | 95 % |
|---|---:|---|
| TrOCR full − VLM, all | −0.003 | [−0.017, +0.012] |
| TrOCR full − VLM, clean | **−0.020** | [−0.031, −0.010] |
| TrOCR full − VLM, without Königsfelden | **−0.019** | [−0.029, −0.010] |
| TrOCR full − VLM, Königsfelden | **+0.092** | [+0.039, +0.130] |
| TrOCR full − TrOCR 100 pages (187 pages outside the latter's training) | **−0.044** | [−0.062, −0.030] |
| TrOCR full − untrained | **−0.170** | [−0.193, −0.151] |

### What it says

1. **On the same data the two lines are level overall.** TrOCR has a twelfth of
   the parameters.
2. **Without Königsfelden TrOCR reads better**, by about two points. The largest
   gap is Bullinger (0.173 against 0.303).
3. **On the Königsfelden charters the VLM stays ahead by about nine points.**
   With a ninth of the training lines (T4 of 26.10.2026, 38 pages) the gap was
   about forty (0.495 against 0.100), so most of it was data; what is left is
   the model.
4. **More data helps TrOCR** by 4.4 points against 100 pages per dataset.
5. **Reading cost:** on the same card the draw took TrOCR 35 minutes per arm
   and the VLM 93.

### Limits

* The draw is stratified, at most 400 lines per dataset, not proportional to
  the corpus, so "all" weighs the datasets roughly equally.
* Per dataset there are 1–23 pages; those numbers are tendencies. The full
  table is in the comment on #37.
* Königsfelden is 25 pages from two datasets.
* This is a validation-side draw of a by-page split, not a held-out set by
  document: unknown hands are not in it. See
  [HELDOUT_BOUNDARY.md](HELDOUT_BOUNDARY.md) for how much that flattered the
  medieval page arms.

## Open

* **T5:** serving. `trocr_svc` in serving-atr-inference refuses trained models
  so far, and the promotion gate does not run for trocr
  ([INFRASTRUCTURE.md](INFRASTRUCTURE.md)).
* No peak memory has been measured for a trocr run.
