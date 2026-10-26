# vlm-xix-effective-batch-01

**data** `sha256:2186b1ddf722c29dc7db9f18f4240820ab429f3c1e38282f6d6705d911345d1b` — the four 19th-century corpora of qwen3vl-german-xix-v2 / the 10 % validation partition of that corpus — identical for all three.

**noise floor** 0.1170 — measured over 4 seeds [42, 43, 44, 45] at 3000 steps, commit `d214057`. It is a **lower bound**: rows closer than this are not distinguishable, and rows further apart are not thereby distinguished — no cell's own spread has been measured.

**inside the noise** — `3de4c2e3aef1` at 0.1355 against qwen3vl-german-xix-v2's 0.0533 — 0.0822 worse, and the floor is 0.1170. Not a win and not a loss: this material at this budget cannot tell the two apart, which is a finding about the sweep and not about the model.

**code** — 2 different commits across these runs (`477961941484`, `8ef07661e925`). Rungs trained on different code are not comparable; say which change fell in the middle before reading the ranking.

**metric** cer. **budget** 48,000 training samples at rung 0, ×3 per rung. **corpus** 881,902 training lines.

| rank | config | effective_batch | steps | cer | min | status |
|---|---|---|---|---|---|---|
| **rung 0** | | | | | | |
| =1 | `3de4c2e3aef1` | {'accumulate_grad_batches': 2, 'batch_size': 16, 'lrate': 0.0004} | 1,500 | 0.2233 | 185 | promoted |
| =1 | `0632d5e0fa27` | {'accumulate_grad_batches': 1, 'batch_size': 16, 'lrate': 0.0002} | 3,000 | 0.2275 | — | eliminated |
| =1 | `a046cc9d3020` | {'accumulate_grad_batches': 4, 'batch_size': 16, 'lrate': 0.0008} | 750 | 0.2894 | 183 | eliminated |
| **rung 1** | | | | | | |
| 1 | `3de4c2e3aef1` | {'accumulate_grad_batches': 2, 'batch_size': 16, 'lrate': 0.0004} | 4,500 | 0.1355 | 429 | promoted |

> **1 run(s) record no held-out check.** `reserved_pages: null` means nothing looked, which is not the same as looking and finding nothing (#119). A CER from a run that may have trained on the measurement set is not a CER on held-out material.

> **Corpus reused.** The reservation count of some runs comes from the artefact they adopted rather than from their own prepare: artefact 1ab0a28006d1 built by 20261012T135706Z-vlm-xix-effective-batch-01-noise-s42. The cache key carries the held-out fingerprint, so an artefact built under a different reservation is never adopted — the corpus is clean, and this line is what lets the record say so.

`=n` — this row is not distinguishable from the best row of its band: the gap is smaller than the noise floor, so repeating the same configuration would move it further than the difference. Bands are formed against their own leader, never chained.

> **Decided by noise: rung 0 on 0.0043** — each inside the floor 0.117. The promotions stand, because a rung has to narrow, but nothing here says the promoted configurations are better than the eliminated ones. This is the state the first architecture search published a ranking in, without saying so.
