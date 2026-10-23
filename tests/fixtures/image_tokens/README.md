# image_tokens — a real line and a real page (#278)

Two JPEGs from `dh-unibe/image-text_rats-und-richtebuecher_xv-xvi` (Zurich
Rats- und Richtebücher, 1400–1550, the State Archive of Zurich; a public
dataset of this project), shard
`data/train/Rats-undRichtebücher_MF_1_3543/20260404_134612-0000.parquet`, page
`1608169_0002_60467815.jpg` (2902×3674 px, 4 transcribed lines).

| file | what | how it was made |
|---|---|---|
| `line.jpg` | 631×178 px, the line “Rathsbücher”, bbox `[1105, 999, 1736, 1177]` | `vlm_dataset.line_samples` on the page’s PageXML, then `cropping.write_crops` — the bytes a trainer sees, unchanged |
| `page.jpg` | 1287×1629 px, the whole page | `vlm_dataset.fit_pixels(page, VLM_PIXEL_BUDGET["page"])`, JPEG quality 75 — already at the page budget, so fitting it again is a no-op and the count at the page budget is the count of the full page |

They exist so `scripts/count_image_tokens.py` and `tests/test_image_tokens.py`
count the same image everywhere: a token count is only comparable across bases,
transformers versions and machines when the pixels are the same. Do not replace
them without re-measuring the table in both.
