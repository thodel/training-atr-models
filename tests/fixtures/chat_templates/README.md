# Chat templates, verbatim from the Hub (#274)

The templates the VLM runners render through `processor.apply_chat_template`,
copied byte for byte so the tests can ask what each one *reads* — and, for the
Qwen3.5 line, what `enable_thinking` does to the render — without a tokenizer,
a model or the network.

| file | Hub repository | commit | fetched |
|---|---|---|---|
| `qwen3.5-4b.jinja` | `Qwen/Qwen3.5-4B` (`chat_template.jinja`) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 2026-10-22 |
| `olmocr-2-7b-1025.jinja` | `allenai/olmOCR-2-7B-1025` (`chat_template.jinja`) | `e52d6f090b7a9007afffbbd6ce510876222fea93` | 2026-10-22 |
| `qwen3-vl-4b-instruct.jinja` | `Qwen/Qwen3-VL-4B-Instruct` (the `chat_template` string of `chat_template.json`) | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | 2026-10-22 |
| `gemma-4-e4b-it.jinja` | `google/gemma-4-E4B-it` (`chat_template.jinja`) | `ee0ef6023621cff504d758262d4e04895a5af4a2` | 2026-10-22 |

A base model's template can change under the same name; when a run's render
does not match what these tests pin, refresh the file from the commit the run
loaded and say so here.
