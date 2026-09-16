"""Training ATR models: the orchestration half, without the ML stack.

Empty on purpose until #3 moves `src/atr_serving/training/` here. The directory
itself is not optional even now: `[tool.setuptools.packages.find] where = ["src"]`
makes an editable install fail with

    error in 'egg_base' option: 'src' does not exist or is not a directory

before pip ever reaches the test suite, which is how the first CI run failed.

What will live here is the part that *decides* — job store, stage machine,
dataset selection, split, contracts, metrics — and what deliberately will not is
anything that imports torch. The three engine runners under `engines/` are
spawned in their own venvs (.venvs/{kraken,trocr,vlm}-train), so the service can
supervise a QLoRA fine-tune without being able to run one.
"""
