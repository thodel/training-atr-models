#!/usr/bin/env python3
"""Does this kraken job's batch fit the card it was given? (#171)

    python ubelix/kraken_batch_fits.py <job.json>              # the card from nvidia-smi
    python ubelix/kraken_batch_fits.py <job.json> --card-mib 23028

Run by ``kraken_train.sbatch`` inside the image, on the GPU node, before the
runner starts: an OOM there is minutes into the first epoch, after the queue
wait; a refusal here is seconds.

The table is the only kraken peak measurement there is — asteraix gpu0,
07.10.2026, ``german_val.arrow``, ``KRAKEN_PLUS_SPEC`` from scratch, one full
epoch each (``scripts/measure_kraken_arms.py``):

    batch    32      64      128     256
    own MiB  12'220  13'766  20'134  43'536

It grows far faster than linearly — a batch is padded to its widest line, and a
larger batch is likelier to hold one — so a batch between two measured ones is
judged by the larger. UBELIX' free cards are RTX 4090 with 23'028 MiB: the
project default of 256 does not fit, 128 does with ~2.9 GB to spare.

Three answers, exit code in brackets:

* ``fits`` (0) — the measured peak at this batch, or the next measured one
  above it, plus 1 GiB is within the card.
* ``refused`` (3) — it is not; the largest batch that does fit is named.
* ``not checked`` (0) — the job is not the measured case (a fine-tune loads
  another network at another input height, a custom ``spec`` is another
  network), the batch is above every measurement, or the card could not be
  read. Not checked is not "fits": the line says so, and the run goes ahead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

#: Peak MiB of the training process by batch size (see the docstring).
MEASURED_MIB = {32: 12_220, 64: 13_766, 128: 20_134, 256: 43_536}
#: What the CUDA context, the allocator's slack and a second process on the
#: card are allowed beyond the measured peak.
HEADROOM_MIB = 1_024


def judge(batch: int, card_mib: Optional[int], measured_case: bool) -> Tuple[str, str]:
    """``("fits" | "refused" | "not checked", message)``."""
    if not measured_case:
        return "not checked", ("the peak table is for KRAKEN_PLUS_SPEC from scratch; "
                               "this job loads or defines another network")
    if card_mib is None:
        return "not checked", "the card's memory could not be read"
    above = [b for b in sorted(MEASURED_MIB) if b >= batch]
    if not above:
        return "not checked", ("batch %d is above every measurement (largest %d)"
                               % (batch, max(MEASURED_MIB)))
    judged_by = above[0]
    need = MEASURED_MIB[judged_by] + HEADROOM_MIB
    if need <= card_mib:
        return "fits", ("batch %d: peak %d MiB measured at %d, + %d headroom <= %d MiB card"
                        % (batch, MEASURED_MIB[judged_by], judged_by, HEADROOM_MIB, card_mib))
    fitting = [b for b in sorted(MEASURED_MIB)
               if MEASURED_MIB[b] + HEADROOM_MIB <= card_mib]
    largest = ("batch_size %d" % fitting[-1]) if fitting else "no measured batch"
    return "refused", ("batch %d needs ~%d MiB (measured %d MiB at batch %d, + %d headroom), "
                       "the card has %d MiB. %s fits; set params.batch_size and resubmit, "
                       "or run it on a larger card"
                       % (batch, need, MEASURED_MIB[judged_by], judged_by, HEADROOM_MIB,
                          card_mib, largest[0].upper() + largest[1:]))


def card_mib(nvidia_smi: str = "nvidia-smi") -> Optional[int]:
    """Total MiB of the first visible card, or None when it cannot be read."""
    try:
        out = subprocess.run([nvidia_smi, "--query-gpu=memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        return int(out.splitlines()[0].strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def measured_case(request: dict) -> bool:
    """From scratch with the project's default network — the case the table is."""
    from atr_training.contracts import KRAKEN_PLUS_SPEC

    params = request.get("params") or {}
    return (not request.get("base_model")
            and (params.get("spec") or KRAKEN_PLUS_SPEC) == KRAKEN_PLUS_SPEC)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_json", type=Path)
    parser.add_argument("--card-mib", type=int, default=None,
                        help="skip nvidia-smi and judge against this card")
    args = parser.parse_args(argv)

    request = json.loads(args.job_json.read_text(encoding="utf-8"))["request"]
    if request.get("engine", "kraken") != "kraken":
        print("kraken batch check: not a kraken job — not checked")
        return 0
    batch = int((request.get("params") or {}).get("batch_size") or 256)
    card = args.card_mib if args.card_mib is not None else card_mib()
    verdict, message = judge(batch, card, measured_case(request))
    print("kraken batch check: %s — %s" % (verdict, message),
          file=sys.stderr if verdict == "refused" else sys.stdout)
    return 3 if verdict == "refused" else 0


if __name__ == "__main__":
    sys.exit(main())
