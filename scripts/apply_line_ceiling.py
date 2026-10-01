#!/usr/bin/env python3
"""Apply today's line ceiling to a page pool that was built before it (#115).

serving-atr-inference#90 was fixed in this repo on 17.09.2026: ``prepare`` drops
``TextLine`` elements whose box is absurdly wider than it is tall
(:data:`pagexml.MAX_LINE_ASPECT`, 60:1), because kraken pads every batch to its
widest member and one such line is paid for by every other line in its batch.

The fix does not travel backwards. ``sweep_train.arrow`` was compiled from a page
pool materialised on **05.09.2026**, and it still holds a 177:1 line — four
attempts to fine-tune on it died of CUDA OOM, three to thirteen minutes in,
mid-epoch, which is the signature of a peak that hangs on one outlier rather than
on the mean. The code did not need writing; it needed **applying**.

    scripts/apply_line_ceiling.py --pool  .../pages \\
                                  --out   .../pages-ceiling-60 \\
                                  --dry-run

Then without ``--dry-run``, and compile the new directory.

**The source pool is never written to.** The identity of the measurement set
hangs on it: ``german_test`` was built from this pool, and the acceptance figure
0.2131 means 188,022 errors over 882,255 characters of it. Editing it in place
would quietly redefine what every past number was measured against. So a new
directory is written, the images are linked rather than copied, and this refuses
to run if the output would land inside the input.

A page that loses *every* line is dropped, exactly as ``prepare`` drops it: both
mean nothing trainable is left.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_training.pagexml import (  # noqa: E402
    MAX_LINE_ASPECT,
    drop_wide_lines,
    page_stats,
)

#: Tried in this order for every image. A hard link costs nothing and reads as a
#: normal file; CIFS may refuse it, and then a symlink; and if the pool and the
#: output are on different filesystems, a copy.
LINK_MODES = ("hardlink", "symlink", "copy")


@dataclass
class Outcome:
    pages_read: int = 0
    pages_written: int = 0
    pages_dropped: int = 0
    lines_before: int = 0
    lines_dropped: int = 0
    max_aspect: float = 0.0
    #: The widest ratio still present after the cut. Measured rather than
    #: assumed: a line with no parsable geometry is left alone by
    #: ``drop_wide_lines`` on purpose, so "the cut ran" and "the tail is gone"
    #: are two statements and only the second one matters to the next run.
    max_aspect_after: float = 0.0
    images_linked: int = 0
    images_missing: list[str] = field(default_factory=list)
    link_mode: str = ""

    @property
    def share(self) -> float:
        return 0.0 if not self.lines_before else self.lines_dropped / self.lines_before

    def report(self) -> str:
        return (
            f"{self.pages_read:,} pages read, {self.pages_written:,} written, "
            f"{self.pages_dropped:,} dropped (nothing trainable left)\n"
            f"{self.lines_dropped:,} of {self.lines_before:,} lines removed "
            f"({100 * self.share:.2f} %), widest before the cut "
            f"{self.max_aspect:.1f}:1, after {self.max_aspect_after:.1f}:1\n"
            f"{self.images_linked:,} images {self.link_mode}ed"
            + (f", {len(self.images_missing)} missing" if self.images_missing else ""))


def _link(src: Path, dest: Path, mode: str) -> str:
    """Put ``src`` at ``dest`` as cheaply as the filesystem allows."""
    modes = LINK_MODES[LINK_MODES.index(mode):] if mode in LINK_MODES else LINK_MODES
    for candidate in modes:
        try:
            if candidate == "hardlink":
                os.link(src, dest)
            elif candidate == "symlink":
                os.symlink(src.resolve(), dest)
            else:
                shutil.copy2(src, dest)
            return candidate
        except OSError:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            continue
    raise OSError(f"could not place {src.name} in {dest.parent} by any means")


def apply_ceiling(pool: Path, out: Path, *, max_aspect: float = MAX_LINE_ASPECT,
                  dry_run: bool = False, limit: int | None = None) -> Outcome:
    """Copy ``pool`` to ``out`` with the over-wide lines removed.

    Returns what it did. ``dry_run`` measures and writes nothing, which is how
    this should be read first: the share of lines it would remove is the number
    that says whether the corpus needs rebuilding at all.
    """
    result = Outcome()
    mode = LINK_MODES[0]
    pages = sorted(pool.glob("*.xml"))
    if not pages:
        raise SystemExit(f"no *.xml in {pool} — is that the pages directory?")
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)

    for xml_path in pages[:limit]:
        result.pages_read += 1
        text = xml_path.read_text(encoding="utf-8", errors="replace")
        before = page_stats(text)
        result.lines_before += before.lines
        result.max_aspect = max(result.max_aspect, before.max_aspect or 0.0)
        edited, dropped = drop_wide_lines(text, max_aspect)
        result.lines_dropped += dropped
        after = page_stats(edited)
        if not after.usable:
            result.pages_dropped += 1
            continue
        result.max_aspect_after = max(result.max_aspect_after, after.max_aspect or 0.0)
        result.pages_written += 1
        if dry_run:
            continue
        (out / xml_path.name).write_text(edited, encoding="utf-8")
        image = next((p for p in (xml_path.with_suffix(suffix)
                                  for suffix in (".jpg", ".jpeg", ".png", ".tif"))
                      if p.exists()), None)
        if image is None:
            result.images_missing.append(xml_path.name)
            continue
        target = out / image.name
        if not target.exists():
            mode = _link(image, target, mode)
            result.images_linked += 1
    result.link_mode = mode
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]))
    parser.add_argument("--pool", type=Path, required=True,
                        help="the pages directory to read — never written to")
    parser.add_argument("--out", type=Path, required=True,
                        help="where the cut copy goes")
    parser.add_argument("--max-aspect", type=float, default=MAX_LINE_ASPECT)
    parser.add_argument("--dry-run", action="store_true",
                        help="measure what would be removed and write nothing")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N pages, for a look before the run")
    args = parser.parse_args(argv)

    pool, out = args.pool.resolve(), args.out.resolve()
    if not pool.is_dir():
        print(f"ERROR: {pool} is not a directory", file=sys.stderr)
        return 2
    # The one thing that must not happen. The measurement set's identity hangs
    # on this pool, so there is no --force for it.
    if out == pool or pool in out.parents:
        print(f"ERROR: --out would land inside --pool ({out}). The source pool "
              "defines what every past number was measured against; it is not "
              "edited in place, here or by a flag.", file=sys.stderr)
        return 2

    result = apply_ceiling(pool, out, max_aspect=args.max_aspect,
                           dry_run=args.dry_run, limit=args.limit)
    print(result.report())
    if result.images_missing:
        print(f"\nfirst few pages with no image beside them: "
              f"{result.images_missing[:5]}", file=sys.stderr)
    if args.dry_run:
        print(f"\n--dry-run: nothing written. {result.lines_dropped:,} lines "
              f"({100 * result.share:.2f} %) would go.")
        return 0
    print(f"\nwritten to {out}. Widest line left {result.max_aspect_after:.1f}:1, "
          f"under the {args.max_aspect:.0f}:1 ceiling. Compile it, and the corpus "
          "that comes out is one today's prepare would also produce.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
