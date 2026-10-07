"""Build a line-level dataset from a page-shaped one and push it to the hub.

The datasets this project trains on are page-shaped: one row per scan, with the
PageXML beside it. The training pipeline cuts the lines itself, so a line-shaped
copy buys the *trainer* nothing — `data-towerbooks-textlines` is read through
`source: line` and the rest through materialize-and-crop, and both routes work.
What a line-shaped copy does buy is everything outside this pipeline: a dataset
anyone can open with `load_dataset` and see one crop with one transcription,
without a PageXML parser, a segmenter, or this repository.

**The crops are this project's own.** The chain here is the chain the prepare
stage runs — `HFPageSource.stream` → `prepare.materialize` → `samples_for` →
`cropping.write_crops` — so a line in the published dataset is byte-identical to
the line the trainer sees. A second implementation would have been a second set of
numbers.

    python ubelix/make_line_dataset.py --repo dh-unibe/image-text_aaeb-xiv-xvii-part-2 \
        --revision ce2739fce41c0b8f8673127525839842411020b0 \
        --out dh-unibe/image-text_aaeb-xiv-xvii-part-2-lines --max-pages 200

`--dry-run` stops before the push and leaves the parquet on disk to inspect.

**Schema.** `image` (JPEG bytes), `text`, `page`, `line_index`, `project`,
`source_repo`, `source_revision`. It is deliberately narrower than
`data-towerbooks-textlines`, which also carries `line_id`, `region_id` and the two
reading orders: those come from the PageXML and this project's `Sample` does not
keep them, so publishing them would mean inventing them.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from atr_training.contracts import DatasetSpec
from atr_training.cropping import write_crops
from atr_training.hf_source import (
    data_files_for,
    expand_all_projects,
    list_projects,
    whole_split_glob,
)
from atr_training.prepare import HFPageSource, materialize
from atr_training.vlm_dataset import samples_for


def source_body(repo: str, revision: str) -> str:
    """The source card's prose, without its YAML frontmatter.

    The frontmatter declares the *source's* features and split sizes, which are
    wrong for a line-level copy — `push_to_hub` writes correct ones. The prose is
    the part worth carrying over: it names the archives, the period and the
    projects, and that provenance should not be lost because the rows were cut
    differently.
    """
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo, "README.md", repo_type="dataset", revision=revision)
    except Exception as exc:  # a dataset without a card is not an error here
        logger.warning("{}: no README to carry over ({})", repo, exc)
        return ""
    text = Path(path).read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    return text.strip()


def existing_frontmatter(out: str) -> str:
    """The YAML block `push_to_hub` wrote for the new dataset, verbatim.

    Replacing the whole README loses it, and with it the declared features and
    split sizes the dataset viewer needs — the hub answers that with "empty or
    missing yaml metadata in repo card" and a dataset nobody can preview. Only
    the prose below it is ours to write.
    """
    from huggingface_hub import hf_hub_download
    try:
        text = Path(hf_hub_download(out, "README.md", repo_type="dataset")).read_text("utf-8")
    except Exception as exc:
        logger.warning("{}: no card to take frontmatter from ({})", out, exc)
        return ""
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    return text[:end + 4] if end != -1 else ""


def card(out: str, repo: str, revision: str, lines: int, chars: int, summary: str) -> str:
    """The new dataset's card: its own frontmatter, our summary, the source's prose."""
    body = source_body(repo, revision)
    head = existing_frontmatter(out)
    parts = ([head, ""] if head else []) + [
        "# %s" % out.split("/")[-1],
        "",
        "**Line-level variant** of [`%s`](https://hf.co/datasets/%s) at revision "
        "`%s`. One row per transcribed line — the cropped line image and its "
        "transcription — so the material can be read without a PageXML parser or a "
        "line segmenter." % (repo, repo, revision),
        "",
        "| | |",
        "|---|---|",
        "| Lines | %d |" % lines,
        "| Characters | %d |" % chars,
        "| Source | [`%s`](https://hf.co/datasets/%s) |" % (repo, repo),
        "| Source revision | `%s` |" % revision,
        "",
        "What the source reported while being read: `%s`" % summary,
        "",
        "The crops are cut by the same code that feeds this project's training runs "
        "(`cropping.write_crops` in "
        "[thodel/training-atr-models](https://github.com/thodel/training-atr-models)), "
        "so a line here is byte-identical to the line a trainer sees.",
        "",
        "## Columns",
        "",
        "`image` — the line crop, JPEG bytes. `text` — its transcription. "
        "`page` — the PageXML the line came from. `line_index` — position in the "
        "cropping order. `project` — the source project, where the source records "
        "one. `source_repo`, `source_revision` — the pinned origin of every row.",
        "",
        "Deliberately narrower than `dh-unibe/data-towerbooks-textlines`, which also "
        "carries `line_id`, `region_id` and two reading orders: those live in the "
        "PageXML and this project's sample does not keep them, so publishing them "
        "would mean inventing them.",
    ]
    if body:
        parts += ["", "---", "",
                  "## The source dataset's own description",
                  "",
                  "Carried over unchanged from [`%s`](https://hf.co/datasets/%s):" % (repo, repo),
                  "", body]
    return "\n".join(parts) + "\n"


def train_globs(repo: str, revision: str, max_pages: int,
                projects: list[str] | None) -> list[str]:
    """The ``data_files`` globs to read, for either dataset layout.

    Most dh-unibe exports are laid out ``data/<split>/<project>/``, and
    ``all_projects`` enumerates those directories. Two are not:
    `image-text_koenigsfelden-charters-part-3` and
    `transkribus-exports-bullinger-handschrift` hold flat shards
    (``data/train-00000-of-00002.parquet``), so the enumeration finds nothing and
    `expand_all_projects` raises — which is how both lost their jobs. They are
    page-shaped all the same, so the fallback is the whole split, asked for
    explicitly here rather than by weakening the guard in
    :func:`hf_source.data_files_for`: an *empty* selection must still never mean
    "everything", because on a repo that does have project directories that
    silently reads 6.6 TB.
    """
    if projects:
        spec = DatasetSpec(hf_repo=repo, granularity="line", train_projects=list(projects),
                           max_pages=max_pages, revision=revision)
        files = data_files_for(expand_all_projects(spec))
        return files.get("train") or next(iter(files.values()))

    if not list_projects(repo, "train", revision):
        logger.info("{}: no project directories — reading the whole split", repo)
        return [whole_split_glob("train")]

    spec = DatasetSpec(hf_repo=repo, granularity="line", all_projects=True,
                       max_pages=max_pages, revision=revision)
    files = data_files_for(expand_all_projects(spec))
    return files.get("train") or next(iter(files.values()))


@dataclass
class Built:
    """Crops on disk, plus how to read them back one at a time.

    ``records`` is a generator, not a list, because holding every crop's JPEG in
    memory at once is what killed four of the sixteen jobs on 05.10.2026 — all
    four died at the hand-over from cropping to `Dataset.from_list`, and
    `rats-und-richtebuecher` got as far as crop 139'707 of 139'708 before the
    OOM killer took it. Slurm's `MaxRSS` under-reported it (9–47 GB against a
    48 G request) because it samples every 30 s and missed the spike. The crops
    are already on disk; reading them back one at a time keeps peak memory flat
    and lets `Dataset.from_generator` write Arrow incrementally.
    """

    manifest: list[tuple[str, str, str, str]]
    workdir: Path
    summary: str
    repo: str
    revision: str

    @property
    def lines(self) -> int:
        return len(self.manifest)

    @property
    def chars(self) -> int:
        return sum(len(text) for _, text, _, _ in self.manifest)

    @property
    def crop_bytes(self) -> int:
        return sum((self.workdir / rel).stat().st_size for rel, _, _, _ in self.manifest)

    def records(self):
        for index, (rel, text, page, project) in enumerate(self.manifest):
            path = self.workdir / rel
            yield {
                "image": {"bytes": path.read_bytes(), "path": Path(rel).name},
                "text": text,
                "page": page,
                "line_index": index,
                "project": project,
                "source_repo": self.repo,
                "source_revision": self.revision,
            }


def build(repo: str, revision: str, max_pages: int, workdir: Path,
          projects: list[str] | None = None) -> Built:
    """Crop every transcribed line of ``repo`` and return what to publish.

    ``projects`` names the project directories to read instead of all of them.
    Needed where a dataset holds more than one shape of the same material:
    `image-text_sg-missiven` carries `sg-missiven` as pages and
    `sg-missiven-singleline` as the same pages with the text in one line, and
    neither carries line coordinates.
    """
    globs = train_globs(repo, revision, max_pages, projects)
    logger.info("{}: {} data file glob(s)", repo, len(globs))

    rows = HFPageSource(cache=True).stream(repo, globs, revision=revision)
    pages = materialize(rows, workdir, role="pool", max_pages=max_pages)
    logger.info("{}: {}", repo, pages.summary)
    if not pages.xml_paths:
        return Built([], workdir, pages.summary, repo, revision)

    samples = samples_for(pages.xml_paths, "line", root=workdir)
    logger.info("{}: {} line sample(s) before cropping", repo, len(samples))
    if not samples:
        # The pool counts a "transcribed line" from the PageXML's text alone
        # (`pagexml.page_stats`), so a page-level export with no <Coords> reports
        # thousands of lines and yields none to crop: `image-text_sg-missiven` said
        # 25'814 lines over 3'334 pages and produced 0 samples, with "worst aspect
        # 0:1" as the only hint. Say so here rather than let the caller read the
        # pool's promise as a contradiction.
        logger.warning("{}: the source reports lines but carries no line coordinates — "
                       "a line dataset needs segmentation first", repo)
        return Built([], workdir, pages.summary, repo, revision)

    cropped = write_crops(samples, workdir, workdir / "crops")

    manifest = [
        (sample.image, sample.text, sample.page or "", sample.source or "")
        for sample in cropped
        if (workdir / sample.image).is_file()
    ]
    return Built(manifest, workdir, pages.summary, repo, revision)


def settle_visibility(out: str, want_private: bool) -> bool:
    """Decide the target's visibility before any image is uploaded.

    **Public is the normal case for a dataset** (Tobias, 07.10.2026): a line-level
    copy exists so the material can be used without a PageXML parser, and a copy
    nobody can open does not serve that. ``--private`` is the deliberate
    exception, for a source whose terms require it.

    Models are the other way round and stay so — see
    ``scripts/publish_to_hub.py``: private unless ``--public``, and made public by
    hand in the web interface.

    Two rules from that same house pattern apply here as well:

    * **An existing repo keeps its visibility.** It is reported and not changed,
      in either direction. ``push_to_hub``'s own docstring says the ``private``
      flag "is ignored if the repo already exists", so pretending otherwise would
      be a promise this code cannot keep — and on 05.10.2026 it did not: three
      datasets went up public although every commit asked for ``private=True``.
    * **Visibility is only set at creation.** Which is the one moment the flag
      works.

    The refusal built for the opposite default survives, behind the flag: with
    ``--private``, a target that is or comes out public stops the run before the
    first image, and a hub that cannot be reached stops it too, because "could not
    look" is not "is private". Without the flag neither case matters.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi()
    try:
        info = api.dataset_info(out)
    except RepositoryNotFoundError:
        api.create_repo(out, repo_type="dataset", private=want_private, exist_ok=True)
        if not want_private:
            print(f"   target: created {out} as public")
            return True
        try:
            created = api.dataset_info(out)
        except Exception as exc:  # noqa: BLE001 — cannot verify ⇒ do not upload
            print(f"created {out} but could not read back its visibility: {exc}",
                  file=sys.stderr)
            return False
        if created.private:
            print(f"   target: created {out} as PRIVATE (--private)")
            return True
        print(f"created {out} and the hub reports it PUBLIC despite private=True — "
              "refusing to upload because --private was asked for. Set it private "
              "in the HF UI, then re-run.", file=sys.stderr)
        return False
    except Exception as exc:  # noqa: BLE001
        if not want_private:
            print(f"   target: could not read {out}'s visibility "
                  f"({type(exc).__name__}) — uploading anyway, public is the default")
            return True
        print(f"could not check whether {out} is private ({type(exc).__name__}: {exc}); "
              "--private was asked for, so refusing to upload rather than guess",
              file=sys.stderr)
        return False

    state = "PRIVATE" if info.private else "public"
    if want_private and not info.private:
        print(f"{out} already exists and is public, and --private was asked for. "
              "push_to_hub cannot change that — the flag is ignored for an existing "
              "repo — so the crops would stay public. Set it private in the HF UI, "
              "or drop --private.", file=sys.stderr)
        return False
    print(f"   target: {out} exists and is {state} — kept as it is")
    return True


def confirm_private(out: str) -> bool:
    """Read the visibility back after the push, for a run that asked for private.

    Only called with ``--private``. Without it a public result is the intent, not
    a finding.
    """
    from huggingface_hub import HfApi
    try:
        info = HfApi().dataset_info(out)
    except Exception as exc:  # noqa: BLE001
        print(f"pushed to {out} but could not confirm its visibility: {exc}",
              file=sys.stderr)
        return False
    if info.private:
        return True
    print(f"pushed to {out} and the hub now reports it PUBLIC. The crops are "
          "readable by anyone. Make it private in the HF UI "
          f"(https://hf.co/datasets/{out}/settings) and check what else that run "
          "published.", file=sys.stderr)
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="the page-shaped source dataset")
    p.add_argument("--revision", required=True, help="full 40-hex SHA of the source")
    p.add_argument("--out", required=True, help="the dataset repo to create, e.g. owner/name-lines")
    p.add_argument("--max-pages", type=int, required=True,
                   help="bound on pages read; DatasetSpec refuses all_projects without one")
    p.add_argument("--projects", default=None,
                   help="comma-separated project directories to read; default is all of them")
    p.add_argument("--dry-run", action="store_true", help="build and report, do not push")
    p.add_argument("--private", action="store_true",
                   help="create the dataset private. Public is the default: a "
                        "line-level copy exists so the material can be used, and "
                        "one nobody can open does not serve that. Use this for a "
                        "source whose terms require it. An existing repo keeps "
                        "whatever visibility it has, in either direction.")
    # The pages and crops land here before they are packed. /tmp on the login node
    # has ~18 GB and `materialize` wants 50 free, so the default is scratch.
    p.add_argument("--workdir", default=None,
                   help="where to materialize (default: $ATR_TRAIN_SCRATCH or "
                        "/scratch/network/users/$USER/line-datasets)")
    args = p.parse_args(argv)

    if len(args.revision) != 40:
        print("--revision must be a full 40-character SHA", file=sys.stderr)
        return 2

    if not args.dry_run and not settle_visibility(args.out, args.private):
        return 1

    base = Path(args.workdir or os.environ.get("ATR_TRAIN_SCRATCH")
                or "/scratch/network/users/%s/line-datasets" % os.environ.get("USER", "unknown"))
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="line-ds-", dir=str(base)) as tmp:
        chosen = [x.strip() for x in args.projects.split(",") if x.strip()] if args.projects else None
        built = build(args.repo, args.revision, args.max_pages, Path(tmp), chosen)
        if not built.lines:
            print(f"no transcribed lines in {args.repo} — nothing to publish "
                  f"({built.summary})", file=sys.stderr)
            return 1
        print(f"\n== {args.repo} -> {args.out}")
        if chosen:
            print(f"   projects: {', '.join(chosen)}")
        print(f"   {built.lines} line(s), {built.chars} characters, "
              f"{built.crop_bytes / 1e6:.1f} MB of crops")
        print(f"   source: {built.summary}")
        print(f"   example: {built.manifest[0][1][:70]!r}")

        from datasets import Dataset, Features, Image, Value
        features = Features({
            "image": Image(decode=False),
            "text": Value("string"),
            "page": Value("string"),
            "line_index": Value("int32"),
            "project": Value("string"),
            "source_repo": Value("string"),
            "source_revision": Value("string"),
        })
        # from_generator, not from_list: see Built.records. The cache goes next to
        # the crops, because $HOME on UBELIX is not sized for 140'000 line images.
        ds = Dataset.from_generator(
            built.records, features=features, cache_dir=str(Path(tmp) / "arrow"))
        if args.dry_run:
            local = Path.cwd() / (args.out.split("/")[-1] + ".parquet")
            ds.to_parquet(str(local))
            print(f"   dry run: wrote {local} ({local.stat().st_size / 1e6:.1f} MB), nothing pushed")
            return 0
        ds.push_to_hub(args.out, private=args.private)
        from huggingface_hub import HfApi
        text = card(args.out, args.repo, args.revision, built.lines, built.chars, built.summary)
        HfApi().upload_file(
            path_or_fileobj=text.encode("utf-8"), path_in_repo="README.md",
            repo_id=args.out, repo_type="dataset",
            commit_message="Describe the line-level variant, and keep the source's own description")
        print(f"   card: {len(text)} characters, source description "
              f"{'carried over' if 'own description' in text else 'unavailable'}")
        if args.private:
            if not confirm_private(args.out):
                return 1
            print(f"   pushed as PRIVATE to https://hf.co/datasets/{args.out}")
        else:
            print(f"   pushed to https://hf.co/datasets/{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
