"""What changed in the training material since the last look — and what it would do (#32).

Every early run drew on a dataset whose German content was ~291 pages, while a
9,885-page set of Zurich council books sat unqueried for a month. Nothing watched
for the next one: ``plan_corpus`` answers "what is the best corpus *now*" when
someone thinks to run it, and has no notion of what changed.

This module is the memory and the comparison. A run takes a :class:`Snapshot` of
an org's datasets — what each card says, reduced to what the planner reads — and
:func:`diff` reports **changes, not inventory**:

* a dataset that is new, or gone from the listing;
* projects that appeared in, or left, a dataset's "Projects Included";
* a material change in page count;
* card metadata that changes scoring — period, languages, document type, or the
  card's own statement that its text is machine output.

Each change carries its **consequence**: the dataset's score against every target
before and after, and where the corpus plan puts it before and after. A change
that moves nothing is still reported, and says so — that is information too. An
unchanged Hub produces no changes, and the caller writes no report.

Two refusals keep the memory honest:

* A dataset listed but unreadable this run keeps its last known state and is
  reported as unreadable, never as gone (:meth:`Snapshot.carry_forward`).
* :func:`implausible` names a listing too small to trust; the caller then keeps
  the old snapshot rather than overwrite it with an outage.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from atr_training.corpus_plan import (
    MEDIEVAL_GERMAN,
    Candidate,
    CorpusPlanError,
    Target,
    parse_card,
    plan_corpus,
    score_candidate,
)

__all__ = ["DatasetState", "Snapshot", "Change", "diff", "implausible", "render",
           "SNAPSHOT_VERSION", "SIZE_MIN_PAGES", "SIZE_MIN_RATIO"]

SNAPSHOT_VERSION = 1

#: A page count has to move by at least this many pages **and** this share of the
#: old count to be a change. Both, so a 40-page set gaining 30 pages counts (75 %)
#: while a 500,000-page set drifting by 200 does not.
SIZE_MIN_PAGES = 100
SIZE_MIN_RATIO = 0.05

#: Below this share of the previous listing, a run is presumed broken (auth,
#: network, a Hub hiccup) and must not replace the memory.
PLAUSIBLE_SHARE = 0.5


@dataclass(frozen=True)
class DatasetState:
    """One dataset as the planner sees it — the card reduced to what scores."""

    repo: str
    pages: int
    gb: float
    projects: tuple[str, ...] = ()
    period: tuple[int, int] | None = None
    languages: tuple[str, ...] = ()
    doc_type: str = ""
    ground_truth: bool = True

    @classmethod
    def from_row(cls, row: dict) -> "DatasetState":
        c = parse_card(row["repo"], row.get("card", ""), int(row.get("pages") or 0),
                       float(row.get("gb") or 0.0), row.get("projects") or ())
        return cls(repo=c.repo, pages=c.pages, gb=c.gb, projects=tuple(c.projects),
                   period=c.period, languages=tuple(c.languages), doc_type=c.doc_type,
                   ground_truth=c.ground_truth)

    def candidate(self) -> Candidate:
        return Candidate(repo=self.repo, pages=self.pages, gb=self.gb,
                         period=self.period, languages=self.languages,
                         doc_type=self.doc_type, projects=self.projects,
                         ground_truth=self.ground_truth)

    @classmethod
    def from_json(cls, data: dict) -> "DatasetState":
        period = data.get("period")
        return cls(repo=data["repo"], pages=int(data["pages"]), gb=float(data["gb"]),
                   projects=tuple(data.get("projects") or ()),
                   period=tuple(period) if period else None,
                   languages=tuple(data.get("languages") or ()),
                   doc_type=data.get("doc_type", ""),
                   ground_truth=bool(data.get("ground_truth", True)))


@dataclass
class Snapshot:
    org: str
    taken_at: str
    datasets: dict[str, DatasetState] = field(default_factory=dict)
    #: Listed by the Hub but not readable this run, with the reason.
    unreadable: dict[str, str] = field(default_factory=dict)
    #: Unreadable now, state carried from an earlier snapshot — and since when.
    carried: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_catalogue(cls, org: str, listed: Iterable[str], rows: Iterable[dict],
                       unreadable: dict[str, str] | None = None,
                       taken_at: str | None = None) -> "Snapshot":
        states = {s.repo: s for s in map(DatasetState.from_row, rows)}
        unreadable = dict(unreadable or {})
        # A listed dataset that produced neither a row nor an error is unreadable
        # too: silence must not become absence.
        for repo in listed:
            if repo not in states and repo not in unreadable:
                unreadable[repo] = "listed, but no card was read"
        return cls(org=org, taken_at=taken_at or _now(), datasets=states,
                   unreadable=unreadable)

    def listed(self) -> set[str]:
        return set(self.datasets) | set(self.unreadable)

    def carry_forward(self, previous: "Snapshot | None") -> "Snapshot":
        """The snapshot to *keep*: unreadable datasets keep their last known state.

        Without this the next readable run would report every project of a dataset
        that had a bad morning as newly added.
        """
        if previous is None:
            return self
        kept = dict(self.datasets)
        carried = {}
        for repo in self.unreadable:
            if repo in previous.datasets and repo not in kept:
                kept[repo] = previous.datasets[repo]
                carried[repo] = previous.carried.get(repo, previous.taken_at)
        return Snapshot(org=self.org, taken_at=self.taken_at, datasets=kept,
                        unreadable=dict(self.unreadable), carried=carried)

    def to_json(self) -> dict:
        return {"version": SNAPSHOT_VERSION, "org": self.org, "taken_at": self.taken_at,
                "datasets": {r: asdict(s) for r, s in sorted(self.datasets.items())},
                "unreadable": dict(sorted(self.unreadable.items())),
                "carried": dict(sorted(self.carried.items()))}

    @classmethod
    def from_json(cls, data: dict) -> "Snapshot":
        if data.get("version") != SNAPSHOT_VERSION:
            raise ValueError(f"snapshot version {data.get('version')!r}, "
                             f"this code reads {SNAPSHOT_VERSION}")
        return cls(org=data["org"], taken_at=data["taken_at"],
                   datasets={r: DatasetState.from_json(s)
                             for r, s in data["datasets"].items()},
                   unreadable=dict(data.get("unreadable") or {}),
                   carried=dict(data.get("carried") or {}))

    def dumps(self) -> str:
        return json.dumps(self.to_json(), indent=1, ensure_ascii=False) + "\n"


@dataclass(frozen=True)
class Change:
    repo: str
    #: new, gone, projects, size, metadata, unreadable
    kind: str
    what: str
    #: What it would do, per target. Empty for kinds that have no consequence.
    consequence: tuple[str, ...] = ()


def implausible(previous: Snapshot | None, current: Snapshot) -> str | None:
    """Why this listing should not replace the memory, or None when it may."""
    if previous is None or not previous.listed():
        return None if current.listed() else "the Hub listed no dataset at all"
    if len(current.listed()) < PLAUSIBLE_SHARE * len(previous.listed()):
        return (f"the Hub listed {len(current.listed())} dataset(s) where the last run "
                f"saw {len(previous.listed())} — more likely an outage than a deletion")
    if len(current.datasets) < PLAUSIBLE_SHARE * len(previous.datasets):
        return (f"only {len(current.datasets)} card(s) were readable where the last run "
                f"read {len(previous.datasets)}")
    return None


def diff(previous: Snapshot, current: Snapshot,
         targets: Sequence[Target] = (MEDIEVAL_GERMAN,)) -> list[Change]:
    """Every change between two snapshots, each with its consequence.

    *current* is the snapshot as fetched, before :meth:`Snapshot.carry_forward` —
    an unreadable dataset is reported as unreadable, not compared.
    """
    before = _positions(previous, targets)
    after = _positions(current.carry_forward(previous), targets)
    changes: list[Change] = []

    for repo in sorted(current.datasets.keys() - previous.listed()):
        state = current.datasets[repo]
        changes.append(Change(repo, "new",
                              f"new dataset: {state.pages:,} pages, "
                              f"{len(state.projects)} project(s), {_profile(state)}",
                              _consequence(repo, before, after, targets)))

    for repo in sorted(set(current.unreadable) - previous.listed()):
        changes.append(Change(repo, "new",
                              f"new dataset, but its card is not readable "
                              f"({current.unreadable[repo]}) — nothing to score yet"))

    for repo in sorted(previous.listed() - current.listed()):
        changes.append(Change(repo, "gone", "no longer listed by the Hub",
                              _consequence(repo, before, after, targets)))

    for repo in sorted(set(current.unreadable) & previous.datasets.keys()):
        since = previous.carried.get(repo)
        if since:
            continue  # already reported the first time; the report lists it below
        changes.append(Change(repo, "unreadable",
                              f"card not readable this run ({current.unreadable[repo]}); "
                              "last known state kept"))

    for repo in sorted(current.datasets.keys() & previous.datasets.keys()):
        old, new = previous.datasets[repo], current.datasets[repo]
        found = _compare(old, new)
        if found:
            consequence = _consequence(repo, before, after, targets)
            changes.extend(Change(repo, kind, what, consequence) for kind, what in found)
    return changes


def render(changes: Sequence[Change], previous: Snapshot, current: Snapshot) -> str:
    """Markdown for a human. Empty when nothing changed — no report is the report."""
    if not changes:
        return ""
    lines = [f"# Training material: {len(changes)} change(s) in `{current.org}`", "",
             f"Compared {previous.taken_at} → {current.taken_at}. Scores and plan "
             "positions are estimates from the cards; only `prepare` knows real counts.",
             ""]
    by_repo: dict[str, list[Change]] = {}
    for change in changes:
        by_repo.setdefault(change.repo, []).append(change)
    for repo, items in by_repo.items():
        lines.append(f"## `{repo.split('/')[-1]}`")
        for item in items:
            lines.append(f"- **{item.kind}** — {item.what}")
        consequence = items[0].consequence
        for line in consequence:
            lines.append(f"  - {line}")
        lines.append("")
    carried = current.carry_forward(previous).carried
    if carried:
        lines.append("Still unreadable, last known state kept: " + ", ".join(
            f"`{r.split('/')[-1]}` (since {since})" for r, since in sorted(carried.items())))
    return "\n".join(lines).rstrip() + "\n"


# ── internals ───────────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M %Z")


def _profile(state: DatasetState) -> str:
    period = f"{state.period[0]}–{state.period[1]}" if state.period else "period unknown"
    langs = ", ".join(state.languages) or "languages unknown"
    kind = state.doc_type or "type unknown"
    gt = "" if state.ground_truth else ", MACHINE OUTPUT per card"
    return f"{period}, {langs}, {kind}{gt}"


def _compare(old: DatasetState, new: DatasetState) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    added = [p for p in new.projects if p not in set(old.projects)]
    removed = [p for p in old.projects if p not in set(new.projects)]
    if added or removed:
        parts = []
        if added:
            parts.append(f"{len(added)} project(s) added ({_names(added)})")
        if removed:
            parts.append(f"{len(removed)} removed ({_names(removed)})")
        found.append(("projects", "; ".join(parts)))
    delta = new.pages - old.pages
    if abs(delta) >= SIZE_MIN_PAGES and abs(delta) >= SIZE_MIN_RATIO * max(old.pages, 1):
        found.append(("size", f"{old.pages:,} → {new.pages:,} pages ({delta:+,})"))
    meta = []
    if old.period != new.period:
        meta.append(f"period {old.period} → {new.period}")
    if set(old.languages) != set(new.languages):
        meta.append(f"languages {list(old.languages)} → {list(new.languages)}")
    if old.doc_type != new.doc_type:
        meta.append(f"type {old.doc_type!r} → {new.doc_type!r}")
    if old.ground_truth != new.ground_truth:
        meta.append("card now says the text is MACHINE OUTPUT" if not new.ground_truth
                    else "card no longer says the text is machine output")
    if meta:
        found.append(("metadata", "; ".join(meta)))
    return found


def _names(projects: Sequence[str], show: int = 4) -> str:
    head = ", ".join(f"`{p}`" for p in projects[:show])
    return head + (f" and {len(projects) - show} more" if len(projects) > show else "")


def _positions(snapshot: Snapshot, targets: Sequence[Target]) -> dict:
    """Per target: each dataset's score, and its rank and pages in the plan."""
    candidates = [s.candidate() for s in snapshot.datasets.values() if s.pages]
    out = {}
    for target in targets:
        scores = {c.repo: score_candidate(c, target).score for c in candidates}
        try:
            plan = plan_corpus(candidates, target, min_pages=100)
            ranks = {s.repo: (i + 1, s.pages) for i, s in enumerate(plan.selections)}
        except CorpusPlanError:
            ranks = {}
        out[target.period] = (scores, ranks)
    return out


def _consequence(repo: str, before: dict, after: dict,
                 targets: Sequence[Target]) -> tuple[str, ...]:
    lines = []
    for target in targets:
        old_scores, old_ranks = before[target.period]
        new_scores, new_ranks = after[target.period]
        label = f"{target.period[0]}–{target.period[1]}"
        old_s, new_s = old_scores.get(repo), new_scores.get(repo)
        score = (f"score {_fmt(old_s)} → {_fmt(new_s)}" if old_s != new_s
                 else f"score {_fmt(new_s)}, unchanged")
        old_r, new_r = old_ranks.get(repo), new_ranks.get(repo)
        if old_r is None and new_r is not None:
            plan = f"**would enter the plan at rank {new_r[0]}** ({new_r[1]:,} pages)"
        elif old_r is not None and new_r is None:
            plan = f"**would leave the plan** (was rank {old_r[0]}, {old_r[1]:,} pages)"
        elif old_r is not None and new_r is not None and old_r != new_r:
            plan = (f"plan rank {old_r[0]} → {new_r[0]}, "
                    f"{old_r[1]:,} → {new_r[1]:,} pages")
        elif new_r is not None:
            plan = f"stays at plan rank {new_r[0]}"
        else:
            plan = "not in the plan, before or after"
        lines.append(f"against {label}: {score}; {plan}")
    return tuple(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"
