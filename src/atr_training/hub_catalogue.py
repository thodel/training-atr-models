"""Read an org's datasets off the HuggingFace Hub: card, page count, size (#87, #32).

Shared by ``scripts/plan_corpus.py`` (what is the best corpus now) and
``scripts/discover_datasets.py`` (what changed since the last look). Both need the
same three facts per dataset, and two copies of how to read them would drift.

Needs ``huggingface_hub`` and, for the gated dh-unibe datasets, a logged-in token
— so it runs in a trainer venv (``.venvs/kraken-train``), never in the gateway's.

What comes back separates three things a single list would blur:

* ``listed`` — every dataset the Hub names for the org;
* ``rows`` — the ones whose card could be read, with what it says;
* ``unreadable`` — listed, but the card could not be fetched this time, and why.

A dataset in ``unreadable`` is not gone. Treating "could not read it today" as
"it disappeared" is how a discovery run would report the whole org removed the
morning the network is down.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from atr_training.corpus_plan import parse_projects

__all__ = ["Catalogue", "fetch_catalogue"]


@dataclass
class Catalogue:
    org: str
    listed: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    unreadable: dict[str, str] = field(default_factory=dict)


def fetch_catalogue(org: str, *, log=sys.stderr) -> Catalogue:
    """Every dataset of ``org``, with its card text, page count and size.

    Three Hub calls per dataset (list, card, info); the Hub allows 1,000 per five
    minutes, and ``dh-unibe`` held 49 datasets on 22.10.2026 — 29 s for all.
    """
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415 — trainer venvs only

    api = HfApi()
    catalogue = Catalogue(org=org)
    for info in api.list_datasets(author=org):
        catalogue.listed.append(info.id)
        try:
            path = hf_hub_download(info.id, "README.md", repo_type="dataset")
            card = Path(path).read_text(encoding="utf-8", errors="replace")
            detail = api.dataset_info(info.id, files_metadata=False)
        except Exception as exc:  # noqa: BLE001 — gated, no card, offline: say which
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            catalogue.unreadable[info.id] = reason
            print(f"  skip {info.id}: {reason}", file=log)
            continue
        card_data = getattr(detail, "cardData", None) or {}
        dataset_info = card_data.get("dataset_info") or {}
        if isinstance(dataset_info, list):
            dataset_info = dataset_info[0] if dataset_info else {}
        splits = dataset_info.get("splits") or []
        catalogue.rows.append({
            "repo": info.id,
            "card": card,
            "pages": sum(s.get("num_examples") or 0 for s in splits),
            "gb": round(sum(s.get("num_bytes") or 0 for s in splits) / 1024**3, 1),
            "projects": list(parse_projects(card)),
        })
    return catalogue
