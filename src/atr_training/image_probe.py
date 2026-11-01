"""Sehen, dass Bildwege altern, bevor jemand sie braucht (#199).

Zwei von sechs Quellen hatten beim ersten Lauf einen toten Bildweg (08.10.2026):
Weisthuemer nagelte einen archive.org-Knoten fest, der nicht mehr antwortete, und
dach-gt trug in 27 Dokumenten Bild-URLs mit HTTP 404. Gefunden hat beides erst
der Import — also der Moment, in dem es zu spät ist.

Dieser Prüflauf ruft je Quelle **fünf** Bild-URLs ab, gleichmässig über die
Seiten gezogen, erste und letzte dabei. Fünf hätten beide Fälle gefunden; alle
990 Bilder zu holen wäre Last auf fremden Servern für keinen Erkenntnisgewinn.

**Vier Lagen, nicht "n von m nicht erreichbar".** Jede hat eine andere Antwort,
und eine Meldung, die sie zusammenwirft, ist der dritte Zustand aus #165 in neuer
Verkleidung:

    unreachable  der Knoten antwortet nicht (http=000): Host gewechselt?
                 Eine Umschreibung ins Register (``foreign_gt.REWRITES``).
    404          die Datei ist weg: Seiten fallen beim Import heraus; die Quelle
                 fragen, ob es einen neuen Ort gibt.
    403          der Zugang hat sich geändert: die Bedingungen der Einrichtung
                 neu lesen, bevor wieder geholt wird (#193).
    timeout      der Server ist langsam oder überlastet: beim nächsten Lauf
                 wieder ansehen, kein Eingriff.

Dazu ``listing_failed``: die Quelle liess sich gar nicht erst auflisten (Klon
oder Register kaputt) — auch das ist nicht "alles in Ordnung".

Berichtet wird nur, was sich gegen den letzten Lauf verändert hat.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "ADVICE",
    "Change",
    "OK",
    "SAMPLE_SIZE",
    "Snapshot",
    "classify_error",
    "diff",
    "probe",
    "render",
    "sample",
]

OK = "ok"
NOT_FOUND = "404"
FORBIDDEN = "403"
TIMEOUT = "timeout"
UNREACHABLE = "unreachable"
LISTING_FAILED = "listing_failed"

#: Je Quelle so viele URLs. Fünf hätten Weisthuemer und dach-gt beide gefunden.
SAMPLE_SIZE = 5

#: Was auf jede Lage zu tun ist — steht im Bericht neben der Lage.
ADVICE = {
    UNREACHABLE: "Knoten antwortet nicht — Host gewechselt? Eine Umschreibung in "
                 "`foreign_gt.REWRITES` eintragen, keine neue Funktion.",
    NOT_FOUND: "Datei weg — diese Seiten fallen beim Import heraus; die Quelle "
               "fragen, ob die Bilder einen neuen Ort haben.",
    FORBIDDEN: "Zugang geändert — die Bedingungen der Einrichtung neu lesen, bevor "
               "wieder geholt wird (#193).",
    TIMEOUT: "Server langsam oder überlastet — beim nächsten Lauf wieder ansehen.",
    LISTING_FAILED: "Quelle liess sich nicht auflisten — Klon oder Register prüfen.",
    "invalid_url": "keine gültige URL — der Bildweg im Register ist falsch.",
}

UA = "training-atr-models/image-probe (DH Bern; Forschung)"


def sample(urls: Mapping[str, str], n: int = SAMPLE_SIZE) -> list[tuple[str, str]]:
    """``n`` Paare ``(Seite, URL)``, gleichmässig über die nach Pfad sortierten Seiten.

    Deterministisch und mit erster und letzter Seite: zwei Läufe prüfen dieselben
    URLs, sonst wäre eine Veränderung von einem anderen Los nicht zu unterscheiden.
    """
    items = sorted(urls.items())
    if len(items) <= n:
        return items
    if n <= 1:
        return items[:1]
    picks = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    return [items[i] for i in picks]


def classify_error(exc: BaseException) -> str:
    """Die Lage, die ein gescheiterter Abruf bedeutet."""
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404 or exc.code == 410:
            return NOT_FOUND
        if exc.code in (401, 403):
            return FORBIDDEN
        return f"http_{exc.code}"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return TIMEOUT
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason):
            return TIMEOUT
        return UNREACHABLE
    if isinstance(exc, (ConnectionError, OSError)):
        return UNREACHABLE
    if isinstance(exc, ValueError):
        return "invalid_url"          # ein Registerfehler, keine Alterung
    return f"error_{type(exc).__name__}"


def _fetch_head_bytes(url: str, timeout: float) -> int:
    """Die ersten Bytes holen; der Status. Range, damit kein ganzes Bild kommt."""
    request = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-1023"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read(1024)
        return response.status


def probe(url: str, *, timeout: float = 30,
          fetch: Callable[[str, float], int] = _fetch_head_bytes) -> str:
    """Eine URL, eine Lage. Ein 2xx ist ``ok``; alles andere sagt, was es ist."""
    try:
        status = fetch(url, timeout)
    except Exception as exc:  # noqa: BLE001 — jede Ausnahme ist eine Lage, keine Panne
        return classify_error(exc)
    return OK if 200 <= status < 300 else f"http_{status}"


@dataclass
class Snapshot:
    """Was ein Lauf gesehen hat: je Quelle ``{URL: Lage}``."""

    taken_at: str
    sources: dict[str, dict[str, str]] = field(default_factory=dict)

    def dumps(self) -> str:
        return json.dumps({"taken_at": self.taken_at, "sources": self.sources},
                          ensure_ascii=False, indent=1, sort_keys=True)

    @classmethod
    def loads(cls, text: str) -> "Snapshot":
        data = json.loads(text)
        return cls(taken_at=data["taken_at"], sources=data.get("sources", {}))

    @staticmethod
    def now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass(frozen=True)
class Change:
    source_id: str
    before: dict[str, str]
    after: dict[str, str]

    @property
    def counts_before(self) -> Counter:
        return Counter(self.before.values())

    @property
    def counts_after(self) -> Counter:
        return Counter(self.after.values())


def diff(previous: Snapshot | None, current: Snapshot) -> list[Change]:
    """Die Quellen, deren Lage sich verändert hat.

    Eine Quelle ändert sich, wenn eine geprüfte URL eine andere Lage hat oder die
    Stichprobe eine andere ist. Ohne früheren Lauf ist "vorher" leer: dann
    erscheinen genau die Quellen, die nicht vollständig ``ok`` sind — eine tote
    Quelle beim ersten Lauf ist der Fall, für den es diesen Lauf gibt.
    """
    changes = []
    before_all = previous.sources if previous else {}
    for source_id in sorted(set(before_all) | set(current.sources)):
        before = before_all.get(source_id, {})
        after = current.sources.get(source_id, {})
        if previous is None:
            if after and all(state == OK for state in after.values()):
                continue
        elif before == after:
            continue
        changes.append(Change(source_id, before, after))
    return changes


def _summary(states: Mapping[str, str]) -> str:
    if not states:
        return "—"
    counts = Counter(states.values())
    ok = counts.pop(OK, 0)
    rest = ", ".join(f"{n}× {state}" for state, n in sorted(counts.items()))
    return f"{ok}/{len(states)} ok" + (f" ({rest})" if rest else "")


def render(changes: Sequence[Change], previous: Snapshot | None,
           current: Snapshot) -> str:
    """Der Bericht, oder ``""`` wenn sich nichts verändert hat."""
    if not changes:
        return ""
    since = previous.taken_at if previous else "dem ersten Lauf (keine frühere Messung)"
    lines = [f"# Bildwege: Veränderungen seit {since}", "",
             f"Gemessen {current.taken_at}, je Quelle {SAMPLE_SIZE} URLs (#199).", "",
             "| Quelle | vorher | jetzt |", "|---|---|---|"]
    for change in changes:
        lines.append(f"| `{change.source_id}` | {_summary(change.before)} | "
                     f"{_summary(change.after)} |")
    seen_states: set[str] = set()
    for change in changes:
        bad = {url: state for url, state in change.after.items() if state != OK}
        if not bad:
            continue
        lines += ["", f"## {change.source_id}", ""]
        for url, state in sorted(bad.items(), key=lambda kv: (kv[1], kv[0])):
            lines.append(f"- **{state}** — `{url}`")
            seen_states.add(state)
    if seen_states:
        lines += ["", "## Was zu tun ist", ""]
        for state in sorted(seen_states):
            advice = ADVICE.get(state, "unerwarteter HTTP-Status — von Hand ansehen.")
            lines.append(f"- **{state}**: {advice}")
    return "\n".join(lines) + "\n"
