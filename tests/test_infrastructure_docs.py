"""The infrastructure documents stay true to the code and the deploy files (#16).

A description of a machine goes stale without a word: a port moves in the unit,
a venv is renamed in make_venvs.sh, a status joins the lifecycle, and the page
still reads well. Each test here pins one fact the documents state against the
file that decides it, so the change and the stale sentence fail together.

Only the documents this issue wrote or changed are read. The repo-wide sweep of
the old host names is serving-atr-inference#136, not this.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest

from atr_training.backends import BACKENDS
from atr_training.contracts import JobStatus
from atr_training.jobstore import TRANSITIONS

REPO = Path(__file__).resolve().parents[1]
INFRA = REPO / "docs" / "INFRASTRUCTURE.md"
OPERATIONS = REPO / "docs" / "OPERATIONS.md"
README = REPO / "README.md"
ENV_EXAMPLE = REPO / ".env.example"
UNITS = sorted((REPO / "deploy" / "systemd").glob("*.service"))
MAKE_VENVS = REPO / "scripts" / "make_venvs.sh"

#: The leading document, in the serving repo, at its final path on main.
SERVING_DOC = "https://github.com/thodel/serving-atr-inference/blob/main/docs/INFRASTRUCTURE.md"
#: Its table of the values both machines must agree on.
SERVING_SHARED_TABLE = SERVING_DOC + "#shared-values"

#: The documents #16 wrote or changed; the host-name and diagram checks read these.
DOCS = (README, INFRA, OPERATIONS)
NAMED_DOCS = DOCS + (ENV_EXAMPLE,)

#: The first word of a Mermaid block that GitHub renders.
DIAGRAM_TYPES = frozenset({
    "flowchart", "graph", "sequenceDiagram", "stateDiagram-v2", "stateDiagram",
    "classDiagram", "erDiagram", "gantt", "pie", "journey", "gitGraph", "mindmap",
    "timeline",
})
FLOW_DIRECTIONS = frozenset({"TB", "TD", "BT", "LR", "RL"})
#: Words Mermaid's flowchart grammar reserves; a node with one of these ids breaks it.
RESERVED_IDS = frozenset({"end", "graph", "flowchart", "subgraph", "style", "class",
                          "classDef", "click", "linkStyle", "default", "direction"})

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── parsing helpers ─────────────────────────────────────────────────────────
def fenced_blocks(text: str) -> list[tuple[str, list[str], int]]:
    """``(info, body lines, opening line number)`` for every fenced block.

    Raises AssertionError for a fence that is never closed: everything after it
    would render as code, and a Mermaid block that swallows the rest of the page
    is exactly what this is for.
    """
    blocks: list[tuple[str, list[str], int]] = []
    fence: str | None = None
    info = ""
    body: list[str] = []
    start = 0
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if fence is None:
            if match:
                fence, info, body, start = match.group(1), match.group(2).strip(), [], number
            continue
        if match and match.group(1).startswith(fence) and not match.group(2).strip():
            blocks.append((info, body, start))
            fence = None
            continue
        body.append(line)
    assert fence is None, f"the fence opened on line {start} ({info or 'no info'}) is never closed"
    return blocks


def mermaid_blocks(path: Path) -> list[tuple[list[str], int]]:
    return [(body, start) for info, body, start in fenced_blocks(read(path)) if info == "mermaid"]


def diagram(path: Path, kind: str) -> list[str]:
    """The body of the one Mermaid block of ``kind`` in ``path``."""
    found = [body for body, _ in mermaid_blocks(path)
             if next((ln.split()[0] for ln in body if ln.strip()), "") == kind]
    assert len(found) == 1, f"{path.name}: expected one {kind} block, found {len(found)}"
    return found[0]


def section(text: str, heading: str) -> str:
    """The text under ``## heading`` up to the next ``## `` heading."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"no section '## {heading}'"
    return match.group(1)


def table_rows(text: str) -> list[list[str]]:
    """The cells of every Markdown table row in ``text``, header and rule excluded."""
    rows = []
    for line in text.splitlines():
        if not line.startswith("|") or re.match(r"^\|[\s|:-]+\|$", line):
            continue
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def unit_facts(unit: Path) -> dict[str, str]:
    text = read(unit)
    exec_start = re.search(r"^ExecStart=(.*)$", text, re.M).group(1)
    facts = {"name": unit.name, "exec_start": exec_start}
    if port := re.search(r"--port\s+(\d+)", exec_start):
        facts["port"] = port.group(1)
    if host := re.search(r"--host\s+(\S+)", exec_start):
        facts["bind"] = host.group(1)
    if venv := re.search(r"\.venvs/([^/\s]+)/bin/", exec_start):
        facts["venv"] = venv.group(1)
    if kill := re.search(r"^KillMode=(\S+)$", text, re.M):
        facts["killmode"] = kill.group(1)
    return facts


def shared_entries(text: str) -> dict[str, str]:
    """``{variable: its comment block}`` for every ``>>> SHARED <<<`` marker.

    A marker belongs to the first ``NAME=`` line after it; the header line that
    only explains the marker is not an entry.
    """
    entries: dict[str, str] = {}
    block: list[str] | None = None
    for line in text.splitlines():
        if ">>> SHARED <<<" in line and "They are marked" not in line:
            block = []
        if block is None:
            continue
        if variable := re.match(r"^([A-Z][A-Z0-9_]*)=", line):
            entries[variable.group(1)] = "\n".join(block)
            block = None
        else:
            block.append(line)
    assert block is None, "a >>> SHARED <<< marker is followed by no variable"
    return entries


def made_venvs() -> set[str]:
    match = re.search(r"^ALL=\(([^)]*)\)", read(MAKE_VENVS), re.M)
    assert match, "make_venvs.sh has no ALL=(...) list"
    return set(match.group(1).split())


def state_edges(body: list[str]) -> set[tuple[str, str]]:
    edges = set()
    for line in body:
        if match := re.match(r"^\s*(\[\*\]|\w+)\s*-->\s*(\[\*\]|\w+)\s*(?::.*)?$", line):
            edges.add((match.group(1), match.group(2)))
    return edges


def github_slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces to -."""
    text = heading.strip().lower().replace("`", "")
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


# ── the unit ────────────────────────────────────────────────────────────────
def test_there_is_a_unit_to_document():
    assert UNITS, "deploy/systemd/ holds no unit; the checks below would pass on nothing"


@pytest.mark.parametrize("unit", UNITS, ids=lambda u: u.name)
def test_every_unit_is_documented_with_its_port(unit):
    """Each unit file appears on a line of INFRASTRUCTURE.md with the port it binds."""
    facts = unit_facts(unit)
    lines = [ln for ln in read(INFRA).splitlines() if f"`{facts['name']}`" in ln]
    assert lines, f"{facts['name']} is not in docs/INFRASTRUCTURE.md"
    if "port" in facts:
        assert any(re.search(rf"(?<!\d){facts['port']}(?!\d)", ln) for ln in lines), (
            f"no line naming {facts['name']} gives its port {facts['port']}")


def test_the_documented_port_matches_the_unit():
    """The service table says what atr-train.service says: port, bind, venv, KillMode.

    And every bind or address:port written for this host in the documents is
    that port — the diagram, the README and the commands included.
    """
    facts = unit_facts(REPO / "deploy" / "systemd" / "atr-train.service")
    rows = {row[0]: row for row in table_rows(read(INFRA))
            if row and row[0] == "`atr-train.service`"}
    assert rows, "the service table has no row for atr-train.service"
    header = next(row for row in table_rows(read(INFRA)) if row[:2] == ["Unit", "Port"])
    row = dict(zip(header, rows["`atr-train.service`"]))
    assert row["Port"] == facts["port"]
    assert row["Bind"] == f"`{facts['bind']}`"
    assert row["venv"] == f"`{facts['venv']}`"
    assert row["KillMode"] == f"`{facts['killmode']}`"
    assert f"`{facts['exec_start'].replace('%h/Repo/training-atr-models/', '')}`" in read(INFRA), (
        "the ExecStart quoted in the settings table is not the unit's")

    for doc in DOCS:
        for address in re.findall(r"(?:0\.0\.0\.0|130\.92\.59\.242):(\d+)", read(doc)):
            assert address == facts["port"], f"{doc.name} names port {address} for this host"


# ── the venvs ───────────────────────────────────────────────────────────────
def test_the_documented_venvs_are_the_ones_make_venvs_builds():
    """The venv table, and the diagram's venv nodes, list what make_venvs.sh builds.

    The engine column must also be the engine each venv serves in backends.py.
    """
    built = made_venvs()
    rows = [row for row in table_rows(section(read(INFRA), "The venvs"))
            if row and re.fullmatch(r"`[a-z0-9-]+`", row[0])]
    documented = {row[0].strip("`") for row in rows}
    assert documented == built, f"docs {sorted(documented)} vs make_venvs.sh {sorted(built)}"

    engine_of = {backend.venv: backend.engine for backend in BACKENDS.values()}
    for row in rows:
        assert row[1].startswith(f"`{engine_of[row[0].strip('`')]}`"), row

    for path in (INFRA, README):
        body = "\n".join(diagram(path, "flowchart"))
        subgraph = re.search(r"subgraph venvs\[.*?\n(.*?)\n\s*end\b", body, re.S)
        assert subgraph, f"{path.name}: the machine diagram has no venvs subgraph"
        drawn = set(re.findall(r'\["([a-z0-9-]+)(?:<br/>|")', subgraph.group(1)))
        assert drawn == built, f"{path.name}: diagram {sorted(drawn)} vs {sorted(built)}"


# ── the job lifecycle ───────────────────────────────────────────────────────
def test_every_job_status_is_in_the_diagram():
    """Every value of contracts.JobStatus is a state in the stateDiagram."""
    states = {state for edge in state_edges(diagram(INFRA, "stateDiagram-v2")) for state in edge}
    missing = set(get_args(JobStatus)) - states
    assert not missing, f"JobStatus values missing from the diagram: {sorted(missing)}"
    unknown = states - set(get_args(JobStatus)) - {"[*]"}
    assert not unknown, f"the diagram draws states the code does not have: {sorted(unknown)}"


def test_the_diagram_draws_exactly_the_lifecycle():
    """The drawn transitions are jobstore.TRANSITIONS, no more and no fewer.

    Plus the entry into ``queued`` and the exit from each terminal status.
    """
    drawn = state_edges(diagram(INFRA, "stateDiagram-v2"))
    allowed = {(source, target) for source, targets in TRANSITIONS.items() for target in targets}
    terminal = {status for status, targets in TRANSITIONS.items() if not targets}
    expected = allowed | {("[*]", "queued")} | {(status, "[*]") for status in terminal}
    assert drawn - expected == set(), f"drawn but not allowed: {sorted(drawn - expected)}"
    assert expected - drawn == set(), f"allowed but not drawn: {sorted(expected - drawn)}"


def test_the_status_table_explains_every_status():
    statuses = {row[0].strip("`") for row in table_rows(section(read(INFRA), "The life of a job"))
                if row and row[0].startswith("`")}
    assert set(get_args(JobStatus)) <= statuses, sorted(set(get_args(JobStatus)) - statuses)


# ── the shared values ───────────────────────────────────────────────────────
def test_every_shared_value_is_linked():
    """Each ``>>> SHARED <<<`` entry in .env.example points at the serving table,
    and appears in this repo's table as a link to it — and nothing else does."""
    entries = shared_entries(read(ENV_EXAMPLE))
    assert entries, ".env.example has no >>> SHARED <<< entry"
    for variable, block in entries.items():
        assert SERVING_SHARED_TABLE in block, f".env.example: {variable} does not link the table"

    text = section(read(INFRA), "Values shared with idhefix")
    linked = {}
    for row in table_rows(text):
        if match := re.fullmatch(r"\[`([A-Z0-9_]+)`\]\(([^)]+)\)", row[0]):
            linked[match.group(1)] = match.group(2)
    assert set(linked) == set(entries), (
        f"table {sorted(linked)} vs .env.example {sorted(entries)}")
    assert set(linked.values()) == {SERVING_SHARED_TABLE}, linked
    count = NUMBER_WORDS[len(entries)]
    assert f"{count.capitalize()} values in this host's `.env` must agree" in text.replace("\n", " ")


# ── host names ──────────────────────────────────────────────────────────────
HOSTS = {"130.92.59.240": "idhefix", "130.92.59.242": "asteraix"}
HOST_NAME = re.compile(r"idhefix|asteraix", re.I)
#: How far from an address a host name counts as "next to" it.
NEAR = 60


def misnamed(text: str) -> list[str]:
    """Every address whose nearest host name, within NEAR characters, is the wrong one."""
    names = [(m.start(), m.end(), m.group(0).lower()) for m in HOST_NAME.finditer(text)]
    wrong = []
    for ip, host in HOSTS.items():
        for match in re.finditer(re.escape(ip) + r"(?!\d)", text):
            # Characters between the name and the address, on whichever side it is.
            distances = [(max(match.start() - end, start - match.end()), name)
                         for start, end, name in names]
            near = [(distance, name) for distance, name in distances if distance <= NEAR]
            if near and min(near)[1] != host:
                context = text[max(0, match.start() - 40):match.end() + 40].replace("\n", " ")
                wrong.append(f"{ip} next to {min(near)[1]}: …{context}…")
    return wrong


def test_the_misnaming_check_sees_a_misnaming():
    assert misnamed("the serving box asterAIx (130.92.59.240)")
    assert misnamed("idhefix is 130.92.59.242")
    assert not misnamed("asteraix admits only idhefix (130.92.59.240)")


@pytest.mark.parametrize("doc", NAMED_DOCS, ids=lambda p: p.name)
def test_the_docs_name_the_hosts_correctly(doc):
    """130.92.59.240 is idhefix and 130.92.59.242 is asteraix, wherever both appear.

    For months the serving box was called asterAIx (serving-atr-inference#136).
    """
    assert misnamed(read(doc)) == []


# ── Mermaid ─────────────────────────────────────────────────────────────────
def flowchart_problems(body: list[str]) -> list[str]:
    problems = []
    labels: dict[str, str] = {}
    for line in body[1:]:
        # HTML (&amp;) and Mermaid's own (#amp;, #35;) entity codes alike.
        if re.search(r"&#?\w+;|#\w+;", line):
            problems.append(f"entity code: {line.strip()}")
        bare = re.sub(r'"[^"]*"', '""', line)
        if '"' in bare.replace('""', ""):
            problems.append(f"unbalanced quote: {line.strip()}")
        if "&" in bare:
            problems.append(f"'&' chaining: {line.strip()}")
        if "|" in bare:
            problems.append(f"pipe label, quote it as -- \"...\" --> instead: {line.strip()}")
        if re.search(r"--\s+[^\s\"-]", bare):
            problems.append(f"unquoted edge label: {line.strip()}")
        for opener in re.finditer(r"[\[({]+", bare):
            if not bare[opener.end():].startswith('"'):
                problems.append(f"unquoted node label: {line.strip()}")
        for node in re.finditer(r'(?:^|\s)(?:subgraph\s+)?([A-Za-z_][\w-]*)[\[({]+("[^"]*")', line):
            node_id, label = node.group(1), node.group(2)
            if node_id in RESERVED_IDS:
                problems.append(f"reserved word as node id: {node_id}")
            if labels.setdefault(node_id, label) != label:
                problems.append(f"{node_id} is defined twice with different labels")
    return problems


def state_problems(body: list[str]) -> list[str]:
    problems = []
    for line in body[1:]:
        if not line.strip():
            continue
        match = re.match(r"^\s*(\[\*\]|\w+)\s*-->\s*(\[\*\]|\w+)\s*(?::(.*))?$", line)
        if not match:
            problems.append(f"not a plain transition: {line.strip()}")
            continue
        label = match.group(3) or ""
        if re.search(r"[()<>{}\"#;&]", label):
            problems.append(f"label with a character the state grammar trips on: {line.strip()}")
    return problems


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_mermaid_block_is_closed(doc):
    """A cheap syntax probe: fence closed, known diagram type, and the house rules
    that keep GitHub's renderer happy. The rendering itself is the PR preview's."""
    for body, start in mermaid_blocks(doc):
        first = next((ln.split() for ln in body if ln.strip()), [""])
        where = f"{doc.name}:{start}"
        assert first[0] in DIAGRAM_TYPES, f"{where}: unknown diagram type {first[0]!r}"
        if first[0] in ("flowchart", "graph"):
            assert len(first) > 1 and first[1] in FLOW_DIRECTIONS, f"{where}: no direction"
            assert flowchart_problems(body) == [], where
        if first[0].startswith("stateDiagram"):
            assert state_problems(body) == [], where


def test_the_mermaid_checks_see_what_they_are_for():
    assert flowchart_problems(["flowchart TB", "  a[unquoted] --> b"])
    assert flowchart_problems(["flowchart TB", '  a["x"] -- label --> b["y"]'])
    assert flowchart_problems(["flowchart TB", '  a["x"] & c["z"] --> b["y"]'])
    assert flowchart_problems(["flowchart TB", '  end["x"] --> b["y"]'])
    assert flowchart_problems(["flowchart TB", '  a["x"]', '  a["y"]'])
    assert flowchart_problems(["flowchart TB", '  a["x &amp; y"]'])
    assert flowchart_problems(["flowchart TB", '  a["x #quot;y#quot;"]'])
    assert not flowchart_problems(["flowchart TB", '  s[("x")] -- "l" --> b["y<br/>z"]'])
    assert state_problems(["stateDiagram-v2", "  a --> b: cancel (SIGTERM)"])
    assert not state_problems(["stateDiagram-v2", "  [*] --> a: POST /jobs", "  a --> [*]"])
    with pytest.raises(AssertionError, match="never closed"):
        fenced_blocks("```mermaid\nflowchart TB\n")


def test_the_readme_shows_the_machine_diagram_of_the_document():
    """The README's entry point is a copy; a copy that drifts is worse than none."""
    assert diagram(README, "flowchart") == diagram(INFRA, "flowchart")


# ── links ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_link_between_the_docs_resolves(doc):
    """Relative links point at files that exist, and anchors at headings they have.

    The serving document is linked by URL and not checked here: it lives in
    another repository.
    """
    text = re.sub(r"```.*?```", "", read(doc), flags=re.S)
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://")):
            if "serving-atr-inference" in target and "INFRASTRUCTURE.md" in target:
                assert target in (SERVING_DOC, SERVING_SHARED_TABLE), target
            continue
        path, _, anchor = target.partition("#")
        linked = (doc.parent / path).resolve() if path else doc
        assert linked.exists(), f"{doc.name}: {target} does not exist"
        if anchor:
            headings = re.findall(r"^#{1,6} (.+)$", read(linked), re.M)
            assert anchor in {github_slug(h) for h in headings}, (
                f"{doc.name}: {linked.name} has no heading for #{anchor}")
