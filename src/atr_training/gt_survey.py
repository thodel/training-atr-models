"""Eine fremde Quelle erheben, ohne sie einzulesen (#200, Schritt 1).

Der Importweg (#187, :mod:`atr_training.foreign_gt`) liest nur Quellen aus dem
Register, und das Register wächst per PR. Diese Erhebung nimmt einem Menschen die
Recherche ab, die vor so einem PR steht: Wo steht die Lizenz, sagen die Fundstellen
dasselbe, wie gross ist die Quelle, liegen Bilder dabei, welcher Stand wird gelesen.
Sie lädt nichts hoch, schreibt nichts und urteilt nicht.

**Sie urteilt nicht** heisst: Bei Widerspruch nennt sie beide Fundstellen und setzt
im Entwurf keine Lizenz. Bei den 24 Quellen aus #186 widersprachen sich vier selbst
(#191), und eine überschrieb pauschal fremdes Material mit CC BY 4.0. Welche Angabe
gilt, klärt eine Mail, nicht ein Abruf.

**Nur GitHub und Zenodo.** Die Erhebung läuft im Trainer-Dienst, und ein Dienst,
der jede URL abruft, die ihm jemand schickt, ist ein Weg ins interne Netz. Die zwei
Hosts, auf denen alle 24 Quellen lagen, genügen; alles andere wird abgelehnt.

Die Lizenz wird an diesen Stellen gesucht:

* GitHub: die Lizenzerkennung der API (``license.spdx_id``), jede ``LICENSE*``-,
  ``LICENCE*``- oder ``COPYING*``-Datei im Wurzelverzeichnis, ``METADATA.yml``,
  ``.zenodo.json``, ``htr-united.yml`` und ``CITATION.cff``.
* Zenodo: ``metadata.license.id`` der API und jede ``LICENSE*``-Datei der
  Hinterlegung. Was in den Archiven liegt, liest erst der Import (er lädt sie
  ohnehin); die Erhebung sagt das.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable

from atr_training.foreign_gt_zenodo import IMAGE_SUFFIXES, licence_from_text, spdx

#: ``fetch(url) -> (HTTP-Status, Inhalt)``. Status 0 heisst: keine Antwort.
Fetch = Callable[[str], "tuple[int, bytes]"]

TIMEOUT_S = 20
MAX_BYTES = 2 * 1024 * 1024
USER_AGENT = "atr-training-gt-survey (+https://github.com/thodel/training-atr-models)"

_GITHUB = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)"
    r"(?:\.git)?(?:/tree/(?P<branch>[A-Za-z0-9_./-]+?))?/?$")
_ZENODO = re.compile(
    r"^(?:https://(?:zenodo\.org/records?/|doi\.org/10\.5281/zenodo\.)|10\.5281/zenodo\.)"
    r"(?P<record>[0-9]{1,12})/?$")

#: Dateien im Wurzelverzeichnis eines Repos, in denen Quellen ihre Lizenz nennen.
METADATA_FILES = ("METADATA.yml", ".zenodo.json", "htr-united.yml", "CITATION.cff")
_LICENCE_FILE = re.compile(r"^(LICEN[CS]E|COPYING)(\.[A-Za-z0-9]+)?$", re.IGNORECASE)
#: Hinweise auf einen Bildweg ausserhalb des Repos: Bezugsskripte und URL-Listen.
_IMAGE_HINT = re.compile(r"(get_?images|download|fetch|urls?|images?)[^/]*\.(sh|txt|lst|py|csv)$",
                         re.IGNORECASE)

#: SPDX-Schreibweisen, wie Metadaten sie tragen (``CC-BY-4.0``, ``cc-by-sa-4.0``,
#: ``CC0-1.0``), vom Spezifischen zum Allgemeinen.
_SPDX = (
    (r"\bCC[- ]?BY[- ]NC[- ]SA[- ]4\.0\b", "CC-BY-NC-SA-4.0"),
    (r"\bCC[- ]?BY[- ]NC[- ]ND[- ]4\.0\b", "CC-BY-NC-ND-4.0"),
    (r"\bCC[- ]?BY[- ]NC[- ]4\.0\b", "CC-BY-NC-4.0"),
    (r"\bCC[- ]?BY[- ]ND[- ]4\.0\b", "CC-BY-ND-4.0"),
    (r"\bCC[- ]?BY[- ]SA[- ]4\.0\b", "CC-BY-SA-4.0"),
    (r"\bCC[- ]?BY[- ]4\.0\b", "CC-BY-4.0"),
    (r"\bCC0(?:[- ]1\.0)?\b|\bcc-zero\b", "CC0-1.0"),
    (r"\bMIT\b", "MIT"),
    (r"\bApache[- ]2\.0\b|Apache License,? Version 2\.0", "Apache-2.0"),
)


class SurveyError(ValueError):
    """Eine URL, die die Erhebung nicht annimmt."""


def default_fetch(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json, */*"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310
            return response.status, response.read(MAX_BYTES)
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, b""


def licence_from_any(text: str) -> str | None:
    """Die Lizenz, die ein Text nennt: ein Lizenztext oder eine SPDX-artige Kennung."""
    named = licence_from_text(text)
    if named:
        return named
    for pattern, licence in _SPDX:
        if re.search(pattern, text, re.IGNORECASE):
            return licence
    return None


def _metadata_licence(name: str, text: str) -> str | None:
    """Die Lizenz aus einer Metadatei; nur die Zeilen, die von Lizenz handeln."""
    if name == ".zenodo.json":
        try:
            value = json.loads(text).get("license")
        except (ValueError, AttributeError):
            return None
        if isinstance(value, dict):
            value = value.get("id")
        return spdx(value) if isinstance(value, str) and value else None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*-?\s*licen[cs]e", line, re.IGNORECASE):
            found = licence_from_any(" ".join(lines[i:i + 4]))
            if found:
                return found
    return None


@dataclass
class Finding:
    at: str
    licence: str | None
    excerpt: str = ""

    def as_dict(self) -> dict:
        return {"at": self.at, "licence": self.licence, "excerpt": self.excerpt}


def _excerpt(text: str) -> str:
    return " ".join(text.split())[:200]


def _judge(findings: list[Finding]) -> dict:
    """Was die Fundstellen zusammen sagen — beschrieben, nicht entschieden."""
    named = sorted({f.licence for f in findings if f.licence})
    unread = [f.at for f in findings if not f.licence]
    if not findings:
        state = "none"
    elif len(named) > 1:
        state = "contradiction"
    elif not named:
        state = "unrecognised"
    elif unread:
        state = "partly_unrecognised"
    else:
        state = "agree"
    return {"state": state, "licences": named, "unrecognised_at": unread}


def _json(fetch: Fetch, url: str) -> dict | list | None:
    status, body = fetch(url)
    if status != 200:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


# ── GitHub ──────────────────────────────────────────────────────────────────

def _survey_github(match: re.Match, fetch: Fetch) -> dict:
    owner, repo = match["owner"], match["repo"]
    if set(owner) == {"."} or set(repo) == {"."}:
        raise SurveyError(f"github.com/{owner}/{repo}: not a repository name")
    api = f"https://api.github.com/repos/{owner}/{repo}"
    meta = _json(fetch, api)
    if not isinstance(meta, dict):
        raise SurveyError(f"github.com/{owner}/{repo}: the GitHub API does not know it "
                          "(private, renamed, or not there)")
    branch = match["branch"] or meta.get("default_branch") or "main"
    commit = _json(fetch, f"{api}/commits/{branch}")
    sha = commit.get("sha") if isinstance(commit, dict) else None
    if not sha:
        raise SurveyError(f"github.com/{owner}/{repo}: no commit on branch {branch!r}")
    tree = _json(fetch, f"{api}/git/trees/{sha}?recursive=1")
    entries = tree.get("tree", []) if isinstance(tree, dict) else []

    findings: list[Finding] = []
    spdx_id = (meta.get("license") or {}).get("spdx_id")
    if spdx_id and spdx_id != "NOASSERTION":
        findings.append(Finding("GitHub-API `license.spdx_id`", spdx(spdx_id), spdx_id))
    elif meta.get("license"):
        findings.append(Finding("GitHub-API `license.spdx_id`", None, str(spdx_id)))

    root_files = [e["path"] for e in entries
                  if e.get("type") == "blob" and "/" not in e.get("path", "")]
    wanted = [p for p in root_files if _LICENCE_FILE.match(p)]
    wanted += [p for p in root_files if p.lower() in {m.lower() for m in METADATA_FILES}]
    for path in wanted:
        status, body = fetch(f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}")
        if status != 200:
            findings.append(Finding(f"`{path}`", None, f"HTTP {status or 'keine Antwort'}"))
            continue
        text = body.decode("utf-8", "replace")
        licence = (licence_from_any(text) if _LICENCE_FILE.match(path)
                   else _metadata_licence(path, text))
        findings.append(Finding(f"`{path}`", licence, _excerpt(text)))

    blobs = [e for e in entries if e.get("type") == "blob"]
    paths = [e["path"] for e in blobs]
    xml = [p for p in paths if p.lower().endswith(".xml")]
    images = [p for p in paths if p.lower().endswith(IMAGE_SUFFIXES)]
    symlinks = [e["path"] for e in blobs if e.get("mode") == "120000"]
    hints = sorted(p for p in paths if _IMAGE_HINT.search(PurePosixPath(p).name))[:20]
    licence = _judge(findings)
    draft = _github_draft(owner, repo, branch, licence, findings)
    return {
        "kind": "github", "url": f"https://github.com/{owner}/{repo}",
        "branch": branch, "default_branch": meta.get("default_branch"), "commit": sha,
        "pushed_at": meta.get("pushed_at"),
        "licence": licence, "found": [f.as_dict() for f in findings],
        "size": {"files": len(paths), "xml": len(xml), "images": len(images),
                 "symlinks": len(symlinks), "truncated": bool((tree or {}).get("truncated"))},
        "images": {"in_repo": len(images),
                   "hints": hints,
                   "note": ("Bilder liegen im Repo" if images else
                            "keine Bilder im Repo — der Bildweg ist zu klären (#193, #199)")},
        "draft": draft,
    }


def _github_draft(owner: str, repo: str, branch: str, licence: dict,
                  findings: list[Finding]) -> str:
    agreed = licence["state"] == "agree"
    lic = licence["licences"][0] if agreed else ""
    at = "; ".join(f"{f.at} ({f.licence})" for f in findings) if agreed else ""
    slug = re.sub(r"[^a-z0-9-]+", "-", repo.lower()).strip("-")
    return "\n".join([
        "Source(",
        f'    id="{repo}",',
        f'    origin="https://github.com/{owner}/{repo}",',
        f'    clone_url="https://github.com/{owner}/{repo}.git",',
        f'    branch="{branch}",',
        '    xml_root="",',
        f'    licence="{lic}",' + ("" if agreed else
                                    "  # nicht gesetzt: " + licence["state"]),
        f'    licence_at="{at}",',
        '    attribution="",  # wer transkribiert hat, und wo die Quelle das sagt',
        '    image_source="",',
        '    script_kind="",',
        '    period="",',
        f'    project="{slug}",',
        f'    target="dh-unibe/image-text_{slug}",',
        "    images=(),  # Bildweg: ImageUrlList / ImageInXml / ImageTemplate / ImageShellScript",
        "),",
    ])


# ── Zenodo ──────────────────────────────────────────────────────────────────

def _survey_zenodo(match: re.Match, fetch: Fetch) -> dict:
    asked = match["record"]
    meta = _json(fetch, f"https://zenodo.org/api/records/{asked}")
    if not isinstance(meta, dict):
        raise SurveyError(f"zenodo {asked}: the Zenodo API does not know it")
    record = str(meta.get("id") or asked)
    doi = meta.get("doi") or f"10.5281/zenodo.{record}"
    concept = meta.get("conceptdoi") or ""
    md = meta.get("metadata") or {}
    findings: list[Finding] = []
    lic = (md.get("license") or {}).get("id")
    findings.append(Finding("Zenodo-API `metadata.license.id`",
                            spdx(lic) if lic else None, lic or "—"))
    files = meta.get("files") or []
    for f in files:
        key = f.get("key", "")
        if _LICENCE_FILE.match(PurePosixPath(key).name):
            status, body = fetch(f"https://zenodo.org/records/{record}/files/{key}?download=1")
            text = body.decode("utf-8", "replace") if status == 200 else ""
            findings.append(Finding(f"`{key}` (Datei der Hinterlegung)",
                                    licence_from_any(text) if text else None,
                                    _excerpt(text) or f"HTTP {status or 'keine Antwort'}"))
    sizes = [(f.get("key", ""), int(f.get("size") or 0)) for f in files]
    licence = _judge(findings)
    notes = ["`LICENSE*` in den Archiven liest erst der Import; er lädt sie ohnehin."]
    if record != asked:
        notes.append(f"{asked} ist nicht die Version: die API antwortete mit {record}. "
                     "Ins Register gehört die versionierte DOI.")
    if concept and concept == doi:
        notes.append(f"{doi} ist der Konzept-DOI; der Import verweigert ihn.")
    related = [r for r in md.get("related_identifiers") or []
               if r.get("relation") in ("isVersionOf", "hasVersion", "isVariantFormOf",
                                        "hasVariantForm", "isIdenticalTo")]
    return {
        "kind": "zenodo", "url": f"https://zenodo.org/records/{record}",
        "record": record, "doi": doi, "conceptdoi": concept,
        "title": md.get("title"), "version": md.get("version"),
        "publication_date": md.get("publication_date"),
        "creators": [c.get("name") for c in md.get("creators") or [] if c.get("name")],
        "licence": licence, "found": [f.as_dict() for f in findings],
        "size": {"files": len(sizes), "bytes": sum(s for _, s in sizes),
                 "list": [{"key": k, "bytes": s} for k, s in sizes]},
        "related": [{"relation": r.get("relation"), "identifier": r.get("identifier")}
                    for r in related],
        "notes": notes,
        "draft": _zenodo_draft(record, doi, md, licence, findings, sizes),
    }


def _zenodo_draft(record: str, doi: str, md: dict, licence: dict,
                  findings: list[Finding], sizes: list[tuple[str, int]]) -> str:
    agreed = licence["state"] == "agree"
    lic = licence["licences"][0] if agreed else ""
    at = "; ".join(f"{f.at} ({f.licence})" for f in findings) if agreed else ""
    slug = re.sub(r"[^a-z0-9-]+", "-", (md.get("title") or record).lower()).strip("-")[:40]
    archives = ", ".join(f'"{k}"' for k, _ in sizes if k.lower().endswith(".zip"))
    return "\n".join([
        "ZenodoRecord(",
        f'    id="{slug}",',
        f'    record="{record}",',
        f'    doi="{doi}",',
        f'    licence="{lic}",' + ("" if agreed else "  # nicht gesetzt: " + licence["state"]),
        f'    licence_at="{at}",',
        '    attribution="",  # die Creators der Hinterlegung nennen, wo die Quelle sie nennt',
        '    image_source="",',
        '    script_kind="",',
        '    period="",',
        f'    project="{slug}",',
        f'    target="dh-unibe/image-text_{slug}",',
        f'    parts=(ZenodoPart("alle", ({archives + "," if archives else ""})),),',
        "),",
    ])


def survey(url: str, fetch: Fetch = default_fetch) -> dict:
    """Lizenz, Fundstellen, Umfang, Bildweg und Stand einer Quelle; ein Register-Entwurf.

    Liest nur. :class:`SurveyError` für eine URL, die weder ein GitHub-Repo noch eine
    Zenodo-Hinterlegung ist, oder die der Host nicht kennt.
    """
    text = (url or "").strip()
    if (m := _GITHUB.match(text)) is not None:
        return _survey_github(m, fetch)
    if (m := _ZENODO.match(text)) is not None:
        return _survey_zenodo(m, fetch)
    raise SurveyError(
        f"{url!r}: the survey reads a GitHub repository (https://github.com/<owner>/<repo>) "
        "or a Zenodo record (https://zenodo.org/records/<id>, 10.5281/zenodo.<id>), "
        "nothing else")
