# Fremde Ground Truth einlesen

Wie aus einem fremden GitHub-Repo oder einer Zenodo-Hinterlegung ein HF-Datensatz
wird, den `prepare` lesen kann — und welche Fallen dabei stumm zuschlagen.

Quellenerhebung: Epic [#186](https://github.com/thodel/training-atr-models/issues/186),
24 Quellen am 08.10.2026 einzeln lizenzgeprüft. Dieser Weg: [#187](https://github.com/thodel/training-atr-models/issues/187).

```bash
.venvs/kraken-train/bin/python scripts/foreign_gt_to_hf.py --list
.venvs/kraken-train/bin/python scripts/foreign_gt_to_hf.py reichsanzeiger-gt --dry-run
.venvs/kraken-train/bin/python scripts/foreign_gt_to_hf.py reichsanzeiger-gt
```

## 1. Was eingelesen ist

Sechs CC0-Quellen, alle am 08.10.2026 gemessen — gezählte `TextLine`-Elemente, nicht
Angaben aus den READMEs:

| Datensatz | Seiten | Zeilen | Bilder | MB | Bildweg |
|---|---:|---:|---:|---:|---|
| `image-text_reichsanzeiger-gt` | 101 | **119.431** | 101 | 910 | URL-Liste (UB Mannheim) |
| `image-text_gt-fraktur` | 207 | 14.617 | 207 | 132 | Vorlage (opendigi Tübingen) |
| `image-text_fibeln` | 409 | 8.895 | 409 | 271 | XML + Vorlage (GEI, Göttingen) |
| `image-text_dtgt` | 177 | 6.599 | 177 | 92 | URL im XML (UB Tübingen) |
| `image-text_dach-gt` | 71 | ~2.000 | 71 | 48 | URL im XML + Vorlage |
| `image-text_weisthuemer` | 25 | ~1.400 | 25 | 13 | Shell-Skript (archive.org) |
| **zusammen** | **990** | **~153.000** | 990 | **1.465** | |

Alle sechs sind **privat**. Die Transkriptionen sind CC0; die **Bildrechte sind
ungeprüft**, und das ist eine andere Frage mit anderen Rechteinhabern
([#193](https://github.com/thodel/training-atr-models/issues/193)).

## 2. Die vier Bildwege, und warum es vier sind

Keine Abstraktion, sondern die vier Formen, die in den Quellen wirklich vorkommen.
`Source.images` ist ein Tupel; die Wege werden in Reihenfolge versucht, der erste
mit einer URL gewinnt. Fibeln und dach-gt brauchen zwei nebeneinander, weil ihr
Bezugsweg **je Unterverzeichnis** verschieden ist.

**`ImageUrlList`** — `<fern> <lokal>`-Paare plus eine Basis-URL, die das Repo
base64-kodiert in seinem eigenen `download_images.sh` hält. Wir entschlüsseln
denselben Block statt den Host zu hartkodieren, **und prüfen ihn gegen den im
Register notierten**: eine Quelle, die ihre Bilder woanders herholt, ist eine
Quelle, deren Bildrechte neu zu prüfen sind. Das darf nicht stumm durchlaufen.

**`ImageShellScript`** — handgeschriebene `curl -o NAME URL`-Zeilen. Beide Formen
werden erkannt, `curl -Lo` und `curl -L -o`; ein Regex, der nur die zweite kennt,
verliert die Hälfte. **Schleifenkörper werden nicht expandiert**: eine falsch
expandierte Schleife holt die falsche Seite, ohne dass es auffiele, und eine Seite
ohne URL ist ein Zustand, den man im Bericht sieht. Bei Weisthuemer stehen zehn der
35 Seiten in zwei Schleifen und fehlen deshalb.

**`ImageInXml`** — die URL steht im PAGE-Dokument, als `externalRef`, wo OCR-D sie
hinschreibt. Gemessen: alle 182 DTGT-Dokumente tragen eine, 147 von 162 bei dach-gt,
41 von 453 bei Fibeln, keines bei gt-fraktur. **Das war ein Fehlschluss von mir:**
DTGT hat kein Bezugs*skript*, und daraus hatte ich „kein Bezugsweg" gemacht. Das
Skript fehlte, der Weg nicht.

**`ImageTemplate`** — eine URL-Vorlage je Sammlung, gefüllt aus Pfad und Dokument.
Platzhalter `{stem}`, `{base}` (Stamm ohne `_NNN`), `{top}`, `{dir}`, `{img}`,
`{imgbase}`, `{imgext}`. `only=` begrenzt sie auf ein Unterverzeichnis — bei dach-gt
holt nur DE-17 aus Darmstadt. Eine Vorlage ohne `{img…}` liest das Dokument gar
nicht; bei Fibeln wären das 409 Dateizugriffe für nichts.

Noch nicht gebaut: **ALTO** und die **Zenodo-ZIP-Form**.

## 3. Die Fallen, und wie jede gefunden wurde

Alle fünf fälschen die Zahlen nach oben, und keine fällt durch einen Absturz auf.
Sie stehen hier, weil jede von ihnen durch ein *Missverhältnis im eigenen Bericht*
entdeckt wurde und nicht durch einen Fehler.

**Dubletten, nicht byte-identisch.** `reichsanzeiger-gt` liefert dieselben 101
Seiten zweimal, mit und ohne `TableRegion`. Die beiden Fassungen unterscheiden
sich, also fängt sie **kein Hash-Vergleich**. Ungefiltert gezählt: 238.862 Zeilen
statt 119.431 — genau das Doppelte, und das Doppelte war der Beweis. Dasselbe bei
`gt-fraktur` (PAGE und ALTO) und `charlottenburger` (mit und ohne Tabellen).

**Symlinks, die wie Dokumente heissen.** `dach-gt` hält 173 Einträge unter
`data/DE-12/.../alto/`, die auf ein nie eingechecktes `gt/` zeigen. Wer nach
`*.xml` filtert, liest 173 einzeilige Textdateien als Ground Truth. Darum filtert
`tracked_pages` nach **Git-Modus** (`120000`), nicht nach Endung.

**Ein wiederkehrender Dateiname ist keine Dublette.** Fibeln hält in **jedem** der
sechs PPN-Verzeichnisse eine `00000001.xml`. Nach Seitenstamm dedupliziert fielen
243 von 409 Seiten als „Dubletten" heraus — verschiedene Werke. Der Schlüssel ist
darum der **Pfad mit entfernten Variantenmarkern** (`Source.variants`), nicht der
Dateiname.

**Dieselbe Verwechslung eine Schicht tiefer, und folgenschwerer.** Die URL-Tabelle
war ebenfalls nach Stamm verschlüsselt, also teilten sich 409 Fibeln-Seiten 166
Bilder: verschiedene Werke hätten **dasselbe Bild** bekommen, stumm. Verraten hat
es nur ein Widerspruch im eigenen Bericht — „166 gelistet, 409 zugeordnet" kann
sich nicht ausgehen. Die Tabelle ist jetzt nach **Pfad** verschlüsselt.

**ALTO ist nicht PAGE mit anderem Namensraum.** Sein Text steht in
`String/@CONTENT`, unser Parser sucht `Unicode`. Stillschweigend durchgelassen käme
die Seite als leer durch, und das ist schlimmer als eine Ablehnung. ALTO wird
gezählt und verworfen, bis es einen Adapter hat.

## 4. Der dritte Zustand, viermal

„Konnte nicht" ist nicht „gibt es nicht"
([#165](https://github.com/thodel/training-atr-models/issues/165)). Der Importweg
verweigert darum an vier Stellen:

1. **Keine Lizenz oder keine Fundstelle** → die Quelle wird nicht einmal angelegt.
2. **Kein Bezugsweg für Bilder** → kein Upload ohne `--no-images`.
3. **Bilder geplant, keines angekommen** → kein Upload.
4. **Einzelne Seiten ohne Bild** → sie fallen heraus statt mit leerer Bildspalte
   mitzukommen; ebenso Seiten, deren Abruf scheiterte (`--keep-imageless` behält
   beides). Bei dach-gt sind das 64 ohne URL und 27 mit HTTP 404.

Und `--public` wird verweigert, solange die Bildrechte `ungeprüft` sind.

## 5. Was in der Datensatzkarte steht, und warum

Die Karte nennt **Lizenz und ihre Fundstelle**, den **Urheber namentlich**, den
**gelesenen Commit**, die Bildquelle und den Zustand der Bildrechte. Je Zeile
zeigen `source_path` und `source_url` auf die Ursprungsdatei, die URL an den
Commit genagelt — ein Link auf `main` wäre eine Behauptung über die Zukunft.

Der Grund für die Pflichtfelder: `Teklia/NewsEye-Austrian-line` gibt eine
CC-BY-4.0-Quelle auf HF als `license: mit` weiter. Eine Lizenz ohne Provenienz ist
genau die Angabe, die so entsteht.

**Die Karte behauptet nichts über die Sichtbarkeit.** Sie sagte einmal „bleibt
dieser Datensatz privat", und als die Sichtbarkeit sich ausserhalb des Repos
änderte, stand das als Unwahrheit in einem öffentlichen Dokument. Ein Dokument im
Repo kann einen Schalter daneben nicht kennen.

## 6. Bildwege verrotten

Zwei von sechs Quellen hatten beim ersten Lauf einen toten Bildweg, und das ist
die wichtigste Erkenntnis für die neunzehn Quellen, die noch kommen:

- **Weisthuemer** nagelt `ia903405.us.archive.org` fest. Dort liegt das Item nicht
  mehr — laut `archive.org/metadata` auf `ia600607` — und der alte Knoten antwortet
  überhaupt nicht: `http=000` nach 60 s. `stabilise_archive_org` schreibt auf
  `archive.org/download/…` um, das immer auf den haltenden Knoten umleitet.
- **dach-gt** trägt in 27 seiner Dokumente Bild-URLs, die HTTP 404 geben.

**Ein Bezugsweg ist so haltbar wie die URL darin, und niemand oben pflegt sie für
uns.** Das verschiebt die Abwägung in #193: „nur XML veröffentlichen und auf die
Bilder verlinken" sieht sauber aus, verlässt sich aber darauf, dass fremde
Bild-URLs halten. Bei zwei von sechs taten sie es schon nicht. Wer die Bilder nicht
hält, hält die Daten nicht.

## 7. Eine neue Quelle aufnehmen

1. Flach klonen, `tracked_pages` zählen lassen, und die Zahl **gegen das README der
   Quelle** stellen. Weichen sie ab, ist das ein Befund, nicht ein Fehler: bei
   `reichsanzeiger-gt` nennt die Quelle vier verschiedene Seitenzahlen, von denen
   keine die gemessene ist.
2. Lizenz **an der Fundstelle lesen** — `LICENSE`, `METADATA.yml`, `.zenodo.json`,
   `htr-united.yml`, `CITATION.cff` — und die Fundstelle mitschreiben. Widersprechen
   sich zwei Angaben, gilt die strengere, und es geht eine Mail raus (#191).
3. Den Urheber suchen; `.zenodo.json` nennt ihn oft mit ORCID.
4. Den Bildweg suchen, in dieser Reihenfolge: URL im Dokument, URL-Liste,
   Bezugsskript, Vorlage. **Ein fehlendes Skript ist kein fehlender Weg.**
5. Den Standard-Branch prüfen: drei der sieben CC0-Quellen liegen auf `master`.
6. Trockenlauf, und die Verwurfszahlen lesen, nicht nur die Seitenzahl.
