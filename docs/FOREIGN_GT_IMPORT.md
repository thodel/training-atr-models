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
| `image-text_dach-gt` | 71 | ~2.000 | 71 | 48 | URL im XML + Vorlage — vor #197; mit ALTO siehe §9 |
| `image-text_weisthuemer` | 25 | ~1.400 | 25 | 13 | Shell-Skript (archive.org) |
| **zusammen** | **990** | **~153.000** | 990 | **1.465** | |
| `image-text_charlottenburger-amtsschrifttum` *(eingetragen 02.11.2026, noch nicht hochgeladen)* | 27 | 3.576 | 27 erreichbar | — | URL-Liste (ZLB Berlin, IIIF `default.jpg`) |

Die sechs hochgeladenen sind **privat**. Die Transkriptionen sind CC0; die **Bildrechte sind
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
35 Seiten (Bd. 6 und 7) in zwei Schleifen. Sie kommen seit #188 über je eine
`ImageTemplate` daneben, die nichts rät: der Schleifenkörper nennt das Archivmitglied
genau so wie die Datei, die er schreibt, und das PAGE nennt diese Datei als
`imageFilename` — die Vorlage ist `…/{base}_tif%2F{img}`. Alle zehn Bilder haben die
Grösse, die ihr PAGE angibt (gemessen 02.11.2026).

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

**ALTO** liest seit #197 ein Adapter, wo die Quelle es verlangt (§9). Die **Zenodo-ZIP-Form** hat einen eigenen Abschnitt (§8).

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
gezählt und verworfen, ausser unter den Pfaden, die `Source.alto` nennt (§9).

## 4. Der dritte Zustand, viermal

„Konnte nicht" ist nicht „gibt es nicht"
([#165](https://github.com/thodel/training-atr-models/issues/165)). Der Importweg
verweigert darum an fünf Stellen:

1. **Keine Lizenz oder keine Fundstelle** → die Quelle wird nicht einmal angelegt.
2. **Kein Bezugsweg für Bilder** → kein Upload ohne `--no-images`.
3. **Bilder geplant, keines angekommen** → kein Upload.
4. **Einzelne Seiten ohne Bild** → sie fallen heraus statt mit leerer Bildspalte
   mitzukommen; ebenso Seiten, deren Abruf scheiterte (`--keep-imageless` behält
   beides). Bei dach-gt sind das 64 ohne URL und 27 mit HTTP 404.
5. **Ein Bild, das nicht die Grösse hat, die das PAGE nennt** → es zählt wie ein
   gescheiterter Abruf und fällt heraus, gemeldet als `falsches Bild — Bild W×H,
   das PAGE sagt W×H` (#188). Der Trainingsweg schneidet die Zeilen mit den
   Koordinaten des Dokuments aus; die falsche Seite oder die richtige in anderem
   Massstab ergäbe falsche Ausschnitte, ohne dass etwas fehlschlägt. Eine
   Fehlerseite mit HTTP 200 fällt hier ebenso heraus (`kein lesbares Bild`).
   Stichprobe am 02.11.2026 (je vier Seiten): reichsanzeiger-gt, DTGT, gt-fraktur,
   Fibeln, Weisthuemer und charlottenburger stimmen aufs Pixel; dach-gt war von
   hier nicht zu erreichen und wird beim nächsten Import auf dem Host geprüft.

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

**Der Prüflauf (#199).** `scripts/probe_image_paths.py` ruft je Quelle **fünf**
Bild-URLs ab, gleichmässig über die Seiten gezogen, erste und letzte dabei — fünf
hätten Weisthuemer und dach-gt beide gefunden. Er läuft wöchentlich
(`deploy/systemd/atr-image-probe.timer`, Montag 06:20, installiert mit
`bash scripts/install_image_probe_timer.sh`) und hängt bewusst nicht am Import:
der läuft, wenn jemand eine Quelle einliest, und dann ist es zu spät.

Zustand in `~/atr-cache/image-probe/`: `snapshot.json`, `reports/YYYY-MM-DD.md`
**nur bei Veränderung** gegen den letzten Lauf, `runs.jsonl` für jeden Lauf. Der
erste Lauf meldet die Quellen, die schon nicht vollständig erreichbar sind.

Der Bericht unterscheidet vier Lagen, weil jede eine andere Antwort hat:

| Lage | heisst | Antwort |
|---|---|---|
| `unreachable` | der Knoten antwortet nicht (`http=000`) | Host gewechselt? Eine Umschreibung ins Register |
| `404` | die Datei ist weg | Seiten fallen beim Import heraus; die Quelle fragen |
| `403` | der Zugang hat sich geändert | die Bedingungen der Einrichtung neu lesen (#193) |
| `timeout` | der Server ist langsam | beim nächsten Lauf wieder ansehen |

Dazu `listing_failed`, wenn sich eine Quelle gar nicht auflisten lässt. Bei
Zenodo-Hinterlegungen werden die Datei-URLs der Archive geprüft.

**Umschreibungen sind Register-Einträge.** `foreign_gt.REWRITES` hält je bekannter
haltbarerer Form einen `UrlRewrite` (Muster, Vorlage, Begründung), und
`image_urls` schickt **jede** URL hindurch, gleich aus welchem Bildweg sie kommt.
Die nächste Quelle mit einem festgenagelten Knoten braucht eine Zeile, keine
Funktion.

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

## 8. Zenodo-Hinterlegungen (#198)

Vier Quellen liegen als Zenodo-ZIP. Sie gehen denselben Weg hinein, ohne Klon:

```bash
.venvs/kraken-train/bin/python scripts/foreign_gt_to_hf.py reichenau-inkunabeln --dry-run
.venvs/kraken-train/bin/python scripts/foreign_gt_to_hf.py reichenau-inkunabeln --part lateinisch
.venvs/kraken-train/bin/python scripts/foreign_gt_to_hf.py fanny-mendelssohn --no-images
```

Gemessen am 01.11.2026, je Teil ein Trockenlauf gegen die echten Archive:

| Quelle | DOI (Version) | Lizenz | Seiten | Zeilen | Teilung der Quelle |
|---|---|---|---:|---:|---|
| `reichenau-inkunabeln` Deutsch | 10.5281/zenodo.11046062 | CC BY-SA 4.0 | 205 | 9.671 | 185 / 20 |
| `reichenau-inkunabeln` Lat.-Dt. | 〃 | 〃 | 393 | 18.610 | 354 / 39 |
| `reichenau-inkunabeln` Latein *(nicht Vorgabe)* | 〃 | 〃 | 1.609 | 97.049 | 1.449 / 160 |
| `newseye-onb` | 10.5281/zenodo.3387369 | CC BY 4.0 | 161 | 58.796 | 148 / 13 |
| `fanny-mendelssohn` *(keine Bilder)* | 10.5281/zenodo.15223355 | CC BY 4.0 | 200 | 2.238 | — |
| `regesta-graphemic` | 10.5281/zenodo.18441031 | CC BY 4.0 | 50 | 2.177 | — |
| `regesta-diplomatic` *(derselbe Seitensatz)* | 10.5281/zenodo.18377766 | CC BY 4.0 | 50 | 2.177 | — |

**Was anders ist als beim Klonen:**

- **Die versionierte DOI ist Pflicht** und steht in der Karte und in `source_url`.
  Der Konzept-DOI zeigt auf „die neueste" — wie ein Link auf `main` — und wird
  verweigert. `source_path` ist `<Datei>/<Eintrag>` im Archiv.
- **Die Lizenz wird an zwei Stellen gelesen:** `metadata.license.id` der API und
  jede `LICENSE*`-Datei, im Datensatz oder im Archiv. Widerspruch, oder eine Datei,
  deren Lizenz nicht erkannt wird, ist Ablehnung mit beiden Fundstellen. Weicht
  die API vom Register ab, ebenso: die Hinterlegung hat sich geändert.
- **Die Vorteilung der Quelle wird übernommen**, nicht neu gezogen: `train` nach
  `data/train/`, `validation` nach `data/validation/`, und jede Zeile trägt sie in
  `source_split`. `prepare` liest nur die ausdrücklich genannten `data_files`, die
  Validierungsseiten der Quelle fliessen also nicht ins Training.
- **Je Teilsammlung ziehbar** (`--part`, `--all-parts`). Reichenaus lateinischer
  Teil — 1.609 von 2.207 Seiten, 1,45 GB — ist nicht Vorgabe: für ein deutsches
  Modell Typen-Robustheit, nicht Sprache.
- **`__MACOSX`-Schatten und Einträge mit `..`** werden verworfen und gezählt. Aus
  dem Archiv wird gelesen, nie ausgepackt.
- **Das Bild zur Seite** wird über `imageFilename` gesucht: daneben, eine Ebene
  höher (Transkribus' `page/`), oder in einem eigenen Bild-Archiv (Regesta). Zwei
  Kandidaten, die keine dieser Regeln entscheidet, sind „mehrdeutig" und werden
  gezählt, nicht geraten.
- **Archive werden gegen Zenodos MD5 geprüft** und bleiben unter
  `<workdir>/_zenodo/<id>/` liegen.

**Ein Zwilling ist keine zweite Quelle.** `regesta-graphemic` (18441031) und
`regesta-diplomatic` (18377766) sind dieselben 50 Bilder auf zwei
Transkriptionsebenen — ihr `images.zip` hat dieselbe MD5. Beide stehen im
Register mit `same_pages_as` aufeinander; die Karte sagt es, und ein Verweis der
Hinterlegung auf eine Version oder Variante, die das Register nicht so führt,
wird beim Einlesen gemeldet.

## 9. ALTO (#197)

`atr_training.alto.alto_to_pagexml` wandelt ALTO in PageXML um, denn das Schema
unserer Datensätze ist PageXML und `prepare` schneidet aus `TextLine`-Koordinaten.

- **Wortfugen aus `SP`**, nicht aus Koordinaten. Zwei `String` ohne `SP` sind ein
  Wort, zwei `SP` sind zwei Leerzeichen, `HYP` bleibt stehen. eScriptorium (dach-gt)
  schreibt eine Zeile als ein `String`, mit den Leerzeichen im `CONTENT`.
- **Polygon und Grundlinie** kommen mit; nur wo eine Zeile keines hat, ihr Rechteck.
- **Leere Zeilen** werden gezählt und verworfen, eine Seite ganz ohne Text auch.

**Gegen PAGE gemessen** an gt-fraktur, wo dieselben 208 Seiten in beiden Formen
liegen: 14.361 von 14.615 Zeilen sind wörtlich gleich. Die übrigen 254 unterscheiden
sich im **Inhalt** der beiden Fassungen der Quelle (die ALTO-Fassung ist weniger
korrigiert: „Nationalism.“ statt „Rationalism.“), keine nur in den Leerzeichen.

**Nur wo die Quelle es verlangt** (`Source.alto`, Pfadfragmente), weil ALTO in
einer Quelle nicht eines ist:

| dach-gt | ALTO-Dateien | gelesen? |
|---|---:|---|
| `DE-4/…/alto` | 672 | ja — Ground Truth; 15 davon auch als `GT-PAGE`, die gewinnt |
| `DE-12/…/alto/ocr` | 173 | **nein — OCR, keine Ground Truth** |
| `DE-12/…/alto/gt` | 19 | nein — dieselben Seiten wie `GT-PAGE`, unter anderen Namen |

Gemessen am 01.11.2026: dach-gt **810 Seiten, 39.786 Zeilen** statt 162 und 4.636,
davon 648 Seiten und 35.150 Zeilen aus ALTO; 761 der 810 Seiten mit Bild-URL. Die
Bilder von DE-4 stehen nicht im Dokument (es nennt die Landingpage), sondern in
der METS der UB Marburg (`jpg/def`, gleiche Grösse wie im ALTO). gt-fraktur
verlangt den Adapter nicht und bleibt bei 207 Seiten.
