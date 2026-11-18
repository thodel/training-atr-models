# Mittelalterliches Seitenmodell — Quellen (Stand 17.11.2026)

Ziel: ein **Seiten**-VLM (ganzes Seitenbild → Transkription) für mittelalterliche
Handschriften und Urkunden, ausgehend von CATMuS Medieval und TRIDIS, ergänzt um
weiteres mittelalterliches Material; Leuven nur als Auszug.

Status: **Erhebung, nichts geladen, nichts trainiert.** Zahlen sind mit ihrer Herkunft
markiert: **(HF)** aus den Parquet-Footern gezählt, **(Karte)** aus Datensatzkarte/Zenodo/HTR-United,
**(Paper)** TRIDIS arXiv 2503.22714 Tab. I, **(gesch.)** geschätzt.

## 1. Warum nicht direkt die beiden HF-Datensätze

`CATMuS/medieval` (194.808 Zeilen, CC-BY-4.0) und `magistermilitum/Tridis` (197.4k Zeilen, MIT)
enthalten **nur Zeilenausschnitte**. CATMuS: keine Seitenkennung, keine Koordinaten, `im.path`
leer. TRIDIS: `Image_name` erlaubt teils eine Gruppierung nach Seite, aber ohne Koordinaten und
ohne verbürgte Zeilenfolge. Seiten gibt es nur aus den **Quellkorpora**. Der Weg dorthin ist
`scripts/foreign_gt_to_hf.py` ([FOREIGN_GT_IMPORT.md](FOREIGN_GT_IMPORT.md)).

Je-Handschrift-Zählung von CATMuS (346 Dateien, eine Handschrift und ein `project` je Datei):
[medieval_page_sources/catmus_medieval_files.tsv](medieval_page_sources/catmus_medieval_files.tsv)
(Pfad, Zeilen, Projekt, Signatur, Sprache, Jahrhundert, Gattung).

## 2. Transkriptionskonventionen → Prompt-Schalter

Entschieden: **ein Modell, Konvention im Prompt**, gesetzt je Dokument, nicht je Zeile.

| Schalter | Herkunft | Abkürzungen |
|---|---|---|
| `graphemic` | CATMuS-Quellen (nach ChocoMufin), Inzigkofen, TranscriboQuest | stehen als Zeichen |
| `expanded` (semi-diplomatisch) | TRIDIS-Quellen, Itinera-Nova `_full`, AMSMB, Klosterneuburg | aufgelöst, Allographe/Interpunktion normalisiert |
| `normalized` | DISTINGUO | normalisiert — eigener Schalter oder weglassen |

Achtung: Die CATMuS-**Quellrepos** enthalten oft die ursprüngliche (allographetische)
Transkription, nicht die CATMuS-normalisierte. Für `graphemic` muss die ChocoMufin-Abbildung
von CATMuS auf die Seitentexte angewandt werden, sonst trainiert der Schalter auf zwei Regeln.

## 3. Quellen

### A. Hinter CATMuS Medieval (graphemisch)

| Quelle (`project`) | Repo | Format | Seitenbilder | Lizenz | Zeilen (HF) / Hss. | Sprachen | Jh. |
|---|---|---|---|---|---|---|---|
| Herne (Kartause) | zenodo 10005366 | PAGE | **nicht im Record**; KBR/Gent/ÖNB holen | CC-BY-4.0 (Text) | 47.046 / 18 | Mittelniederländisch | 14 |
| Towards General Castilian HTR | zenodo 8406222 | ALTO | vermutlich im Record | **CC-BY-NC-SA-4.0** (v2) — prüfen | 29.003 / 29 | Kastilisch | 14–15 |
| CREMMA (+ LAT) | github HTR-United/cremma-medieval, CREMMA-Medieval-LAT | ALTO | im Repo | CC-BY-4.0 | 28.852 / 35 | fr, la | 12–16 |
| HTRomance (5 Repos) | github HTRomance-Project | ALTO | im Repo | CC-BY-4.0 | 26.584 / 86 | fr, la, it, ven, es, ca, nav | 9–16 |
| HTRogène (5 Repos) | github HTRogene | ALTO (gesch.) | im Repo | CC0 (fr, it); Rest **ungeklärt** | 16.874 / 53 | fr, la, it, es, ca, gl, oc | 7–16 |
| Gallicorpora | github Gallicorpora (MSS-15e, incunable, imprimé gothique) | ALTO | im Repo | CC-BY / CC0 | 15.801 / 27 | fr | 14–16 — **Drucke ausschliessen** |
| Eutyches | github malamatenia/Eutyches | ALTO | im Repo | CC-BY-4.0 (Repo: Apache-2.0) | 12.016 / 4 | la | 9–10 |
| PSL Chartes Students | github PSL-Chartes-HTR-Students | ALTO | im Repo | CC-BY-4.0 | 4.365 / 3 | fr, la | 15 |
| DEEDS | github DEEDS-Project/htr-dataset | ALTO/PAGE | im Repo | **ungeklärt** | 3.829 / 22 | la (Urkunden) | 11–16 |
| LIBER | github CIHAM-HTR/Liber | ALTO | im Repo | CC-BY-4.0 | 3.784 / 4 | fr, la | 14–15 |
| Fabliaux | github CIHAM-HTR/Fabliaux | ALTO | im Repo | CC-BY-4.0 | 2.010 / 5 | fr | 13–14 |
| Rescribe | github rescribe/carolineminuscule-groundtruth | ALTO | im Repo | CC-BY-4.0 | 447 / 17 | la | 8–12 |
| eDante, CATMuS in-house, DivaHisDB×CATMuS | **nicht gefunden** | | | | 4.197 / 12 | it, es, en, la | 10–15 |

Abkürzung möglich: `CATMuS/medieval-segmentation` (CC-BY) hat 1.705 Seitenbilder aus
207 Hss. mit Zeilenpolygonen, `shelfmark`, `project` — **ohne Text**. Ein Join mit den
CATMuS-Zeilen über Signatur und Zeile ist nicht geprüft.

### B. Hinter TRIDIS (semi-diplomatisch)

| Korpus | Repo | Format | Seitenbilder | Lizenz | Zeilen | Sprachen | Jh. | Seitenfähig? |
|---|---|---|---|---|---|---|---|---|
| HOME (Alcar u. a.) | zenodo 5600884 | PAGE/ALTO/TEI | Images.zip 11,4 GB + IIIF | CC-BY-4.0 | 95,6k (Paper) | la, afr | 12–14 | ja, **Seiten nur teilweise transkribiert** |
| HIMANIS | zenodo 5535306 | PAGE | IIIF / Zip | CC-BY-4.0 | 30.015 Z. / 1.217 S. (Karte) | la, fr | 14–15 | ja, **teilweise transkribiert** |
| e-NDP | zenodo 7575693 | PAGE | im Zip | CC-BY-4.0 | 34.231 Z. / 512 S. (Karte) | la, fr | 14–15 | ja |
| Königsfelden (ganz) | zenodo 5179361 | PAGE+TEI | 21 GB, gemeinfrei | CC-BY-4.0 | ~60k (Karte), TRIDIS nur 3k | la, de | 14–16 | ja — **liegt in dh-unibe** |
| Bullinger | github pstroe/bullinger-htr | Zeilen-PNG | keine PAGE | CC-BY-SA | 16,5k (Paper) | la, fnhd | 16 | nein — in-house `bullinger-autoren` stattdessen |
| VOC | zenodo 4159268 | PAGE | 18,5 GB | CC-BY-4.0 | ~1,4k (Stichprobe) | nl | **17** | ausserhalb Zeitraum → weglassen |
| CODEA, MLH, LAMOP | — | keine Koordinaten / nicht erreichbar | | | 5,5k / 11k / ? | | | nein |

### C. Weiteres mittelalterliches Material

**In-house (`dh-unibe`, PAGE, auf dem Research-Share gecacht):**
- `image-text_medieval-scripts_xiv-xv-xvi` — **Leuven / Itinera Nova**, siehe §4.
- `koenigsfelden-charters-*` (Teile 2, 3, post-1500, adhr-colmar — publizieren teils dieselben Projektordner, `corpus_plan` dedupliziert), `aaeb-xiv-xvii` (+ part-2), `rats-und-richtebuecher_xv-xvi`, `parzival-part-1`, `hgb-kf_mixture`, `bern-disputation-1528`, `weisthuemer`, `regesta-graphemic`/`-diplomatic` (zwei Konventionen derselben Seiten → **nur eine** oder als Schalter-Paar).
- Diese bilden schon das Korpus hinter `german-medieval-v1`; dessen Held-out (695 Seiten) bleibt reserviert.

**Fremd, offen, mit Seitenbildern** (Kandidaten für `foreign_gt_to_hf.py`):

| Name | Quelle | Format | Lizenz | Umfang | Sprache | Jh. | Typ | Konvention |
|---|---|---|---|---|---|---|---|---|
| Inzigkofen | zenodo 17978574 | PAGE | CC-BY-4.0 | 337 S. / 9.726 Z. (Karte) | gmh/fnhd | 15 | Buch | graphemisch |
| DISTINGUO | nakala 10.34847/nkl.48ad8b8d | PAGE | CC-BY-4.0 | 318 S. / 15.190 Z. (Karte) | la | 13–15 | Buch | normalisiert |
| Hands of Antwerp deeds | github mikekestemont/deeds-of-antwerp | TEI mit Zeilenzonen | MIT / CC-BY | 470 S. (Karte) | mnl | 14 | Urkunden | hyperdiplomatisch, „silver“ |
| Flos unde Blankeflos | zenodo 14869908 | PAGE+ALTO | CC-BY-4.0 | ~100–200 S. (gesch.) | mnd | 14–15 | Buch | diplomatisch |
| Nürnberger Briefbücher 2–5 | zenodo 13881575 | Zeilen + Seitenbilder | CC-BY-4.0 | 1.711 S. (Karte) | fnhd | 15 | Register | mehrere — Koordinaten prüfen |
| AMSMB Notariatsurkunden | dataverse.bsc.es BSC/0VB0MC | PAGE | CC-BY-SA-4.0 | 100 Dok. / 3.369 Z. | la, ca | 13–15 | Urkunden | expanded |
| iForal | github Arch-W/iForal-Dataset | PAGE | CC-BY-4.0 | 180 S. / 8.009 Z. | la, pt | 13–15 | Urkunden | ? |
| Klosterneuburg Cod. 48 | zenodo 7466928 | PAGE | CC-BY-4.0 | 68 S. / 4.605 Z. | fnhd | 15 | Buch | diplomatisch; Bilder separat |
| Wiener Winter School (3 Sätze), ÖNB Cod. 940/2160 | zenodo 10589479, 14524944, 17911438, 7467027 | PAGE/ALTO | CC-BY-4.0 | ~300+ S. (gesch.), 7.835 Z. | la (+cs) | 9–15 | Buch | gemischt — CATMuS-Überlappung prüfen |
| Paris Bible, DReMAR, TranscriboQuest 24/25 | github parisbible; zenodo 13757090, 13757440, 17062963 | ALTO | CC-BY (17062009 ist **NC-SA** → nein) | klein | la, fro, dum, gmh | 9–15 | Buch/Akten | CATMuS |

**Lücken:** Mittelenglisch — keine offene Seiten-GT gefunden. Fribourg-Sätze (St. Gallen,
DIVA) nur zu akademischen Bedingungen.

## 4. Leuven (Itinera Nova) als Auszug

Im Hub-Repo `dh-unibe/image-text_medieval-scripts_xiv-xv-xvi`, Revision `729e9b2721ba`
(am 17.11.2026 gezählt): **694 Projektordner**, davon **663 `SAL…`-Register** (Leuvener
Schöffenregister, mnl + la, 1350–1800), dazu 72 flache Shards `data/train-000NN-of-00072`
(alte Ablage, ~15k Seiten laut Agent-Zählung — **nicht** mitlesen, sonst doppelt).
Lokal liegt alles im Cache `hf_hub/hub/` des Shares (992 GB, RESEARCH_STORAGE.md §11).

Präzedenz: #204 las 576 Register mit Deckel 25 Seiten je Register (≤ 14.400 Seiten).
Für ein gemischtes Seitenmodell zu viel — Leuven würde jede andere Quelle überwiegen.

### Was die Zählung ergab (17.11.2026)

- **Kein Statusfilter möglich:** jede Seite trägt den Transkribus-Status `IN_PROGRESS`.
- **Nackte `SAL…`-Ordner sind kaum transkribiert** (~20 % der Zeilen mit Text), `_full`-Ordner
  zu ~90 %. Nur `_full` kommt in Frage.
- **`_full` und `_full_without_abbr` sind verschiedene Dokumente**, keine Dubletten derselben
  Seiten.
- **Konvention:** der Text behält Kürzungen (`voirs`, `pncia`, `eˀ ibˀ`) — näher an graphemic/
  diplomatisch als an expanded.
- **Datierung:** mittelalterliche Register schreiben Daten römisch oder in Worten; Register
  des 17./18. Jh. haben arabische Jahreszahlen (Median ~1700–1770) und oft verderbten Text.
  Regel je Register in `medieval_page_sources/leuven_registers.tsv` (Spalte `period`): nicht
  `SAL…_full` → other; Anteil Zeilen mit Text < 0,7 → thin; keine arabische Jahreszahl oder
  grösste < 1600 → **medieval**; sonst late. Ergebnis: **318 medieval** (266.769 Seiten,
  Serien SAL73/74/77/78/81/82), 230 late, 13 thin, 108 other.

### Der Auszug (v1)

- **Held-out:** 12 ganze Register (`random.Random(42).sample` aus den 318), in
  `medieval_page_sources/leuven_heldout_registers.txt`.
- **Training:** die übrigen 306 Register, je **9 Seiten** (`max_pages: 9`), gezogen mit festem
  Seed je Register aus der `filename`-Spalte, plus 3 Ersatzseiten für Seiten ohne Zeile;
  **≤ 2.754 Seiten**. Die Seiten sind im neuen Feld `DatasetSpec.pages` festgehalten
  (`max_pages` allein nähme die ersten Seiten jedes Registers, keine Stichprobe).
- **Erzeugt mit** `scripts/draw_leuven_excerpt.py` (liest den lokalen Snapshot) direkt in
  `ubelix/specs/medieval-page-qwen35-leuven-v1.json`: die vier Datensätze von
  `medieval-german-page-qwen35-v1.json` plus 306 Leuven-Einträge. Stichprobe geprüft: 3 Register
  × 12 gepinnte Seiten gefunden, alle mit Text.
- **Rechte:** Karte sagt MIT, eine Leuvener Freigabe ist nicht dokumentiert — klären vor
  jeder Veröffentlichung des Modells.

## 5. Was bei Seitenzielen zusätzlich bricht

1. **Teiltranskribierte Seiten** (HOME, HIMANIS: aus Editionen ausgerichtet, nicht edierte
   Akte und Rubriken fehlen). Ein Seitenmodell lernt daraus, Text auszulassen. Entweder
   nicht transkribierte Regionen im Bild maskieren oder solche Seiten verwerfen;
   Kriterium: Anteil der Textfläche ohne Zeile (mit Segmentierungsmodell messen).
2. **Dubletten über Quellen:** e-NDP (TRIDIS) ↔ HTRogène (CATMuS) teilen AN LL 106B, 108A,
   108B, 110, 125 (1.837 Zeilen) — je Seite eine Konvention, oder beide als Schalter-Paar,
   aber nie dieselbe Seite links im Training und rechts im Test. Königsfelden (TRIDIS-Slice
   ↔ dh-unibe), Regesta (graphemic ↔ diplomatic), Itinera Nova (`_full` ↔ `_without_abbr` sind verschiedene Dokumente, siehe §4).
3. **Held-out je Dokument/Handschrift**, nicht je Seite: CATMuS-`gen_split` gilt pro Zeile
   und ist für Seiten unbrauchbar; Held-out-Handschriften neu ziehen und in
   `config/heldout_eval_documents.json` eintragen.
4. **Drucke** (Gallicorpora incunable/imprimé, reichenau-inkunabeln) nur, wenn gewollt.

## 6. Grössenordnung (nur seitenfähige Quellen, gesch.)

| Block | Zeilen | Konvention |
|---|---:|---|
| A ohne Herne, ohne Unauffindbares, ohne Drucke | ~130k | graphemic |
| Herne, falls Bilder beschaffbar | +47k | graphemic |
| B: HOME, HIMANIS, e-NDP, Königsfelden | ~200k (vor Maskierung/Verwurf) | expanded |
| C fremd | ~60k | gemischt |
| Leuven-Auszug (≤ 2.754 S. × ~30 Z., gesch.) | ~80k | graphemic/diplomatisch |
| dh-unibe deutsch, mittelalterlich (bestehend) | ~220k (BASE_MODEL_LADDER §1f) | graphemic/diplomatisch |

## 7. Nächste Schritte

1. Register in `foreign_gt.py` erweitern um Block A (ohne Herne), e-NDP, HOME, HIMANIS, Block C;
   Lizenzlücken (HTRogène la/es/oc, DEEDS, Castilian NC-SA) vorher klären.
2. ChocoMufin-Abbildung von CATMuS für die Seitentexte von Block A übernehmen.
3. ~~Leuven-Auszug~~ erledigt (§4); nächster Schritt dort: Smoke-Lauf der Spec.
4. Teiltranskription messen (HOME, HIMANIS) und Regel festlegen.
5. Held-out-Handschriften ziehen, dann Smoke-Lauf (Qwen3.5-4B, Seitenmodus), dann Volllauf.
