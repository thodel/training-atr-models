# research-storage: was darauf liegt, und was woanders hingehört

Der **research-storage** ist `/storage/research/wbkolleg_dh_1` auf UBELIX — der
Gruppenbereich des Walter Benjamin Kollegs DH, auf dem GPFS-Dateisystem
`rs_gpfs`. Auf tei ist derselbe Bereich als `/mnt/wbkolleg_dh_1` eingebunden.
Dort liegt auch `HF_HOME`, also der HuggingFace-Zwischenspeicher aller
Trainingsläufe:

```
HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
```

Die tägliche Prüfung läuft über das Werkzeug **`storage`** des lesenden MCP
`atr-results` — typisiert, lesend, und ohne Shell-Freigabe. Von Hand ginge auch:

```bash
python3 ~/ubelix/research_storage.py --check      # sofort, ein statfs
python3 ~/ubelix/research_storage.py --suggest    # + Vorschläge
sbatch ubelix/storage_inventory.sbatch            # das teure du, wöchentlich
```

Der MCP-Probe ruft dasselbe Skript mit `--json` auf, statt dessen Regeln
nachzubauen: welche Verzeichnisse wiederherstellbar sind und was nur die einzige
Kopie ist, ändert sich — am 09.10.2026 zweimal an einem Tag, weil Messungen es
widerlegten. Zwei Fassungen derselben Regeln wären eine Gabelung.

Seit dem 09.10.2026 prüft eine Routine täglich um 07:05 CEST und nennt bei unter
10 % frei, was sich woanders besser aufhebt.

## 0. Was am 09.10.2026 gelöscht wurde

Die **alte HuggingFace-Cache-Form** unter `$HF_HOME` selbst — 59 Einträge,
**1,83 TB**:

```
vorher   12T belegt, 642 GiB frei, 95 %
nachher  9,8T belegt, 2,3 TB frei, 81 %
```

**1,66 TB zurückgewonnen**, der freie Anteil von 5,2 % auf 19,1 %. Protokoll mit
jedem Eintrag und seiner Grösse unter
`~/ubelix/logs/purge-old-cache-<zeitstempel>.log`.

Vier Prüfungen vorher, und jede hat etwas verändert:

1. **Liest die laufende Software diese Pfade?** Nein. `HF_HUB_CACHE` zeigt unter
   `huggingface_hub` 1.31.0 auf `$HF_HOME/hub`; die alte Form liegt unter
   `$HF_HOME` selbst und wird nicht konsultiert. Ein `prepare` lief währenddessen.
2. **Liegt jeder Eintrag auf dem Hub?** 56 von 59. Die drei übrigen —
   `towerbooks-line-test`, `-test-with-inference`, `towerbooks-rawxml-test` —
   geben 404, sind aber **3,5-KB-Negativmarker mit zwei Dateien**, ohne Inhalt.
3. **Ist die alte Form wirklich doppelt?** Nur zu einem Teil. Von den 59 hatten
   **22 eine substanzielle Kopie** in `hf_hub/hub/` (0,35 TB), bei **37 war der
   Eintrag dort ein Stummel** (1,47 TB). Meine Behauptung „36 von 37 doppelt"
   stammte aus Namensvergleich und war falsch.
4. **Und der grösste Posten?** 1,39 der 1,83 TB waren **ein** Eintrag:
   `image-text_medieval-scripts_xiv-xv-xvi` mit 1.386,5 GB. Dateien gezählt:

   | Ort | echte Dateien | parquet | blobs | Grösse |
   |---|---:|---:|---:|---:|
   | `hf_hub/<alt>` | 926 | 691 | 230 | 1.386,5 GB |
   | `hub/` | 699 | **694** | 0 | 992 GB |
   | `hf_hub/hub/` | 5 | 0 | 1 | 21 KB |

   Die alte Form **dupliziert intern** (Snapshot-Dateien *und* Blobs, 1,26× der
   1.098 GB auf HF), während `hub/` mit 694 parquet den vollständigeren Satz bei
   gleicher Revision hält. Darum war die Löschung begründet — und zwar
   umgekehrt zu meiner ersten Vermutung.

## 1. Der Stand, gemessen am 09.10.2026

```
Filesystem      Size  Used Avail Use% Mounted on
rs_gpfs          12T   12T  642G  95% /storage
```

**642 GiB von 12 TiB frei — 5,2 %.** Die Schwelle von 10 % war also von Anfang
an unterschritten.

Für `wbkolleg_dh_1` gibt es **kein Quoten-Fileset** (`mmlsquota: no such fileset
in quota enabled file systems`), weshalb `df` die Zahl des ganzen Dateisystems
nennt. Nach Auskunft der Gruppe ist dieser freie Platz **unserer**; `df` kann
das nicht belegen, und ein `du` über den Bereich lief nach zwei Minuten noch.

## 2. Wo der Platz liegt

39 Verzeichnisse auf der obersten Ebene, das älteste von 2021. Eines davon ist
unseres, und es ist etwa die Hälfte des Bereichs:

| | |
|---|---:|
| `Textrecognition_Training/` | **≈ 5,7 TB** |
| 38 weitere (`Projekt_*`, `Backup_*`, `Lehre`, `Omeka` …) | Rest |

Innerhalb unseres Bereichs:

| Verzeichnis | Grösse | Was es ist |
|---|---:|---|
| `hf_hub/` | **2,9 TB** | HF-Zwischenspeicher, aktuelles `HF_HOME` |
| `training_folder/` | **1,8 TB** | davon `jobs/` **1,8 TB**; `trained/` 6,0 GB, `bases/` 47 MB, `hf-datasets/` leer |
| `hub/` | **992 GB** | ein zweiter HF-Zwischenspeicher — **ein** Datensatz |
| `archive/` | 11 GB | |
| `eval_sets/` | 5,6 GB | Auswertungsziehungen |
| `Conf_Mats/` | 4,3 GB | |
| `trained-ubelix/` | 3,8 GB | trainierte Gewichte |
| `datasets/` | 2,8 GB | |
| `kraken_ocr/`, `TrOCR/`, `xet/`, `registry/`, `tmp/` | < 1 GB | |

## 3. Der Befund: vier Zwischenspeicher nebeneinander

`hf_hub/` ist nicht ein Cache, sondern mehrere — in zwei Formaten, plus zwei
weitere Kopien daneben:

| Ort | Grösse | Einträge |
|---|---:|---|
| `hf_hub/hub/` | 884 GB | 54 Datensätze, 22 Modelle — die **aktuelle** Form |
| `hf_hub/datasets--*` | **1,6 TB** | 37 Datensätze in der **alten** Form |
| `hf_hub/models--*` | 115 GB | 22 Modelle in der alten Form |
| `hub/` | **992 GB** | **ein** Datensatz, Geschwister von `hf_hub` |

Die Teile von `hf_hub` summieren sich auf 2,6 TB gegen die gemessenen 2,9 TB —
die Differenz steckt in `blobs/` und `.locks/`, die hier nicht einzeln
aufgeschlüsselt sind.

**Die Doppelung ist gemessen, nicht vermutet:** von den 37 Datensätzen in der
alten Form liegen **36 auch in `hf_hub/hub/`**, also 97 %. Nur einer ist
alt-exklusiv, 18 sind neu-exklusiv.

**Für `hub/` gilt das Gegenteil, und das ist ein Fehler, den diese Datei
zuerst enthielt.** Ich hatte die 992 GB als Dublette geführt, weil derselbe
Datensatzname auch in `hf_hub/hub/` steht. Gemessen:

| | |
|---|---:|
| `hub/…medieval-scripts_xiv-xv-xvi` | **992 GB** |
| `hf_hub/hub/…medieval-scripts_xiv-xv-xvi` | **21 KB** |

Beide notieren dieselbe Revision `729e9b2721ba`, aber der Eintrag im aktuellen
Cache ist ein Stummel aus Metadaten mit einem einzigen Blob. **`hub/` ist die
einzige lokale Kopie**, nicht die zweite. Übereinstimmende Namen sind kein
Beweis für Doppelung — Grössen sind einer.

**Jeder Byte davon ist aus dem Hub wiederherstellbar.** Das ist nicht eine
Vermutung über die Daten, sondern die Eigenschaft eines Zwischenspeichers: 54
Datensätze und 22 Modelle, alle unter `dh-unibe/` oder bei ihren Urhebern
(`Qwen/`, `allenai/`, `google/`, `microsoft/`, `timm/`). `dh-unibe` hält auf HF
32 Modelle und 46 Datensätze.

Es ist dasselbe Muster, aus dem schon einmal **621 GB** zurückgewonnen wurden.

## 4. Was sich woanders besser aufhebt

Nach Zuverlässigkeit geordnet — die erste Gruppe braucht keine Entscheidung
über Daten, nur eine über Bequemlichkeit:

**1. Wiederherstellbar *und* lokal doppelt — 1,7 TB, der stärkste Fall**

`hf_hub/datasets--*` und `hf_hub/models--*`, die alte Cache-Form, zu 97 % belegt
doppelt. Hier wird nichts aufgegeben, nur eine zweite Kopie derselben Sache. Der
Preis ist Zeit beim nächsten Lauf, nicht Datenverlust — mit der Einschränkung,
dass ein GPU-Knoten Basisgewichte **nicht** selbst holen kann (Job 16191716 starb
daran), also muss `prefetch_bases.sh` vorher als CPU-Job laufen.

**2. Abgeleitet, Wiederherstellung teuer — 1,8 TB, je Job zu entscheiden**

`training_folder/jobs` hält **53** Arbeitsverzeichnisse vom 07.08. bis 07.10.2026:
zugeschnittene Zeilenbilder, Manifeste, Logs. Aus den HF-Datensätzen
wiederherstellbar, aber nur über die prepare-Stufe, und die braucht Stunden
(gemessen: 2 h 35 für elf Datensätze).

**Nur 12 der 53 stehen auf `completed`**, 41 auf etwas anderes oder sind nicht
lesbar — eine pauschale Regel „abgeschlossene weg" greift hier also nicht. Und
anders als auf `/scratch/.../runs/jobs` läuft hier **keine 30-Tage-Regel**: das
Verzeichnis wird nie von selbst leer.

**3. Einzige lokale Kopie, aber auf HuggingFace — 992 GB, kein freier Gewinn**

`hub/` hält `image-text_medieval-scripts_xiv-xv-xvi` als einzige lokale Kopie. Der
Datensatz liegt öffentlich auf HF mit genau dieser Revision (1.098 GB), ist also
wiederherstellbar — aber die Wiederherstellung ist ein Terabyte-Download. Das ist
eine andere Entscheidung als Punkt 1 und gehört nicht in denselben Satz.

**4. Gehört auf HuggingFace, je Modell zu prüfen — 10 GB**

`training_folder/trained` (6,0 GB) und `trained-ubelix` (3,8 GB). Wo das Modell
unter `dh-unibe/` auf HF liegt, ist die lokale Kopie entbehrlich; von den 32
dh-unibe-Modellen auf HF sind nicht alle aus diesen Verzeichnissen, und nicht
alles hier ist dort.

**Was sich nicht lohnt:** `training_folder/bases` sind 47 MB (vier kraken-Modelle
aus Zenodo-Hinterlegungen, nicht die HF-Basismodelle), `hf-datasets` ist leer.
Beides steht hier, damit niemand sie für die grossen Posten hält — ich hatte sie
zuerst dafür gehalten.

**Was hier nicht beurteilt wird**

Die 38 anderen Verzeichnisse der obersten Ebene gehören anderen Projekten und
Personen. `research_storage.py` macht **ausschliesslich innerhalb von
`Textrecognition_Training` Vorschläge** und berichtet den Rest mit Grösse und
ohne Empfehlung. Das ist Absicht: nur dort ist bekannt, was wiederherstellbar
ist, und ein Vorschlag, `Projekt_Bullinger` zu löschen, wäre keine Empfehlung,
sondern ein Risiko.

## 5. Warum die Prüfung zweigeteilt ist

`--check` ist ein `statfs` und dauert Millisekunden. Die Aufschlüsselung je
Verzeichnis ist ein `du` über rund 11 TB auf GPFS und dauert Stunden. Also:

- **täglich** `--check` — eine Zahl, eine Schwelle, und bei Unterschreitung die
  Vorschläge aus dem letzten Inventar
- **wöchentlich** `ubelix/storage_inventory.sbatch` — das `du`, als Slurm-Job
  mit zwei Kernen

Fehlt das Inventar oder ist es älter als 14 Tage, **sagt die Prüfung das** statt
Vorschläge aus veralteten Zahlen zu machen. Kein Inventar und ein altes Inventar
sind zwei Zustände, nicht einer (#165).

## 6. Scratch ist etwas anderes

`/scratch/network/users/$USER` ist **nicht** der research-storage. Dort liegen
die materialisierten Korpora der Läufe, und sie werden nach **30 Tagen**
gelöscht — `atr-results deadlines` nennt je Job das Datum. Platzdruck auf dem
research-storage lässt sich nicht durch Scratch lösen und umgekehrt.
