# research-storage: was darauf liegt, und was woanders hingehört

Der **research-storage** ist `/storage/research/wbkolleg_dh_1` auf UBELIX — der
Gruppenbereich des Walter Benjamin Kollegs DH, auf dem GPFS-Dateisystem
`rs_gpfs`. Auf tei ist derselbe Bereich als `/mnt/wbkolleg_dh_1` eingebunden.
Dort liegt auch `HF_HOME`, also der HuggingFace-Zwischenspeicher aller
Trainingsläufe:

```
HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
```

```bash
python3 ~/ubelix/research_storage.py --check      # sofort, ein statfs
python3 ~/ubelix/research_storage.py --suggest    # + Vorschläge
sbatch ubelix/storage_inventory.sbatch            # das teure du, wöchentlich
```

Seit dem 09.10.2026 prüft eine tägliche Routine um 07:05 CEST die Belegung und
nennt bei unter 10 % frei, was sich woanders besser aufhebt.

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
| `training_folder/` | **1,8 TB** | `bases`, `hf-datasets`, `jobs`, `tmp`, `trained` |
| `hub/` | **992 GB** | ein **zweiter** HF-Zwischenspeicher |
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
| `hub/` | **992 GB** | zwei Einträge, Geschwister von `hf_hub` |
| `training_folder/bases` + `hf-datasets` | Teil der 1,8 TB | noch eine Kopie |

Die Teile von `hf_hub` summieren sich auf 2,6 TB gegen die gemessenen 2,9 TB —
die Differenz steckt in `blobs/` und `.locks/`, die hier nicht einzeln
aufgeschlüsselt sind.

**Die Doppelung ist gemessen, nicht vermutet:** von den 37 Datensätzen in der
alten Form liegen **36 auch in `hf_hub/hub/`**, also 97 %. Nur einer ist
alt-exklusiv, 18 sind neu-exklusiv. Und die beiden Einträge in `hub/` liegen
**beide** ebenfalls in `hf_hub/hub/`.

**Jeder Byte davon ist aus dem Hub wiederherstellbar.** Das ist nicht eine
Vermutung über die Daten, sondern die Eigenschaft eines Zwischenspeichers: 54
Datensätze und 22 Modelle, alle unter `dh-unibe/` oder bei ihren Urhebern
(`Qwen/`, `allenai/`, `google/`, `microsoft/`, `timm/`). `dh-unibe` hält auf HF
32 Modelle und 46 Datensätze.

Es ist dasselbe Muster, aus dem schon einmal **621 GB** zurückgewonnen wurden.

## 4. Was sich woanders besser aufhebt

Nach Zuverlässigkeit geordnet — die erste Gruppe braucht keine Entscheidung
über Daten, nur eine über Bequemlichkeit:

**Wiederherstellbar aus dem Hub** (≈ 3,9 TB)

1. `hf_hub/datasets--*` und `hf_hub/models--*` — **1,7 TB**, zu 97 % belegt
   doppelt. Der stärkste Fall: hier wird nichts aufgegeben, nur eine zweite
   Kopie derselben Sache.
2. `hub/` — **992 GB**, ein Cache, der entstand, als `HF_HOME` eine Ebene höher
   zeigte. Beide Einträge liegen auch im aktuellen Cache.
3. `training_folder/bases` und `hf-datasets` — weitere Kopien. `prefetch_bases.sh`
   holt ein Basismodell gezielt zurück, wenn ein Lauf es braucht.

Der Preis ist **Zeit beim nächsten Lauf**, nicht Datenverlust: ein Basismodell
sind 8–56 GB Download, und ein GPU-Knoten kann sie **nicht** selbst holen (Job
16191716 starb daran), also muss `prefetch_bases.sh` vorher als CPU-Job laufen.

**Gehört auf HuggingFace, ist aber je Modell zu prüfen**

4. `training_folder/trained` und `trained-ubelix` — trainierte Gewichte. Wo das
   Modell unter `dh-unibe/` auf HF liegt, ist die lokale Kopie entbehrlich. Das
   ist je Modell zu prüfen: von den 32 dh-unibe-Modellen auf HF sind nicht alle
   aus diesen Verzeichnissen, und nicht alles hier ist dort.

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
