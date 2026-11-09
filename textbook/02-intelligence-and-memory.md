# Kapitel 2. Architektur von Intelligenz und Gedächtnis: Routing, Wissen, Speicher, Kontextbudget

> Grundlage: Video #2 der Reihe «System Design eines KI-Agenten von innen» — «Teures Modell ≠ kluger Agent. Anatomie des Gehirns eines KI-Agenten» (D. Beresnitski) [1]. Ergänzt und anhand von Primärquellen geprüft; Aussagen ohne Primärquelle sind gekennzeichnet.
> Code zum Kapitel: [`code/ch02/`](../code/ch02/) — [`router.py`](../code/ch02/router.py), [`retrieval.py`](../code/ch02/retrieval.py), [`memory.py`](../code/ch02/memory.py), [`test_ch02.py`](../code/ch02/test_ch02.py) (ausschließlich Python-Standardbibliothek, Ausführung: `python -m unittest test_ch02.py`).
>
> Die Bezeichnungen von Funktionen und Klassen im Text sind anklickbar: Der Link öffnet auf GitHub den jeweiligen Codeabschnitt mit hervorgehobenen Zeilen.

## Lernziele des Kapitels

Nach diesem Kapitel können Sie:

1. Berechnen, was die naive Strategie «Alles in das teuerste Modell» kostet, und erklären, warum die Intelligenz eines Agenten im Harness und nicht allein im Modell verortet ist.
2. Einen dreischichtigen Modell-Router mit Ausfallabsicherung (Fallback) und Berücksichtigung des Prompt-Caches entwerfen.
3. Anhand von fünf Kriterien fundiert zwischen RAG, langen Kontextfenstern und hybriden Architekturen wählen.
4. Das Agenten-Gedächtnis nach dem CoALA-Framework in vier Schichten strukturieren und deren optimale Platzierung im Kontextfenster bestimmen.
5. Asynchrones Lesen und Schreiben von Speicherdaten implementieren, ohne die Antwortzeit zu beeinträchtigen oder Duplikate zu erzeugen.
6. Eine budgetorientierte Kompaktierungsleiter für Kontexte und eine dynamische Thinking-Mode-Policy implementieren.

## Bezug zu Kapitel 1

In Kapitel 1 haben wir das Grundgerüst des Harness aufgebaut: Pipeline, abgesicherter Agent Loop, Werkzeuge. Drei Pipelinestufen blieben dort funktionale Platzhalter: **Gedächtnis & Retrieval** (Stufe 4), **Kontextassemblierung & Budget** (Stufe 6) sowie **Routing & Modus** (Stufe 7). Dieses Kapitel füllt sie mit Leben. Die übergeordnete Ökonomie (API vs. Self-Hosting, TCO) folgt in Kapitel 3; persistentes Langzeitwissen in großem Maßstab in Kapitel 6.

## Durchgängiges Beispiel: Barista Agent v0.2

Unser Café-Assistent aus Kapitel 1 hat neue Fähigkeiten erlernt:

- Er verarbeitet einfache Fragen («Wie viel kostet ein Latte?») ebenso wie komplexe Planungsaufgaben («Stelle eine Bestellung für ein Büro mit 15 Personen zusammen: 3 mit Laktoseintoleranz, Budget 5.000 ₽»);
- Er akzeptiert Bildanhänge («Welches Heißgetränk ist auf diesem Foto zu sehen?»);
- Er durchsucht die interne Wissensbasis nach Rezepturen, Zutaten, Rückgaberegeln und aktuellen Aktionen;
- Er erinnert sich an Vorlieben («trinkt nur Hafermilch», «kein Zucker») und lernt aus Fehlern («Nach Reklamation wegen Allergie: Milchart immer vor dem Bestellabschluss abfragen»).

---

## 2.1. Hauptthese: Teures Modell ≠ kluger Agent

Die naheliegende, aber naive Herangehensweise besteht darin, jede Anfrage mit maximalem Kontext an ein Flaggschiff-Modell (Frontier Model) zu senden. Nach der Beispielrechnung aus dem Video belaufen sich die Kosten pro Vorgang dabei auf rund **3 Dollar**. Bei 1.000 Vorgängen täglich summiert sich dies auf $3 \times 1.000 \times 30 = \mathbf{90.000\ \text{Dollar pro Monat}}$ *(Schätzung des Autors; reale Kosten variieren je nach Modell, Kontextlänge und Iterationszahl)*.

Der überwiegende Teil dieses Budgets wird ineffizient eingesetzt:

- Eine banale Preisabfrage kann von Modellen beantwortet werden, die um das 5- bis 25-Fache günstiger sind;
- Die vollständige Wissensbasis wird in den Prompt geladen, obwohl für die Antwort zwei Zeilen genügen;
- Das Modell generiert teure Denkketten (Thinking Tokens) bei einfachen Werkzeugaufrufen, die rein deterministisch ablaufen sollten.

> **Hauptthese des Kapitels:** Die operative Intelligenz eines Agenten entspringt nicht primär dem Modell, sondern dem umgebenden System: **Routing** (welches Modell bearbeitet die Aufgabe?), **Retrieval** (welches Wissen wird geladen?) und **Kontextbudgetierung** (wie viel wird gelesen und wie tief gedacht?). Ein durchdachter Harness um ein mittelgroßes Modell übertrifft ein unkontrolliertes Spitzenmodell sowohl in puncto Qualität als auch bei den Betriebskosten.

```
Anfrage ─► [Router] ──► Welches Modell und welcher Modus?    (2.2)
             │
             ▼
        [Retrieval] ──► Welches Wissen wird benötigt?         (2.3)
             │
             ▼
        [Speicher] ──► Was wissen wir über Gast und Verlauf? (2.4–2.5)
             │
             ▼
        [Kontextbudget] ──► Was passt hinein und wie viel denken? (2.6–2.7)
             │
             ▼
           Modell
```

---

## 2.2. Intelligentes Routing (Router LLM)

Anfragen variieren in ihrer Komplexität um Größenordnungen. «Was kostet ein Cappuccino?» und «Plane ein Event-Catering mit Diätbeschränkungen» dürfen nicht durch dieselbe teure Recheninstanz bearbeitet werden.

### Die drei Schichten des Routers

| # | Schicht | Aufgabe | Fehlerkosten | Code |
|---|---|---|---|---|
| 1 | **Modalitätenfilter** | Erkennt Medientypen (Text, Bild, Audio) und schließt inkompatible Modelle aus | Keine: deterministische Prüfung | [`route`](../code/ch02/router.py#L99-L146 "code:route") |
| 2 | **Komplexitätsklassifikator** | Kleines Modell oder ML-Klassifikator: «einfach» → günstiges Modell (SLM), «komplex» → Frontier-Modell | Fehleinschätzung nach unten = mangelhafte Antwort; nach oben = Mehrkosten | [`Verdict`](../code/ch02/router.py#L64-L67 "code:Verdict") |
| 3 | **Fallback-Ebene** | Bei Ausfall, Timeout oder niedriger Konfidenz: Keyword-Regeln, danach Standardmodell | Erhöhte Kosten, aber kein Systemausfall | [`rule_fallback`](../code/ch02/router.py#L88-L94 "code:rule_fallback") |

Die dritte Schicht ist unverzichtbar: Der Klassifikator ist ein externer Dienst, der fehlschlagen oder unsichere Ergebnisse liefern kann. Die Benutzeranfrage darf in diesem Fall keinesfalls abbrechen (Test [`test_classifier_failure_falls_back_to_rules`](../code/ch02/test_ch02.py#L42-L45 "code:RouterTests.test_classifier_failure_falls_back_to_rules")).

**Das Standardmodell ist das Frontier-Modell:** Fehlen verlässliche Informationen, ist es im Zweifel sicherer, etwas mehr zu zahlen, als eine fehlerhafte Antwort zu riskieren. Bei extrem preissensitiven Produkten kann diese Entscheidung umgekehrt werden ([`route`](../code/ch02/router.py#L99-L146 "code:route"), Parameter `default_tier`).

> **Präzisierung zum Video:** Eine Klassifizierung im einstelligen Millisekundenbereich gelingt mit traditionellen ML-Modellen (z. B. FastText, Embeddings). Wird ein kleines LLM eingesetzt, bewegt sich die Latenz im Bereich von 50 bis 200 Millisekunden.

### Alternative: Kaskadierung (Escalation)

Statt einer Vorab-Entscheidung kann die Anfrage primär an das günstige Modell gehen. Ein deterministischer Validierer prüft die Ausgabe — erst bei Qualitätsmängeln erfolgt die Eskalation an das Spitzenmodell. Dies eignet sich hervorragend für kurze, gut prüfbare Vorgänge, birgt jedoch Risiken bei langen Chats (siehe nachfolgende Cache-Falle).

Studien zu trainierten Routern (RouteLLM, ICLR 2025) belegen, dass präferenzbasierte Router die Inferenzkosten um über 50% senken können, ohne messbare Qualitätseinbußen im Vergleich zum ausschließlichen Frontier-Einsatz zu verursachen [5].

### Die Falle: Modellwechsel invalidieren den Prompt-Cache

In Kapitel 1 wurde gezeigt, dass Prompt-Caches die Kosten wiederholter Eingaben um rund 90% senken. Dieser Cache ist jedoch **fest an ein spezifisches Modellmodell gebunden**. Wenn ein längerer Dialog über 30 Runden auf einem Frontier-Modell lief und der Router nun entscheidet: «Diese Frage ist simpel — wechsle zum Sparmodell», liest das Sparmodell den gesamten bisherigen Verlauf **ohne Cache** ein und schreibt ihn neu ein (Cache-Writes sind teurer als Standard-Reads).

Beispielrechnung (Frontier: Input \$5, Cache-Read \$0,5, Output \$25 / 1M Token; Sparmodell: Input \$1, Cache-Write \$1,25, Output \$5):

| Dialogrunde mit 100.000 Token Historie | Kosten |
|---|---|
| Auf dem Frontier-Modell verbleiben (Historie aus Cache) | ≈ $0,053 |
| Zum Sparmodell wechseln (Historie ohne Cache neu + Write) | ≈ $0,126 |

Das vorgeblich «günstige» Modell verursacht hier **die 2,4-fachen Kosten**. Bei kurzen Verläufen (500 Token) verhält es sich umgekehrt: Dort spart der Wechsel (\$0,001 vs. \$0,003).

> **Regel:** Die Entscheidung für einen Modellwechsel während einer Sitzung erfolgt auf Basis der **Rundengesamtkosten** unter Berücksichtigung der Cache-Zustände — nicht nach Listenpreis. Modell-Routing ist in der ersten Runde oder bei isolierten Sub-Tasks am wirtschaftlichsten.

Die Hilfsfunktion [`turn_cost`](../code/ch02/router.py#L32-L42 "code:turn_cost") ermittelt die genauen Kosten pro Schritt; [`route`](../code/ch02/router.py#L99-L146 "code:route") vergleicht «bleiben» versus «wechseln» (Tests [`test_cache_aware_stays_on_expensive_model`](../code/ch02/test_ch02.py#L51-L57 "code:RouterTests.test_cache_aware_stays_on_expensive_model") und [`test_short_dialog_switches_to_cheap`](../code/ch02/test_ch02.py#L59-L62 "code:RouterTests.test_short_dialog_switches_to_cheap")).

---

## 2.3. Wissensmanagement: RAG, langes Kontextfenster oder Hybrid

Kontextfenster von 1 bis 2 Millionen Token verleiten oft zu der Annahme, man könne die gesamte Dokumentation ungefiltert in den Prompt laden. Beide Ansätze bringen gravierende Vor- und Nachteile mit sich:

| Kriterium | RAG (Retrieval-Augmented Generation) | Long Context Window |
|---|---|---|
| **Komplexität** | Hoch: Chunking, Embeddings, Vektordatenbank, Reranker, Index-Updates | Minimal: Datei wird direkt in den Prompt geladen |
| **Vollständigkeit** | Gefahr von Lücken bei ungenauem Retrieval oder getrennten Fakten | Das Modell überblickt alles — ideal für globale Synthesen |
| **Wirtschaftlichkeit** | Einmaliges Embedding + gezielte Chunks pro Runde | Gesamtes Dokument wird in jeder Runde gelesen; Caching mildert die Kosten, eliminiert sie aber nicht |
| **Genauigkeit** | Rauscharm: Das Modell muss nicht im Heuhaufen suchen | Risiko von **Lost in the Middle**: Mittlere Kontextbereiche werden schlechter verwertet [3] |
| **Skalierbarkeit** | Praktisch unbegrenzt: Gigabyte bis Terabyte | Begrenzt durch Fenstergröße und Latenzgrenzen |
| **Antwortzeit (TTFT)** | Retrieval dauert Millisekunden, Generierung bei kurzem Prompt extrem schnell | Zeit bis zum ersten Token steigt mit wachsender Kontextlänge |

### Kostenvergleich in Zahlen

Dokument mit 200.000 Token, Dialog über 20 Runden, RAG liefert 3.000 Token pro Runde, Basispreis \$5 pro 1M Eingabetoken ([`compare_costs`](../code/ch02/retrieval.py#L130-L143 "code:compare_costs"), Test [`test_long_context_vs_rag`](../code/ch02/test_ch02.py#L104-L110 "code:RetrievalTests.test_long_context_vs_rag")):

| Architekturvariante | Gesamtkosten des Dialogs |
|---|---|
| Langes Kontextfenster ohne Cache | $20,00 |
| Langes Kontextfenster mit Cache (Read: $0,5 / 1M) | $2,90 |
| RAG (einmaliges Embedding + 3.000 Token pro Runde) | ≈ $0,30 |

Caching verringert den Kostenunterschied von Faktor ~65 auf Faktor ~10, schließt ihn jedoch nicht.

### Die hybride Architektur als Industriestandard

RAG filtert die Wissensbasis auf die relevantesten Textstellen herunter, während das Kontextfenster dem Modell genügend Spielraum lässt, um über diese Chunks im Zusammenhang nachzudenken. Praxismessungen zeigen, dass dieser Hybridansatz den Kontext um den Faktor 4 reduziert und Latenzen drastisch senkt. Unabhängige Benchmarks von Elastic Labs bestätigen diesen Trend: ca. 1 Sekunde Antwortzeit bei RAG gegenüber rund 45 Sekunden bei unkomprimierter Übergabe von ~1 Million Token [4].

Eine interessante Erweiterung stellt die **Self-Route-Methode** dar (Li et al., 2024): Das Modell formuliert zunächst eine Antwort auf Basis der RAG-Auszüge; reicht die Informationsdichte nach eigener Einschätzung nicht aus, wird automatisch der Gesamttext nachgeladen [6].

### Best Practices für zuverlässiges RAG

1. **Überlappendes Chunking:** Nahtstellen zwischen Chunks verhindern Informationsverlust ([`chunk_text`](../code/ch02/retrieval.py#L37-L55 "code:chunk_text")).
2. **Hybride Suche:** BM25 für exakte Artikelnummern und Namen, Vektorsuche für semantische Umschreibungen ([`Index`](../code/ch02/retrieval.py#L58-L86 "code:Index")).
3. **Striktes Wissensbudget:** Chunks greedy nach Relevanz auswählen, bis das Token-Limit erreicht ist ([`assemble`](../code/ch02/retrieval.py#L101-L117 "code:assemble")).
4. **Rand-Verteilung (Edge-Order):** Gegen den «Lost in the Middle»-Effekt werden die relevantesten Chunks an den Anfang und das Ende des Wissensblocks gesetzt ([`edge_order`](../code/ch02/retrieval.py#L89-L98 "code:edge_order")).
5. **Eindeutige Quellenangaben:** Kennzeichnung im Format `[menu#3]`, damit Aussagen prüfbar bleiben.

---

## 2.4. Die vier Gedächtnisschichten (CoALA-Framework)

Das Princeton-Framework **CoALA** (*Cognitive Architectures for Language Agents*, Sumers et al., 2023) [2] unterscheidet vier Gedächtnisschichten, angelehnt an klassische kognitive Architekturen (Soar, ACT-R):

| Typ | Leitfrage | Inhalt | Änderungszyklus | Position im Prompt | Barista Agent | Code |
|---|---|---|---|---|---|---|
| **Prozedural** | *Wie handeln?* | System-Prompt, Tool-Definitionen, Ausführungsregeln | Sehr selten (Releases) | Ganz am Anfang (statischer Cache-Präfix) | Regeln wie «Vor create_order Preis nennen» | [`Procedural`](../code/ch02/memory.py#L26-L33 "code:Procedural") |
| **Semantisch** | *Was wissen?* | Faktenwissen, Speisekarte, FAQ, Gesetze | Bei Wissens-Updates | Dynamischer Block der aktuellen Runde | Menüpreise, Allergene | [`Index`](../code/ch02/retrieval.py#L58-L86 "code:Index") |
| **Episodisch** | *Was geschah?* | Vergangene Vorfälle, Fehler, gelernte Lektionen | Nach Interaktionen fortlaufend | Dynamischer Lektions-Block | «Nach Allergie-Vorfall: Milch immer erfragen» | [`Episode`](../code/ch02/memory.py#L48-L60 "code:Episode") |
| **Arbeitsgedächtnis** | *Was ist jetzt?* | Aktueller Sitzungsverlauf, Tool-Ausgaben | Fortlaufend in jeder Runde | Wachsende Sitzungshistorie | Laufendes Gespräch | [`Working`](../code/ch02/memory.py#L63-L69 "code:Working") |

### Episodische Einträge

Eine gelernte Lektion entfaltet nur dann Wert, wenn sie strukturiert erfasst wird:

```
[Kontext]   Gast erwähnt Laktoseintoleranz
[Aktion]    Agent schlägt klassischen Latte Macchiato vor
[Ergebnis]  Beschwerde des Gastes
[Lektion]   Bei Nennung von Allergien immer alternative Milcharten vorschlagen
[Quelle]    Session 2026-10-12 / Message 14
```

Ohne die Verknüpfung zur Primärquelle (Provenienz-Anker) kann eine Lektion bei fehlerhafter Interpretation später weder verifiziert noch korrigiert werden.

### Anordnung im Kontextfenster

Unter Berücksichtigung der Caching-Regeln ergibt sich folgende Strukturierung:

```
┌─ System-Prompt ────────────────────────┐  Prozedural: Unveränderlich → Maximal gecacht
│ Rolle, Kernregeln, Tools              │
├─ Sitzungsverlauf ──────────────────────┤  Arbeitsgedächtnis: Reine Historie → Gecacht
│ User- / Assistant- / Tool-Messages    │
├─ Dynamischer Block der Runde ──────────┤  Variiert in jeder Runde → Ganz ans Ende
│ <knowledge> RAG-Auszüge                │  Semantisch
│ <memory>    Gastpräferenzen            │
│ <lessons>   Erfahrungswerte            │  Episodisch
└────────────────────────────────────────┘
```

Die Funktion [`build_context`](../code/ch02/memory.py#L219-L237 "code:build_context") setzt diese Aufteilung um; Test [`test_context_static_first_dynamic_last`](../code/ch02/test_ch02.py#L184-L190 "code:MemoryTests.test_context_static_first_dynamic_last") verifiziert die Stabilität des Präfixes.

---

## 2.5. Praxisanalyse: Die Speicherarchitektur von Claude Code

Öffentlich zugängliche Architekturdetails produktiver Speichersysteme sind rar. Ende März 2026 gelangte durch eine Fehlkonfiguration im npm-Paket von Claude Code 2.1.88 ein Source Map an die Öffentlichkeit, aus dem über 500.000 Zeilen TypeScript rekonstruiert werden konnten [7]. Nachfolgend werden zentrale, verifizierte Architekturmuster analysiert:

### Muster 1: Strikte Trennung von Statik und Dynamik
Der System-Prompt ist durch klare Trennmarken unterteilt [8]:
- Ein **statischer Block** oben (Regeln, Tool-Verträge), der für alle Nutzer identisch ist und globale Cache-Hits ermöglicht;
- Ein **dynamischer Block** unten (`CLAUDE.md`, Projektumgebung, Speicherindizes).

### Muster 2: Speicherindex und Prefetching
Anstatt alle Profildateien pauschal zu laden, existiert eine schlanke Indexdatei (`MEMORY.md`, ca. 200 Zeilen). Ein **separates, günstiges Modell** analysiert die Dateibeschreibungen und wählt bis zu 5 relevante Dokumente aus [9].
Entscheidend: **Das Prefetching blockiert den Hauptprozess nicht**. Die Abfrage läuft asynchron im Hintergrund, während Pipeline und Routing vorbereitet werden. Trifft das Ergebnis nicht rechtzeitig ein, arbeitet der Agent ohne Verzögerung mit dem Basiskontext weiter ([`start_prefetch`](../code/ch02/memory.py#L78-L93 "code:start_prefetch"), [`collect_prefetch`](../code/ch02/memory.py#L96-L103 "code:collect_prefetch"), Tests [`test_prefetch_limits_and_filters`](../code/ch02/test_ch02.py#L117-L121 "code:MemoryTests.test_prefetch_limits_and_filters"), [`test_prefetch_does_not_block`](../code/ch02/test_ch02.py#L123-L132 "code:MemoryTests.test_prefetch_does_not_block")).

### Muster 3: Asynchrones Schreiben und Idempotenz
Nach der Antwort an den Benutzer analysiert ein separater Hintergrund-Agent den Dialog und sichert neue Erkenntnisse. Er arbeitet mit einem **Cursor** nach dem **At-Least-Once-Prinzip**: Der Cursor wird erst nach erfolgreichem Schreibvorgang weitergerückt [9].
Daraus folgt zwingend: **Schreiboperationen müssen idempotent sein**, da bei Verbindungsabbrüchen dieselbe Nachricht mehrfach verarbeitet wird ([`EpisodicWriter`](../code/ch02/memory.py#L111-L153 "code:EpisodicWriter"), [`Episode.key`](../code/ch02/memory.py#L57-L60 "code:Episode.key"), Tests [`test_episodic_writer_at_least_once_is_idempotent`](../code/ch02/test_ch02.py#L134-L144 "code:MemoryTests.test_episodic_writer_at_least_once_is_idempotent"), [`test_episodic_cursor_not_advanced_on_failure`](../code/ch02/test_ch02.py#L146-L153 "code:MemoryTests.test_episodic_cursor_not_advanced_on_failure")).

---

## 2.6. Kontext-Kompaktierungsleiter (Compaction Ladder)

Wenn die Historie anwächst und das Token-Limit erreicht wird, muss komprimiert werden. Auch hier gilt: **Von günstigen und verlustfreien Maßnahmen hin zu teuren, verlustbehafteten Schritten**.

| Stufe | Maßnahme | Informationsverlust | Einfluss auf den Cache | Code |
|---|---|---|---|---|
| 1. **Offloading** | Sehr lange Tool-Ergebnisse werden auf Platte ausgelagert; im Prompt verbleiben nur die ersten ~2.000 Zeichen und der Dateipfad | Keiner (Modell kann die Datei gezielt nachlesen) | Keiner: greift sofort beim Eintreffen des Ergebnisses | [`compact`](../code/ch02/memory.py#L175-L214 "code:compact") |
| 2. **Microcompact** | Alte Tool-Ergebnisse im Verlauf werden durch Platzhalter wie `[bereinigt]` ersetzt; jüngste Nachrichten bleiben intakt | Gering: Historische Tool-Rohdaten werden selten erneut gebraucht | Invalidiert den Cache ab der ersten bereinigten Stelle | dort |
| 3. **Autocompact** | Der ältere Teil des Gesprächs wird durch eine strukturierte Zusammenfassung ersetzt | Moderat bis hoch: Nuancen gehen verloren | Vollständiger Reset des Gesprächscaches | dort |

Tests: [`test_offload_keeps_head_and_path`](../code/ch02/test_ch02.py#L155-L163 "code:MemoryTests.test_offload_keeps_head_and_path"), [`test_compaction_ladder_escalates`](../code/ch02/test_ch02.py#L165-L173 "code:MemoryTests.test_compaction_ladder_escalates"), [`test_autocompact_when_micro_not_enough`](../code/ch02/test_ch02.py#L175-L182 "code:MemoryTests.test_autocompact_when_micro_not_enough").

### Anforderungen an ein verlässliches Resümee

Gehen beim Zusammenfassen kritische Fakten verloren, verhält sich der Agent erratisch (z. B. doppelte Abbuchung bereits bezahlter Bestellungen). Ein Resümee muss zwingend enthalten:
- Das ursprüngliche Nutzerziel und den aktuellen Abarbeitungsstand;
- **Bereits ausgeführte irreversible Aktionen** (Transaktions-IDs, bezahlte Beträge);
- Offene Rückfragen und Fehlermeldungen;
- Dateipfade zu ausgelagerten Nutzlasten.

---

## 2.7. Thinking Mode: Gezielt nachdenken statt verschwenderisch rechnen

Reasoning-Token werden als **Ausgabetoken** abgerechnet und sind signifikant teurer als Eingaben. Ein Agenten-Loop gliedert sich in distinkte Arbeitsphasen:

| Phase | Iterationen | Typische Aktivität | Thinking Mode |
|---|---|---|---|
| **Planung** | Erste 1–2 Runden | Aufgabenzerlegung, Budgetberechnung, Strategiewahl | **Aktiviert** |
| **Ausführung** | Nachfolgende Runden | Abarbeitung definierter Tool-Aufrufe | **Deaktiviert** |
| **Re-Planning** | Nach Fehlern | Fehlerursache analysieren, Alternativen prüfen | **Kurzzeitig aktiviert** |

Die Policy [`ThinkingPolicy`](../code/ch02/router.py#L151-L164 "code:ThinkingPolicy") steuert dieses Token-Kontingent dynamisch nach Iterationsnummer und Status (Test [`test_thinking_policy`](../code/ch02/test_ch02.py#L68-L71 "code:RouterTests.test_thinking_policy")).

---

## Zusammenfassung des Kapitels

1. Ein teures Modell macht noch keinen intelligenten Agenten. Die Architektur des Harness entscheidet über Kosten, Latenz und Zuverlässigkeit.
2. Der Router arbeitet mehrschichtig: Modalitätenfilter → Komplexitätsklassifikator → deterministischer Fallback.
3. Cache-Effekte beim Modellwechsel beachten: Bei langen Dialogen ist das Verbleiben auf dem Frontier-Modell oft günstiger als der Wechsel zum Sparmodell.
4. RAG und langes Kontextfenster ergänzen sich im hybriden Ansatz: Geringe Latenz bei hoher Reasoning-Tiefe.
5. Gedächtnis nach CoALA strukturiert ablegen: Statische Prozeduren nach oben, dynamische Fakten und Lektionen nach unten.
6. Lese- und Schreibvorgänge laufen asynchron im Hintergrund; At-Least-Once erfordert strikte Idempotenz.
7. Kontext-Kompaktierung erfolgt stufenweise: Offload vor Bereinigung, Bereinigung vor Zusammenfassung.
8. Reasoning-Modi gezielt in der Planungsphase einsetzen, nicht während standardisierter Tool-Ausführungen.

## Fragen zur Selbstkontrolle

1. Warum sind 3 Dollar pro Task ein Architekturproblem und kein reines Modellproblem? Welche Mechanismen senken diese Kosten?
2. Warum benötigt ein Router zwingend eine Fallback-Ebene, selbst wenn der Klassifikator eine Trefferquote von 95% aufweist?
3. In Runde 40 meldet der Klassifikator eine «einfache Frage». Welche Daten sind erforderlich, um die Wirtschaftlichkeit eines Modellwechsels zu bewerten?
4. Weshalb ist das unbedachte Laden von 1 Million Token selbst bei aktivem Prompt Caching problematisch?
5. Was beschreibt der «Lost in the Middle»-Effekt und wie hilft die «Edge-Order»-Sortierung?
6. Ordnen Sie die folgenden Elemente den vier CoALA-Speichertypen zu: Tool-Schema von `create_order`, Erstattungsrichtlinie des Cafés, Beschwerde über kalten Kaffee von gestern, aktuelle Positionen im Warenkorb.
7. Warum muss ein im Hintergrund agierender Speicher-Writer idempotent arbeiten?
8. Weshalb wird der Speicher-Prefetch mit einem Timeout abgebrochen, statt zwingend auf das Ergebnis zu warten?
9. Welche Daten dürfen bei einer automatischen Kontextzusammenfassung keinesfalls verloren gehen?
10. In welchen Iterationen des Agent Loops sollte der Thinking Mode aktiviert sein und warum?

## Praxisaufgabe: Barista Agent v0.2

Basiscode unter `code/ch02/`.

1. **Ausführen:** Starten Sie `python -m unittest test_ch02.py -v` und ordnen Sie die Tests den Abschnitten des Kapitels zu.
2. **Klassifikator bauen:** Implementieren Sie einen LLM-Klassifikator für [`route`](../code/ch02/router.py#L99-L146 "code:route") mit wenigen Few-Shot-Beispielen und JSON-Rückgabe. Messen Sie Latenz und Treffsicherheit über 50 Testfragen.
3. **Kaskadierung:** Bauen Sie ein Eskalationsmuster: Sparmodell generiert Antwort → Validierer prüft → bei Mängeln Übergabe an Frontier-Modell.
4. **Cache-Break-Even:** Ermitteln Sie anhand realer API-Preise die exakte Kontextlänge, ab der ein Modellwechsel teurer wird als das Verbleiben auf dem teuren Modell.
5. **Hybride Suche:** Erweitern Sie den [`Index`](../code/ch02/retrieval.py#L58-L86 "code:Index") um Vektor-Embeddings und kombinieren Sie die Treffer via Reciprocal Rank Fusion (RRF).
6. **Episodisches Lernen:** Integrieren Sie den [`EpisodicWriter`](../code/ch02/memory.py#L111-L153 "code:EpisodicWriter") in den Agent Loop aus Kapitel 1, sodass Lektionen nach Reklamationen im nächsten Gespräch automatisch bereitstehen.

## Glossar

| Begriff | Definition |
|---|---|
| **At-Least-Once** | Zustellgarantie, bei der Nachrichten mindestens einmal eintreffen; erfordert idempotente Empfänger |
| **BM25** | Bewährter Ranking-Algorithmus für lexikalische Volltextsuchen |
| **Chunking** | Segmentierung von Dokumenten in handhabbare Abschnitte für Indizierung und Retrieval |
| **CoALA** | Cognitive Architectures for Language Agents — Referenzframework für modulare Agentenspeicher |
| **Compaction** | Systematische Verdichtung des Kontextfensters bei drohender Budgetüberschreitung |
| **Embeddings** | Vektorielle Repräsentationen von Texten zur semantischen Ähnlichkeitssuche |
| **Frontier-Modell** | Leistungsfähigstes Flaggschiff-Modell der aktuellen Generation |
| **Idempotenz** | Eigenschaft einer Operation, bei mehrfacher Ausführung dasselbe Resultat wie beim Erstaufruf zu liefern |
| **Lost in the Middle** | Nachlassende Aufmerksamkeit von Sprachmodellen für Informationen in der Mitte langer Prompts |
| **Prefetching** | Asynchrones Vorabladen relevanter Daten parallel zur Hauptverarbeitung |
| **Reranker** | Nachgelagertes Modell zur präzisen Neusortierung gefundener Dokumentfragmente |
| **Router** | Komponente zur dynamischen Zuweisung von Anfragen an passende Modelle und Modi |
| **SLM** | Small Language Model — kompaktes, schnelles Modell für fokussierte Teilaufgaben |
| **Thinking Tokens** | Interne Denkketten-Token zur Vorbereitung komplexer Antworten |
| **TTFT** | Time To First Token — Zeitspanne bis zum Eintreffen des ersten Antwort-Tokens |

## Quellen

1. D. Beresnitski. *Дорогая модель ≠ умный агент. Анатомия мозга AI-агента* (Video #2). <https://youtu.be/vjMxeQ3aIGM>
2. T. Sumers, S. Yao, K. Narasimhan, T. Griffiths. *Cognitive Architectures for Language Agents* (2023). <https://arxiv.org/abs/2309.02427>
3. N. Liu et al. *Lost in the Middle: How Language Models Use Long Contexts* (2023). <https://arxiv.org/abs/2307.03172>
4. Elastic Search Labs. *Longer context ≠ better: Why RAG still matters*. <https://www.elastic.co/search-labs/blog/rag-vs-long-context-model-llm>
5. I. Ong et al. *RouteLLM: Learning to Route LLMs with Preference Data* (ICLR 2025). <https://arxiv.org/abs/2406.18665>
6. Z. Li et al. *Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach* (2024). <https://arxiv.org/abs/2407.16833>
7. DeepLearning.AI, The Batch. *Claude Code's source code leaked* (2026). <https://www.deeplearning.ai/the-batch/claude-codes-source-code-leaked-exposing-potential-future-features-kairos-and-autodream>
8. kuber.studio. *Claude Code's Entire Source Code Got Leaked via a Sourcemap in npm* (2026). <https://kuber.studio/blog/AI/Claude-Code's-Entire-Source-Code-Got-Leaked-via-a-Sourcemap-in-npm,-Let's-Talk-About-it.md>
9. Mem0. *How memory works in Claude Code* (2026). <https://mem0.ai/blog/how-memory-works-in-claude-code>
10. Anthropic. *Prompt caching*, *Extended thinking* (API-Dokumentation). <https://docs.claude.com>
