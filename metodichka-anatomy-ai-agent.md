# Leitfaden: Production AI Agent Engineering (Entwicklung produktionsreifer KI-Agenten)

> **Autorenkurs / Systemanalyse**: Dmitri Beresnitski  
> **Format**: Vollständiges Praxishandbuch zu Architektur, Speicher, Ökonomie, Harness Engineering, Multi-Agenten-Systemen und Produktivbetrieb.  
> **Zielgruppe**: Senior Backend Engineers, AI/ML Engineers, Solutions Architects, Tech Leads.  
> **Status**: Kompakte Kurszusammenfassung. Ausführliche Lehrbuchkapitel mit Code und Übungen befinden sich im Ordner [`textbook/`](textbook/). Modul 1 ist mit dem Transkript von Video #1 abgeglichen, Modul 2 mit dem Skript von Video #2; die Module 3–7 erwarten den Abgleich mit den Videos.

---

## Inhaltsverzeichnis
1. [Modul 1. Anatomie eines KI-Agenten: Pipeline, Loop, Tools, Fallstricke](#modul-1-anatomie-eines-ki-agenten-pipeline-loop-tools-fallstricke)
   - [1.1. Prototyp vs. Produktion](#11-prototyp-vs-produktion)
   - [1.2. Workflow vs. Agent: Die Komplexitätsleiter](#12-workflow-vs-agent-die-komplexitätsleiter)
   - [1.3. Der System-Prompt als Vertrag](#13-der-system-prompt-als-vertrag)
   - [1.4. Pipeline: «Früh blockieren, spät ausgeben»](#14-pipeline-früh-blockieren-spät-ausgeben)
   - [1.5. Ökonomie des Cachings: KV-Cache](#15-ökonomie-des-cachings-kv-cache)
   - [1.6. Absicherung des Agent Loops im Code](#16-absicherung-des-agent-loops-im-code)
   - [1.7. Tool-Design und die MCP-Falle](#17-tool-design-und-die-mcp-falle)
   - [1.8. Observability ab Tag 1](#18-observability-ab-tag-1)
2. [Modul 2. Architektur von Intelligenz und Gedächtnis eines KI-Agenten](#modul-2-architektur-von-intelligenz-und-gedächtnis-eines-ki-agenten)
   - [2.1. Die 4 Speichertypen nach dem CoALA-Framework](#21-die-4-speichertypen-nach-dem-coala-framework)
   - [2.2. Vergleichsanalyse: RAG vs. langes Kontextfenster](#22-vergleichsanalyse-rag-vs-langes-kontextfenster)
   - [2.3. Strategisches Denken (Thinking Mode) und Token-Budgetierung](#23-strategisches-denken-thinking-mode-und-token-budgetierung)
3. [Modul 3. Ökonomie und Modellauswahl in der Produktion](#modul-3-ökonomie-und-modellauswahl-in-der-produktion)
   - [3.1. Strategische Entscheidung: API vs. Self-Hosted (Open-Source)](#31-strategische-entscheidung-api-vs-self-hosted-open-source)
   - [3.2. Berechnung der Total Cost of Ownership (TCO)](#32-berechnung-der-total-cost-of-ownership-tco)
   - [3.3. Hybrides Routing und Klassifikator-Architektur](#33-hybrides-routing-und-klassifikator-architektur)
4. [Modul 4. Harness Engineering: Entwurf der Ausführungsumgebung](#modul-4-harness-engineering-entwurf-der-ausführungsumgebung)
   - [4.1. Paradigmenwechsel: Vom Prompt Engineering zum Harness Engineering](#41-paradigmenwechsel-vom-prompt-engineering-zum-harness-engineering)
   - [4.2. Die 4 Steuerungsfunktionen (OpenAI Framework)](#42-die-4-steuerungsfunktionen-openai-framework)
   - [4.3. Die 7 Schichten des System-Harness (Harness Architecture)](#43-die-7-schichten-des-system-harness-harness-architecture)
   - [4.4. Dreistufige Verifikation von Invarianten](#44-dreistufige-verifikation-von-invarianten)
5. [Modul 5. Multi-Agenten-Systeme und Architekturmuster](#modul-5-multi-agenten-systeme-und-architekturmuster)
   - [5.1. Kognitive Überlastung eines einzelnen Modells](#51-kognitive-überlastung-eines-einzelnen-modells)
   - [5.2. 7 Kernmuster der Multi-Agenten-Interaktion](#52-7-kernmuster-der-multi-agenten-interaktion)
   - [5.3. Overheads, Anti-Patterns und das Prinzip progressiver Komplexität](#53-overheads-anti-patterns-und-das-prinzip-progressiver-komplexität)
6. [Modul 6. Langzeitgedächtnis und Wissensmanagement](#modul-6-langzeitgedächtnis-und-wissensmanagement)
   - [6.1. Architekturgabelung: Synthese beim Schreiben vs. Synthese beim Lesen](#61-architekturgabelung-synthese-beim-schreiben-vs-synthese-beim-lesen)
   - [6.2. Schreibdisziplin: Append-Only-Logs und Provenienz-Anker](#62-schreibdisziplin-append-only-logs-und-provenienz-anker)
   - [6.3. Asynchroner Schiedsrichter-Agent (WikiGroomer) und Concept Drift](#63-asynchroner-schiedsrichter-agent-wikigroomer-und-concept-drift)
   - [6.4. Speichervergiftung (Memory Poisoning) und Prompt Injection](#64-speichervergiftung-memory-poisoning-und-prompt-injection)
7. [Modul 7. Produktivbetrieb: Sicherheit, Resilienz und Observability](#modul-7-produktivbetrieb-sicherheit-resilienz-und-observability)
   - [7.1. Mehrschichtige Sicherheit (Defense-in-Depth) und übermäßige Autonomie](#71-mehrschichtige-sicherheit-defense-in-depth-und-übermäßige-autonomie)
   - [7.2. Ausfallsicherheit (Resilience): Circuit Breakers, Idempotenz und Fehlerstrategie](#72-ausfallsicherheit-resilience-circuit-breakers-idempotenz-und-fehlerstrategie)
   - [7.3. Durchgängige Observability und 5 kritische Agenten-Metriken](#73-durchgängige-observability-und-5-kritische-agenten-metriken)
   - [7.4. Fazit: Deployment-Readiness-Checkliste](#74-fazit-deployment-readiness-checkliste)

---

## Modul 1. Anatomie eines KI-Agenten: Pipeline, Loop, Tools, Fallstricke

> Ausführliche Version des Moduls mit Beispielen, Code und Übungen — [Kapitel 1 des Lehrbuchs](textbook/01-anatomy-of-an-ai-agent.md), Code — [`code/ch01/`](code/ch01/).

### 1.1. Prototyp vs. Produktion
Ein Agent mit LangChain, AutoGen oder CrewAI lässt sich in wenigen Zeilen aufbauen. In der Produktion jedoch stößt das System schnell auf unkontrollierte Kosten, Endlosschleifen und unvorhersehbares Verhalten.

* **DataTalks.Club (Februar 2026)**: Claude Code führte `terraform destroy` aus und löschte die Produktionsinfrastruktur der Kurse mitsamt automatischen Snapshots (~2,5 Jahre studentischer Arbeiten). Ursache war eine Kette ingenieurtechnischer Fehler: Die Terraform-State-Datei war nicht migriert worden, der Agent ersetzte den State durch eine Archivkopie und der menschliche Operator stoppte das `destroy` nicht. Die Daten wurden nach 24 Stunden mit Hilfe des AWS-Supports wiederhergestellt.
* **Gartner (Juni 2025)**: Mehr als 40% der Projekte für agentische KI werden bis Ende 2027 aufgrund von Kosten, unklarem Mehrwert und mangelhafter Risikokontrolle eingestellt werden.
* Im Video werden zudem Rechnungen von ~\$2,5 Mio./Monat gegenüber \$50 im Prototyp sowie ein Bot genannt, der dieselbe Antwort 58 Mal hintereinander wiederholte *(Primärquellen im Video nicht angegeben)*.

> **Hauptthese**: Das Problem liegt nicht im Modell, sondern im Engineering drumherum. Erfolgreiche Systeme basieren auf einfachen, komponierbaren Mustern und striktem Code — nicht auf «magischen» Frameworks.

---

### 1.2. Workflow vs. Agent: Die Komplexitätsleiter
* **Workflow («Feste Schienen»)**: Die Schrittabfolge wird durch deterministischen Code vorgegeben. Das LLM führt den Schritt aus, wählt jedoch **nicht den nächsten Schritt**. Maximales Maß an Kontrolle und Vorhersehbarkeit, minimale Kosten.
* **Agent («Autonomie»)**: Das Modell entscheidet eigenständig über Werkzeugaufrufe und den Folgeschritt. Hohe Flexibilität um den Preis von Latenz, Kosten und Fehlerrisiken.

| # | Stufe | Kernprinzip |
| :--- | :--- | :--- |
| 1 | Single LLM Call | Eine Anfrage — eine Antwort. Wenn ausreichend, keinesfalls verkomplizieren |
| 2 | Prompt Chaining | Ausgabe von Schritt N ist Eingabe von Schritt N+1; hier lässt sich Human-in-the-Loop einbinden |
| 3 | Routing | Schneller, günstiger Klassifikator leitet die Anfrage in eine spezialisierte Kette |
| 4 | Parallelization | Unabhängige Teilaufgaben werden parallel ausgeführt |
| 5 | Orchestrator–Workers | LLM-Orchestrator zerlegt die Aufgabe und verteilt sie an Worker |
| 6 | ReAct Agent | Modell plant selbst, wählt Werkzeuge und iteriert in einer Schleife bis zum Ergebnis |
| 7 | Multi-Agent System | Mehrere Agenten arbeiten als Team zusammen (siehe Modul 5) |

Die Stufen 1–5 sind Workflows, 6–7 sind Agenten. Im Anthropic-Leitfaden «Building effective agents» zählt auch das Muster **Evaluator–Optimizer** (Generierung + Bewertung in einer Schleife) zu den Workflows.

> **Goldene Regel**: Erhöhen Sie die Autonomie nur dann, wenn ein deterministischer Workflow mit der Varianz der Aufgabenstellung nicht mehr zurechtkommt.

---

### 1.3. Der System-Prompt als Vertrag
Der System-Prompt ist ein architektonischer Verhaltensvertrag: Er wird wie Code versioniert und getestet (Eval-Suiten aus typischen und bösartigen Anfragen). Vier obligatorische Abschnitte:

1. **Rolle und explizite Grenzen** — was der Agent tut und was er strikt unterlässt.
2. **Priorisierte Regeln** — nummerierte Liste; bei Konflikten gilt: Sicherheit > Budget > Komfort.
3. **Antwortformat** — JSON-Schema, Markdown etc.
4. **Beispiel (Few-Shot)** — idealtypischer Beispieldialog.

Dynamische Daten (Uhrzeit, Benutzername) gehören nicht in den System-Prompt — sie zerstören den KV-Cache (Abschnitt 1.5).

---

### 1.4. Pipeline: «Früh blockieren, spät ausgeben»
Günstige Prüfungen zuerst, teure LLM-Aufrufe so spät wie möglich. Jede Stufe kann die Verarbeitung abbrechen.

```
[Anfrage]
  1. Validation ($0) ─────────────► Abweisung: leer / ungültig / zu lang
  2. Input Security ──────────────► Abweisung: Prompt Injection
  3. Context Enrichment             (Kalender, Tasks, Systemstatus)
  4. Memory & Retrieval             (Präferenzen, RAG)
  5. Content Filtering ───────────► Bereinigung: fehlerhafte Dokumente, Leaks, indirekte Injections
  6. Context Assembly & Budgeting   (Greedy / Zusammenfassung / Compaction)
  7. Smart Routing & Mode           (günstiges/teures Modell, Thinking Mode)
  8. Generation                     (LLM-Aufruf / Agent Loop)
  9. Output Security ─────────────► Sperre: PII, System-Prompt-Leak (Canary-Token)
 10. Formatting & Validation ─────► Retry / Fehler: invalides JSON
[Antwort]
```

* Gefundene Dokumente und **Tool-Antworten** sind nicht vertrauenswürdige Eingaben: Indirekte Prompt Injections gelangen über sie ins System, weshalb Prüfungen in Stufe 2 allein nicht ausreichen.
* Im Video werden 12 Stufen angekündigt, im Transkript 10 konkretisiert; Stufen können zusammengefasst oder aufgeteilt werden — entscheidend ist die Reihenfolge «günstig → teuer».

---

### 1.5. Ökonomie des Cachings: KV-Cache
* In einem Agent Loop liest das Modell in jeder Iteration den gesamten Kontext neu; das Verhältnis von Eingabe- zu Ausgabetoken liegt oft bei etwa 100:1 (Schätzung Manus).
* Der KV-Cache (Prompt Caching) vergünstigt wiederholt gelesene Präfixe um etwa den Faktor 10 — **jedoch nur, wenn das Präfix unverändert bleibt**.
* Ohne Caching wachsen die Kosten einer Aufgabe quadratisch mit der Anzahl der Schritte (in Schritt *i* werden ~*i·k* Token gelesen).

**Was den Cache zerstört**: Zeitangaben zu Beginn des System-Prompts, Modifikationen in der Mitte des Kontexts, instabile Schlüsselreihenfolgen in JSON.

> **Regel**: Statisches nach vorn, Dynamisches ans Ende; in der Schleife werden Daten ausschließlich angehängt; JSON wird mit stabiler Schlüsselreihenfolge serialisiert.

---

### 1.6. Absicherung des Agent Loops im Code
Risiken eines ungeschützten Loops: Endlosschleifen, Kostenexplosionen, Halluzinationskaskaden (der Fehler aus Schritt N wird zum «Fakt» für Schritt N+1).

Schutzbarrieren gehören **in den Code, nicht in den Prompt**:

1. **Iterationslimit** — z. B. ~5 für Kundensupport, ~25 für Coding-Agenten.
2. **Token-Budget** pro Aufgabe/Sitzung.
3. **Limit identischer aufeinanderfolgender Tool-Aufrufe** (Name + Argumente). Wiederholungen nach Timeouts oder Statusabfragen sind nicht zwingend fehlerhaft, daher werden exakt identische Aufrufe gezählt.
4. **Human-in-the-Loop** vor irreversiblen Aktionen (Löschen, Bezahlen, E-Mail-Versand): Der Loop pausiert unter Sicherung des Zustands und setzt erst nach Freigabe durch den Menschen fort.
5. **Zeitlimit** für die Gesamtaufgabe und **Timeout** für jedes einzelne Werkzeug *(Ergänzung zum Video)*.

Wird ein Limit erreicht, erhält das Modell eine konstruktive Handlungsanweisung statt einer abstrakten Fehlermeldung: «Das Tool wurde 4 Mal aufgerufen, Daten liegen vor. Erstelle jetzt die finale Antwort». Danach werden ihm keine Tools mehr übergeben.

---

### 1.7. Tool-Design und die MCP-Falle
* **Anzahl**: Bei einer wachsenden Anzahl von Werkzeugen (ab ca. 20–50) droht der «Tod durch tausend Instruktionen» — Grenzen verschwimmen, das Modell verwechselt Werkzeuge.
* **Beschreibung**: Namenskonvention `verb_substantiv` (`get_user_preferences`), positive Formulierungen («was und wann zu tun ist»), striktes JSON-Schema und Standardwerte.
* **Fehler als Instruktionen**: Kein `500 Internal Server Error`, sondern eine Erklärung, was schiefging und wie der nächste Schritt korrigiert werden kann.
* **Die Falle automatischer MCP-Wrapper**: Die automatische Generierung eines MCP-Servers aus einer OpenAPI-Spezifikation deckt nur den Happy Path ab. Ohne Timeout-Handling, domänenspezifische Fehlerbehandlung und Kennzeichnung gefährlicher Operationen scheitert das System im Produktivbetrieb. Werkzeuge müssen für die Aufgaben des Agenten entworfen werden, nicht 1:1 nach REST-Endpunkten.

---

### 1.8. Observability ab Tag 1
Observability beantwortet die Frage, warum eine Anfrage \$12 statt \$0,02 gekostet und 8 Sekunden gedauert hat. Ab Tag 1 müssen geloggt werden: Start und Ende jeder Pipelinestufe, Modellauswahl und Prompt-Version, Loop-Iterationen und Tokenverbrauch, Werkzeugparameter und -antworten, **KV Cache Hit Rate** sowie die Gesamtkosten der Anfrage *(Details in Modul 7.3)*.

---

## Modul 2. Architektur von Intelligenz und Gedächtnis eines KI-Agenten

> Ausführliche Fassung des Moduls — [Kapitel 2 des Lehrbuchs](textbook/02-intelligence-and-memory.md), Code — [`code/ch02/`](code/ch02/). Das Kapitel behandelt zudem cache-bewusstes Routing, das Speichersystem von Claude Code und die Kompaktierungsleiter für Kontexte.

**Kernbotschaft von Video #2**: Ein teures Modell ergibt noch keinen intelligenten Agenten. «Alles in ein Frontier-Modell mit vollem Kontext» entspricht ca. \$3 pro Aufgabe ≈ \$90.000/Monat bei 1.000 Aufgaben täglich *(Autorenschätzung)*. Die Intelligenz des Agenten entsteht in der Umgebung: Routing, Retrieval, Kontextbudgetierung.

### 2.1. Die 4 Speichertypen nach dem CoALA-Framework
Gemäß dem CoALA-Framework (Cognitive Architectures for Language Agents, Sumers et al., Princeton, 2023) unterteilt sich das Gedächtnis eines KI-Agenten in 4 Schichten mit unterschiedlichen Lebenszyklen. CoALA bezieht sich dabei auf kognitive Architekturen (Soar, ACT-R); die Analogie «Kontext = RAM» wurde durch MemGPT/Letta geprägt.

```
+--------------------+--------------------+-------------------------------+-------------------------------+
|                                     AI AGENT MEMORY                                                   |
+--------------------+--------------------+-------------------------------+-------------------------------+
| Prozedural         | Semantisch         | Episodisch                    | Working Memory                |
| (Procedural)       | (Semantic)         | (Episodic)                    | (Working)                     |
+--------------------+--------------------+-------------------------------+-------------------------------+
| System-Prompt,     | Wissensbasen, RAG, | Erfahrungsjournal, Protokoll  | Aktuelles Kontextfenster,     |
| Tool-Instruktionen | Projektfakten      | von Erfolgen und Fehlern      | Zustand der aktiven Aufgabe   |
| Ändert sich selten | Wächst stetig      | Bereichert Handlungen         | Wird nach Abschluss geleert   |
+--------------------+--------------------+-------------------------------+-------------------------------+
```

1. **Prozedurales Gedächtnis (Procedural Memory)**: Wissen darüber, *wie Aufgaben gelöst werden*. Beinhaltet System-Prompts, Validierungsregeln, Tool-Aufrufverträge; in CoALA auch den Code des Agenten und die Modellgewichte. Wird vom Entwickler zwischen Releases gepflegt.
2. **Semantisches Gedächtnis (Semantic Memory)**: Wissen über *Fakten und die Welt*. Umfasst externe Wissensdatenbanken, Vektorspeicher (RAG) und Dokumentationen.
3. **Episodisches Gedächtnis (Episodic Memory)**: Erfahrung aus *vergangenen Aktionen und Lektionen*. Speichert Einträge im Format: `[Kontext] -> [Aktion] -> [Ergebnis] -> [Erkenntnis]`. Verhindert, dass der Agent Fehler von gestern wiederholt *(siehe Implementierung in [Modul 6.3](#63-asynchroner-schiedsrichter-agent-wikigroomer-und-concept-drift))*.
4. **Arbeitsgedächtnis (Working Memory)**: Der operative Kontext der aktuellen Aufgabe. Begrenzt durch das Kontextfenster; wird nach Sitzungsende geleert oder zusammengefasst.

---

### 2.2. Vergleichsanalyse: RAG vs. langes Kontextfenster
Mit dem Aufkommen von Modellen mit 1–2 Millionen Token Kontext entstand der Irrglaube, RAG sei überflüssig. Eine ingenieurtechnische Analyse offenbart klare Kompromisse:

| Kriterium | RAG (Retrieval-Augmented Generation) | Long Context Window (1M+ Token) |
| :--- | :--- | :--- |
| **Infrastrukturkomplexität** | Hoch (Vektordatenbanken, Chunking, Embeddings, Reranking) | Minimal (gesamte Datei direkt in den Prompt laden) |
| **Betriebskosten** | Gering: Einmalige Indexierung + gefundene Chunks pro Runde + Infrastruktur | Hoch: Das gesamte Dokument wird bei jedem Aufruf gelesen (Caching reduziert den Abstand von ca. 65× auf 10×, hebt ihn aber nicht auf — siehe Kapitel 2) |
| **Präzision lokaler Suchen** | Hoch (gezielte Extraktion relevanter Abschnitte ohne Rauschen) | Anfällig für *Lost in the Middle* und *Context Erosion* |
| **Vollständigkeit globaler Analysen** | Risiko von Informationsverlust an Chunk-Grenzen | Vollständig (Modell erfasst das Dokument im Ganzen) |

> **Industriestandard**: Der hybride Ansatz. RAG dient der Vorfilterung einer gigabytegroßen Wissensbasis auf einige zehntausend Token, die anschließend für tiefes Reasoning in den Kontext geladen werden.

---

### 2.3. Strategisches Denken (Thinking Mode) und Token-Budgetierung
Der Reasoning-Modus (Thinking / Reasoning Mode) generiert vor der Ausgabe eine interne Gedankenkette.

* **Wirtschaftlicher Faktor**: Thinking-Token werden als *Ausgabetoken* abgerechnet, die bei Anthropic **5-mal teurer als Eingabetoken** sind (bei anderen Providern gelten abweichende Verhältnisse).
* **Dynamisches Reasoning**:
  * Einfache semantische Anfragen («Wie teuer ist der Versand?») → *Thinking deaktiviert* (Günstiges Modell).
  * Komplexe Architekturanalysen oder anfängliche Aufgabenzerlegung → *Thinking aktiviert* in den ersten 1–2 Iterationen des Loops; anschließend Deaktivierung für routinemäßige Tool-Aufrufe *(siehe [Modul 3.3](#33-hybrides-routing-und-klassifikator-architektur))*.

---

## Modul 3. Ökonomie und Modellauswahl in der Produktion

### 3.1. Strategische Entscheidung: API vs. Self-Hosted (Open-Source)
Die Entscheidung zwischen API-Providern (Anthropic, OpenAI, Google) und eigenem Hosting von Open-Source-Modellen (Llama, Qwen, DeepSeek) wird von drei Dimensionen bestimmt:

```
                       [Compliance- und PII-Anforderungen]
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
            [Ja / Strikt]                             [Nein / Flexibel]
                  │                                         │
       [Cloud-Nutzung verboten?]                   [TCO-Kalkulation & Skalierung]
            ┌─────┴─────┐                             ┌─────┴─────┐
            ▼           ▼                             ▼           ▼
       [Self-Host]  [Enterprise API]             [Self-Host]    [API]
```

1. **Regulatorische Vorgaben (Data Privacy)**: Bei DSGVO-, HIPAA- oder strengen internen Datenschutzanforderungen ohne Option auf isolierte Enterprise-VPCs führt kein Weg an Self-Hosting vorbei.
2. **Lastprofil und Traffic-Muster**: Bei geringem bis mittlerem Volumen sind APIs meist günstiger, da keine Leerlaufkosten für reservierte GPUs anfallen.
3. **Bedarf an Fine-Tuning**: Werden spezialisierte Tool-Calling-Formate oder domänenspezifische Sprachen benötigt, spielt Open-Source seine Stärken aus.

---

### 3.2. Berechnung der Total Cost of Ownership (TCO)
Beim Kostenvergleich **darf nicht allein der API-Tokenpreis den GPU-Mietkosten gegenübergestellt werden**. Die reale TCO-Formel für Self-Hosting umfasst 4 Komponenten:

```
TCO(Self-Host) = Cost(GPU) + Cost(Infra) + Cost(DevOps/SRE) + Cost(Complexity)
```

* **Cost GPU**: Reine Hardware- bzw. Instanzmiete (A100/H100) oder Clusterkosten.
* **Cost Infra**: Netzwerk-Traffic, Storage für Modellgewichte und Vektordatenbanken.
* **Cost DevOps/SRE**: Arbeitsstunden hochqualifizierter Ingenieure für vLLM/TGI-Setup, Quantisierungsoptimierung und 99,9% SLA-Support. *Dieser Posten übersteigt die GPU-Miete häufig um das 5- bis 10-Fache.*
* **Cost Complexity**: Opportunitätskosten durch verzögerte Feature-Releases.

---

### 3.3. Hybrides Routing und Klassifikator-Architektur
Zur Kostenoptimierung bei gleichbleibender Qualität kommt eine **hybride Routing-Architektur** (Router LLM) zum Einsatz:

```
[Benutzeranfrage] ──► [Klassifikator (Small Model / Fast Classifier)]
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  [Einfache Anfrage]        [Mittlere Komplexität]    [Komplex / Kritisch]
   Kleines / lokales Modell      Mittleres Modell        Frontier-Modell
      (Kosten: 1x)              (Kosten: ~10x)          (Kosten: ~50x+)
```

1. **Klassifikator (Router)**: Ein schneller Klassifikator oder ein kleines Modell (≈1–3B Parameter) bewertet die Komplexität der Anfrage (ML-Klassifikator im Millisekundenbereich, kleines LLM in zig bis Hunderten ms). Bei Fehlern oder geringer Konfidenz greift die Fallback-Schicht: Feste Keyword-Regeln, danach Standardmodell (Kapitel 2.2).
2. **Routing-Tabellen**: Einfache CRUD-Aktionen und Standardfragen gehen an lokale/günstige Modelle. Mehrstufiges Reasoning und Code-Generierung an führende Frontier-Modelle.
3. **KV-Cache beim Routing berücksichtigen**: Modellwechsel innerhalb eines laufenden Dialogs invalidieren den KV-Cache des vorherigen Modells. Bei langen Verläufen kann das «günstige» Modell dadurch teurer werden als das Verbleiben auf dem bestehenden Modell (Kapitel 2.2).

---

## Modul 4. Harness Engineering: Entwurf der Ausführungsumgebung

### 4.1. Paradigmenwechsel: Vom Prompt Engineering zum Harness Engineering
> *«Die Leistungsfähigkeit eines Agenten wird weniger durch das Modell selbst bestimmt als durch den Code der Umgebung (Harness) um dieses Modell herum».*

Prompt Engineering als alleinige Disziplin reicht nicht mehr aus. An seine Stelle tritt **Harness Engineering** — der Entwurf robuster Ausführungsumgebungen, Verifikationsalgorithmen und automatischer Feedbackschleifen.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        HARNESS ENGINEERING LAYER                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │    Constrain     │  │      Inform      │  │        Verify        │  │
│  │ (Einschränken)   │  │ (Informieren)    │  │    (Verifizieren)    │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘  │
│           │                     │                       │              │
│           └─────────────────────┼───────────────────────┘              │
│                                 ▼                                      │
│                      ┌──────────────────────┐                          │
│                      │       Correct        │                          │
│                      │  (Korrigieren/Retry) │                          │
│                      └──────────────────────┘                          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼
                       [LLM / System-Engine]
```

---

### 4.2. Die 4 Steuerungsfunktionen (OpenAI Framework)
> *Kurszuordnung; die Struktur deckt sich mit der OpenAI-Veröffentlichung «Harness engineering» (2026).*

Der Harness eines Agenten erfüllt 4 zentrale Funktionen:

1. **Constrain (Einschränken)**: Programmatische Grenzen — strikte Token-Budgets, Allowlisten für Tools, isolierte Ausführungssandboxen.
2. **Inform (Informieren)**: Bereitstellung von Kontext, Werkzeugbeschreibungen und dynamischem Systemzustand.
3. **Verify (Verifizieren)**: Automatische Überprüfung der Arbeitsergebnisse durch Invarianten, Linter und Test-Suites.
4. **Correct (Korrigieren)**: Automatische Formulierung konstruktiver Fehlerhinweise und Rückführung des Agenten in eine Korrekturschleife.

---

### 4.3. Die 7 Schichten des System-Harness (Harness Architecture)
Ein moderner Produktions-Harness gliedert sich in 7 funktionale Schichten:

1. **Execution Sandbox (Ausführungskern)**: Isolierte Umgebung (Docker, gVisor, WASM) zur Ausführung von Code und Shell-Befehlen.
2. **Tooling & Protocols (Werkzeugschicht)**: Validierung von Schemata (JSON Schema, MCP Protocol), Timeout-Handling und Fehlerharmonisierung.
3. **Context Manager (Kontextverwaltung)**: Prompt-Assemblierung, Entfernen veralteter Daten, Verdichtungsstrategien (Compaction).
4. **Lifecycle & State Machine (Zustandsverwaltung)**: Orchestrierung von Zuständen, Unterbrechungsbehandlung, Sitzungswiederherstellung (Recovery).
5. **Observability Layer (Beobachtbarkeit)**: Durchgängiges Tracing, Metriken, Schritt- und Aufruf-Logging *(siehe Modul 7.3)*.
6. **Verification Layer (Prüfschicht)**: Invarianten im Code, LLM-as-a-Judge-Instanzen, Qualitätsprüfungen.
7. **Governance & Security (Governance und Sicherheit)**: Schutz vor Prompt Injections, Rechtestrukturierung, Tagesbudget-Limits.

---

### 4.4. Dreistufige Verifikation von Invarianten
Die Verifikation von Ergebnissen erfolgt in einer nach Latenz und Kosten gestaffelten Hierarchie:

```
Level 1: Invariants in Code (0$ / 0ms) ──► Sofortige Prüfung durch deterministische Regeln
Level 2: Benchmark & Evaluation Runs  ──► Periodische Testläufe auf Referenzdatensätzen
Level 3: Human-in-the-Loop Validation ──► Menschliche Freigabe risikoreicher Aktionen
```

* **Level 1: Invarianten im Code (Invariants in Code)**: Unmittelbare, kostenlose Validierung. *Beispiel*: Ein Erstattungsbetrag darf die Bestellsumme nicht übersteigen; ein Releasedatum darf nicht in der Vergangenheit liegen.
* **Level 2: Referenzmessungen (Eval Runs)**: Automatisierte Regressionstests bei Modellwechseln oder Prompt-Updates.
* **Level 3: Menschliche Kontrolle (Human-in-the-Loop)**: Bestätigung kritischer Aktionen.

---

## Modul 5. Multi-Agenten-Systeme und Architekturmuster

### 5.1. Kognitive Überlastung eines einzelnen Modells
Wird eine komplexe Gesamtaufgabe («Führe ein Code-Audit durch, finde Fehler, refaktoriere Module, aktualisiere die Dokumentation und erstelle einen Report») an ein einzelnes Modell übergeben, kommt es zur **kognitiven Überlastung** (Cognitive Overload).
* Das Modell verliert den Fokus über Dutzende Instruktionen hinweg (*Loss of Focus*).
* Die Fehlerwahrscheinlichkeit bei der Werkzeugauswahl steigt rapide (ab ca. 20 Werkzeugen sinkt die Trefferquote signifikant).

*Lösung*: Arbeitsteilung und Spezialisierung — Aufbau eines Multi-Agenten-Teams, in dem jeder Agent einen fokussierten Kontext und ein abgegrenztes Werkzeugset besitzt.

---

### 5.2. 7 Kernmuster der Multi-Agenten-Interaktion
In der Praxis haben sich 7 fundamentale Entwurfsmuster etabliert:

```
1. Pipeline        : [A1] ──► [A2] ──► [A3]
2. Router          : [Router] ──► [A_fast] ODER [A_expert]
3. Parallelization : [A_split] ──► [A1] + [A2] + [A3] ──► [Aggregator]
4. Reflection      : [Generator] ◄──► [Critic / Reflector]
5. Tool Use        : [Agent] ──► [Orchestrator] ──► [Externe APIs]
6. Planning        : [Planner] ──► [Task Graph] ──► [Workers]
7. Collaboration   : [Rolle A] ◄──► [Rolle B] ◄──► [Rolle C] (Gemeinsame Umgebung)
```

1. **Pipeline Pattern (Sequenz)**: Schrittweise Übergabe der Ausgabe eines Agenten an den nächsten. *Einsatz*: Dokumentenverarbeitung.
2. **Router Pattern (Verteiler)**: Ein primärer Agent klassifiziert die Aufgabe und delegiert an spezialisierte Agenten.
3. **Parallelization Pattern (Parallelisierung)**: Zerlegung in unabhängige Teilaufgaben, gleichzeitige Bearbeitung und finale Aggregation. *Einsatz*: Datenanalysen.
4. **Reflection Pattern (Reflexion)**: Duo aus «Generator» und «Kritiker». Der Kritiker prüft die Ausgabe anhand definierter Kriterien und veranlasst Nachbesserungen.
5. **Tool Use Pattern (Werkzeugnutzung)**: Orchestrierung externer Schnittstellen mit Rückgabewert-Validierung.
6. **Planning Pattern (Planung)**: Ein Planer erstellt einen dynamischen Aufgabengraphen (Task Graph), der von Workern abgearbeitet wird.
7. **Collaboration Pattern (Kollaboration)**: Ein Team spezialisierter Rollen (z. B. Entwickler, Tester, Produktmanager) interagiert in einem geteilten Kontext.

---

### 5.3. Overheads, Anti-Patterns und das Prinzip progressiver Komplexität
Multi-Agenten-Systeme bringen erhebliche Zusatzkosten mit sich:
* **Token-Explosion**: Nach Messungen von Anthropic verbraucht ein Multi-Agenten-Forschungssystem **~15× mehr Token als ein einfacher Chat** (ein Einzelagent ca. ~4×), also etwa das 3- bis 4-Fache eines Einzelagenten.
* **Latenzanstieg**: Netzwerkaufrufe und wiederholte Abstimmungsrunden verlängern die Antwortzeiten erheblich.

#### Anti-Patterns:
* Einsatz von Multi-Agenten-Teams für simple FAQs oder lineare Abläufe.
* Übermäßige Koordinationsschleifen in Echtzeitsystemen.

> **Prinzip progressiver Komplexität**: *Beginnen Sie mit einem einzelnen Agenten -> Fügen Sie Reflexion hinzu -> Implementieren Sie Routing -> Gehen Sie erst dann zu Parallelisierung und Planung über, wenn die einfacheren Stufen nachweislich nicht ausreichen.*

---

## Modul 6. Langzeitgedächtnis und Wissensmanagement

### 6.1. Architekturgabelung: Synthese beim Schreiben vs. Synthese beim Lesen
Beim Design von Wissensspeichern für KI-Agenten existiert eine grundlegende Weichenstellung:

```
                    [EINGEHENDE DATEN / QUELLEN]
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[Synthese beim Schreiben (Write-Time)]         [Synthese beim Lesen (Read-Time)]
- Fakten werden SOFORT verdichtet.             - Daten werden ROH gespeichert.
- Schreiben teuer, Lesen günstig.              - Schreiben kostenlos, Lesen teuer.
- RISIKO: Verlustbehaftete Kompression.        - VORTEIL: Vollständige Originaltreue.
```

---

### 6.2. Schreibdisziplin: Append-Only-Logs und Provenienz-Anker
Die häufigste Ursache für Datenkorruption in Multi-Agenten-Speichern sind konkurrierende Schreibzugriffe (*Lost Updates* / *Write Collisions*).

#### Produktionsstandards für Schreibdisziplin:
1. **Append-Only Log (Reines Anhängen)**: Kein Agent darf bestehende Einträge überschreiben oder löschen. Neue Informationen werden ausschließlich am Ende angefügt.
2. **Provenienz-Anker (Source Anchoring)**: Jeder Speichereintrag muss auf seine unveränderliche Primärquelle verweisen (Transkript-ID, Git-Commit-Hash, Dokument-URL, Zeitstempel).

---

### 6.3. Asynchroner Schiedsrichter-Agent (WikiGroomer) und Concept Drift
Mit der Zeit veralten Fakten in der Wissensbasis und widersprechen einander (Concept Drift). Für die Speicherhygiene sorgt ein dedizierter Hintergrundprozess — der **WikiGroomer**:

#### Aufgaben des WikiGroomer-Agenten:
1. **Konsolidierung und Deduplizierung**: Regelmäßiges Durchforsten der Rohdaten und Erstellen bereinigter Zusammenfassungen.
2. **Widerspruchserkennung**: Auffinden inkonsistenter Fakten (z. B. «Wir nutzen GitFlow» vs. «Umstieg auf Trunk-Based Development erfolgt») und Eskalation an das Team.
3. **Gültigkeitsversionierung**: Veraltete Einträge werden nicht gelöscht, sondern mit einem Gültigkeitsende versehen (`valid_to`).

---

### 6.4. Speichervergiftung (Memory Poisoning) und Prompt Injection
Ein geteilter Speicher stellt eine Angriffsfläche dar. Liest ein Agent eine manipulierte Website oder ein präpariertes Dokument (indirekte Prompt Injection), kann er bösartige Befehle in den gemeinsamen Speicher schreiben.

#### Das Schutzprinzip «Zweier-Regel» (Meta, Agents Rule of Two) und die «tödliche Triade» (S. Willison):
Gefahr entsteht beim gleichzeitigen Zusammentreffen von 3 Bedingungen:  
1. Zugriff auf sensible/private Daten.  
2. Nicht vertrauenswürdiger Eingabestrom.  
3. Ein Kanal zur Ausgabe von Informationen nach außen.

> **Schutz**: Ein Agent darf innerhalb einer Sitzung maximal **zwei** dieser drei Eigenschaften besitzen. Unterbrechen Sie die Kette, indem Sie bei Kontakt mit unmoderiertem Speicher externe Ausgabekanäle sperren (Human-in-the-Loop) *(siehe [Modul 7.1](#71-mehrschichtige-sicherheit-defense-in-depth-und-übermäßige-autonomie))*.

---

## Modul 7. Produktivbetrieb: Sicherheit, Resilienz und Observability

### 7.1. Mehrschichtige Sicherheit (Defense-in-Depth) und übermäßige Autonomie
Angriffe auf KI-Agenten gehen weit über klassische Web-Schwachstellen hinaus:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DEFENSE-IN-DEPTH ARCHITECTURE                      │
│                                                                         │
│  [Eingabe] ──► 1. Input Security Gate (Guard Models)                    │
│                     │                                                   │
│                     ▼                                                   │
│                2. Content & RAG Integrity Verification (Hashes & Evals) │
│                     │                                                   │
│                     ▼                                                   │
│                3. Isolated Execution Sandbox (z. B. gVisor / Docker)    │
│                     │                                                   │
│                     ▼                                                   │
│                4. Output Validation & Human Gate ──► [Ausgabe]          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Das Risiko übermäßiger Autonomie (Excessive Agency / Confused Deputy):
Der Agent verfügt über Rechte zur Ausführung mächtiger Befehle, verliert aber im Laufe des Dialogs das Verständnis für seine Sicherheitsgrenzen.

#### Schutzstrategie:
* **Rechteminimierung (Least Privilege)**: Zuteilung kurzlebiger Micro-Tokens mit minimal erforderlichen Rechten.
* **Programmatischer Proxy**: Die Gefährlichkeit einer Tool-Ausführung wird durch externen Proxy-Code bewertet, nicht durch Prompt-Instruktionen.

---

### 7.2. Ausfallsicherheit (Resilience): Circuit Breakers, Idempotenz und Fehlerstrategie
Ein Produktionsagent muss auch bei Störungen externer LLM-Provider funktionsfähig bleiben:

1. **Circuit Breaker**: Steigt die Fehlerrate (5xx/429) über einen Schwellenwert, schaltet das System automatisch auf lokale Fallbacks oder Ausweichregionen um.
2. **Idempotenzschlüssel (Idempotency Keys)**: Jeder schreibende oder finanzielle Werkzeugaufruf muss einen eindeutigen Schlüssel (`idempotency_key`) mitführen, um Doppelbuchungen bei Verbindungsabbrüchen zu verhindern.
3. **Strategien der Fehlerbehandlung**:
   * **Fail Closed**: Abbruch der Operation bei Störungen (z. B. Finanzbuchungen).
   * **Fail Open**: Fortführung mit Warnhinweis und Logging (unkritische Empfehlungssysteme).
   * **Graceful Degradation**: Rückfall auf deterministische Algorithmen oder Vorlagen bei LLM-Ausfall.

---

### 7.3. Durchgängige Observability und 5 kritische Agenten-Metriken
Klassisches Monitoring (HTTP 200/500) greift bei Agenten zu kurz: Ein System kann HTTP 200 liefern, während es in einer Endlosschleife das Budget verbrennt.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      5 CRITICAL AGENTIC METRICS                         │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. Tool Call Success Rate (granular pro Werkzeug)                       │
│ 2. Semantic Loop Distance (Abstand im Embedding-Raum)                   │
│ 3. Context Utilization & Compaction Frequency                           │
│ 4. Tool Selection Accuracy (Treffsicherheit der Werkzeugwahl)           │
│ 5. Total Cost per Task Execution (TCO pro Geschäftsaufgabe)             │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Tool Call Success Rate**: Anteil erfolgreicher Werkzeugaufrufe, aufgeschlüsselt nach einzelnen Tools.
2. **Semantic Loop Distance**: Distanz im Embedding-Raum zwischen aufeinanderfolgenden Tool-Aufrufen. Unterschreitet der Wert eine Schwelle, befindet sich der Agent in einer Halluzinationsschleife.
3. **Context Utilization & Compaction Frequency**: Häufigkeit von Kontextverdichtungen. Hohe Frequenzen deuten auf das Risiko hin, dass wichtige Systemanweisungen verdrängt werden.
4. **Tool Selection Accuracy**: Relevanz der gewählten Werkzeuge bezogen auf die konkrete Aufgabenstellung.
5. **Total Cost per Task Execution**: Kumulierte Kosten aller Iterationen, Token und API-Calls im Rahmen eines Vorgangs.

---

### 7.4. Fazit: Deployment-Readiness-Checkliste
Vor dem Release eines KI-Agenten in die Produktion muss folgende Checkliste erfüllt sein:

- [ ] Iterations- und Token-Limits sind **im Code** verankert, nicht im Prompt.
- [ ] Alle irreversiblen Aktionen mit hohem Risiko sind durch **Human-in-the-Loop** abgesichert.
- [ ] Alle Tools mit Seiteneffekten (Zahlungen, Schreibzugriffe, Versand) akzeptieren **Idempotenzschlüssel**.
- [ ] Sitzungstracing ist aktiv und erfasst die Metrik **Cost per Task**.
- [ ] Ein **Circuit Breaker** mit Exponential Backoff und Jitter ist für alle externen APIs implementiert.
- [ ] Der System-Prompt ist versioniert und durch Eval-Tests abgedeckt; die KV Cache Hit Rate wird überwacht.
- [ ] Wissensspeicher arbeiten im **Append-Only**-Modus mit Quellenreferenzen.

---

## Quellen

1. D. Beresnitski. Videoreihe «System Design eines KI-Agenten von innen». Teil 1: [Anatomy of an AI Agent for Seniors: Pipeline, Cycle, Tools, Traps](https://www.youtube.com/watch?v=rN4_Y67Tr8I) (2026).
2. Anthropic. [Building effective agents](https://www.anthropic.com/research/building-effective-agents) (2024).
3. Anthropic. [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) (2025).
4. A. Grigorev. [How I Dropped Our Production Database and Now Pay 10% More for AWS](https://alexeyondata.substack.com/p/how-i-dropped-our-production-database) (2026).
5. Gartner: Mehr als 40% der agentischen KI-Projekte vor dem Aus bis Ende 2027 (Juni 2025), [Bericht](https://w.media/over-40-percent-of-agentic-ai-projects-could-face-the-axe-by-end-of-2027-gartner/).
6. Manus. Context Engineering for AI Agents: Lessons from Building Manus (2025).
7. T. Sumers et al. Cognitive Architectures for Language Agents (CoALA), arXiv:2309.02427 (2023).
8. S. Willison. [New prompt injection papers: Agents Rule of Two…](https://simonwillison.net/2025/Nov/2/new-prompt-injection-papers/) (2025).

---
*Dokument erstellt auf Grundlage des Kurses «Anatomie eines KI-Agenten» von Dmitri Beresnitski.*
