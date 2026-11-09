# Kapitel 1. Anatomie eines KI-Agenten: Pipeline, Loop, Tools, Fallstricke

> Grundlage: Video #1 der Reihe «System Design eines KI-Agenten von innen» (D. Beresnitski), ergänzt und anhand von Primärquellen verifiziert.
> Code zum Kapitel: [`code/ch01/`](../code/ch01/) — [`agent_loop.py`](../code/ch01/agent_loop.py), [`pipeline.py`](../code/ch01/pipeline.py), [`test_ch01.py`](../code/ch01/test_ch01.py) (ausschließlich Python-Standardbibliothek, Ausführung: `python -m unittest test_ch01.py`).
>
> Die Bezeichnungen von Funktionen und Klassen im Text sind anklickbar: Der Link öffnet auf GitHub den jeweiligen Codeabschnitt mit hervorgehobenen Zeilen.

## Lernziele des Kapitels

Nach diesem Kapitel können Sie:

1. Die minimal notwendige Komplexitätsstufe eines Systems wählen — von einem einzelnen LLM-Aufruf bis hin zum Multi-Agenten-System — und diese Wahl fundiert begründen.
2. Einen System-Prompt als versionierten, viergliedrigen Verhaltensvertrag formulieren.
3. Eine Anfrage-Pipeline nach dem Prinzip «früh blockieren, spät ausgeben» entwerfen.
4. Den Kontext so zusammenstellen, dass der KV-Cache intakt bleibt.
5. Einen Agent Loop mit Schutzbarrieren im Code statt im Prompt programmieren.
6. Werkzeuge (Tools) so gestalten, dass das Modell Fehler versteht und selbstständig korrigieren kann.
7. Observability ab dem ersten Tag im System verankern.

## Durchgängiges Beispiel: Barista Agent

Durch das gesamte Kapitel (und das gesamte Buch) zieht sich ein zentrales Beispiel: Der Agent eines Cafés. Er beantwortet Fragen zum Menü, erinnert sich an Vorlieben der Gäste, nimmt Bestellungen auf, wickelt Zahlungen ab und prüft den Bestellstatus.

| Werkzeug | Funktion | Reversibel? |
|---|---|---|
| `get_menu` | Menü und Preise abrufen | Ja (reiner Lesezugriff) |
| `get_user_preferences` | Gästepräferenzen («Hafermilch») | Ja |
| `create_order` | Bestellung aufgeben und bezahlen | **Nein** → Menschliche Freigabe erforderlich |
| `check_order_status` | Status der Bestellung abfragen | Ja |

Das Beispiel ist kompakt, enthält jedoch alles, was in Produktivsystemen fehlschlagen kann: Zahlungsflüsse, externe Dienste, personenbezogene Daten und irreversible Operationen.

---

## 1.1. 5-Minuten-Prototyp vs. Produktionsrealität

Mit LangChain, AutoGen oder CrewAI lässt sich ein Agent in wenigen Zeilen Code zusammenstellen. In der Entwicklungsumgebung funktioniert er. In der Produktionsumgebung wird dasselbe System mit drei gravierenden Problemen konfrontiert: **unkontrollierten Kosten, Endlosschleifen und unvorhersehbarem Verhalten**.

Praxisfälle aus dem Video:

- **Rechnungsschock statt Prototyp:** Ein Team erhielt eine Monatsrechnung von rund 2,5 Millionen Dollar, nachdem der Prototyp lediglich 50 Dollar gekostet hatte *(Zahl aus dem Video, Primärquelle nicht genannt)*.
- **Endlosschleife:** Ein Bot wiederholte 58 Mal hintereinander exakt dieselbe Antwort und ignorierte sämtliche Stopp-Befehle *(aus dem Video, Primärquelle nicht genannt)*.
- **Löschung der Produktionsinfrastruktur:** Bei DataTalks.Club führte der Claude-Code-Agent `terraform destroy` aus und vernichtete die gesamte Produktionsumgebung: RDS-Datenbank, VPC, ECS-Cluster und Load Balancer — mitsamt den automatischen Snapshots. In der Datenbank befanden sich studentische Arbeiten aus ca. 2,5 Jahren. Die Ursache lag nicht in einem «Missverständnis» des Modells, sondern in einer Kette fataler Engineering-Entscheidungen: Die Terraform-State-Datei war nicht auf den neuen Rechner übertragen worden, der Agent entpackte ein veraltetes State-Archiv und der menschliche Operator brach das `destroy` nicht ab. Erst nach 24 Stunden gelang die Wiederherstellung über den AWS-Support, der einen verborgenen Snapshot fand. Konsequenz: Der Agent darf kein Terraform mehr ausführen, destruktive Aktionen führt der Mensch manuell aus, Löschsperren wurden aktiviert und Backups werden täglich automatisiert getestet. [3]

Gartner-Prognose (Juni 2025): **Mehr als 40% der agentischen KI-Projekte werden bis Ende 2027 eingestellt** — wegen explodierender Kosten, unklarem Geschäftswert und unzureichendem Risikomanagement. [2]

> **Hauptthese des Kapitels:** Das Problem liegt nicht im Modell — moderne LLMs sind hochgradig leistungsfähig. Das Problem liegt im Engineering drumherum. Erfolgreiche Systeme entstehen nicht durch «magische» Frameworks, sondern durch einfache, komponierbare Entwurfsmuster und strikten, deterministischen Code. [1]

Diese Engineering-Schicht um das Modell — Pipeline, Barrieren, Tools, Prüfungen, Observability — bezeichnen wir fortan als **Harness (Ausführungsumgebung/Objekt обвязка)**. Kapitel 1 etabliert das Grundgerüst; Kapitel 4 vertieft die Details.

---

## 1.2. Workflow oder Agent: Die Komplexitätsleiter

Anthropic unterteilt agentische Systeme in der Publikation «Building effective agents» in zwei Grundtypen [1]:

- **Workflow («Feste Schienen»)** — Der Entwickler legt die Schrittfolge deterministisch im Code fest. Das LLM führt den jeweiligen Einzelschritt aus, entscheidet jedoch **nicht über den nächsten Schritt**. Maximales Maß an Kontrolle und Vorhersehbarkeit, minimale Kosten.
- **Agent («Autonomie»)** — Das Modell entscheidet dynamisch, welches Werkzeug es aufruft und welcher Schritt folgt. Maximale Flexibilität bei hoher Ungewissheit — erkauft durch höhere Latenz, Kosten und Fehlerrisiken.

Hierbei handelt es sich nicht um eine Schwarz-Weiß-Entscheidung, sondern um eine **siebenstufige Komplexitätsleiter**. Eine höhere Stufe sollte erst dann erklommen werden, wenn die vorherige nachweislich versagt.

| # | Stufe | Funktionsweise | Beispiel im Barista Agent |
|---|---|---|---|
| 1 | **Single LLM Call** | Eine Anfrage — eine Antwort. Wenn ausreichend: keinesfalls verkomplizieren | «Beschreibe den Geschmack eines Flat White» |
| 2 | **Prompt Chaining** | Ausgabe von Schritt N ist Eingabe von Schritt N+1. Dazwischen deterministische Prüfungen; hier wird Human-in-the-Loop eingebunden | Bestellung parsen → mit Menü abgleichen → Gast zur Freigabe vorlegen |
| 3 | **Routing** | Schneller, günstiger Klassifikator leitet die Anfrage in eine spezialisierte Kette | «Frage zum Menü» / «Bestellung» / «Reklamation» → getrennte Pfade |
| 4 | **Parallelization** | Unabhängige Teilaufgaben werden parallel ausgeführt | Parallel: Menü laden, Gastpräferenzen abrufen, Auslastung prüfen |
| 5 | **Orchestrator–Workers** | LLM-Orchestrator zerlegt die Aufgabe und delegiert an fokussierte Worker | Bürobestellung über 15 Getränke: Orchestrator teilt nach Getränketyp auf |
| 6 | **ReAct Agent** | Modell erhält ein Ziel, plant autonom, wählt Werkzeuge und iteriert in einer Schleife bis zum Resultat | «Finde etwas Süßes ohne Koffein unter 300 ₽ und bestelle es» |
| 7 | **Multi-Agent System** | Mehrere Agenten kooperieren als autonomes Team | Ein Agent führt den Dialog, ein zweiter verwaltet das Lager, ein dritter den Einkauf |

Die Stufen 1–5 sind Workflows (der Code steuert die Sequenz). Die Stufen 6–7 sind Agenten (das Modell wählt die Sequenz).

> **Abgleich mit der Quelle:** Die Leiter ist das didaktische Schema des Videos. Bei Anthropic [1] werden die Stufen 2–5 als Workflow-Muster beschrieben, ergänzt um **Evaluator–Optimizer**: Ein Modell generiert, ein zweites prüft und gibt Feedback, die Schleife wiederholt sich bis zur Zielqualität (z. B. Prüfung der Bestellung gegen Budget und Menü). Dieser Ansatz wird in Kapitel 5 vertieft.

> **Regel:** Erhöhen Sie die Autonomie nur dann, wenn ein deterministischer Workflow der Aufgabenvarianz nicht mehr gewachsen ist. Die überwiegende Mehrheit produktiver Systeme arbeitet auf den Stufen 2–4.

### Checkliste zur Stufenwahl

1. Lässt sich die Aufgabe mit einem einzigen Call und präzisem Prompt lösen? → Stufe 1.
2. Sind alle Teilschritte im Voraus bekannt? → Stufe 2.
3. Lassen sich die Anfragen in bekannte Kategorien einteilen? → Stufe 3.
4. Gibt es voneinander unabhängige Teilaufgaben und zählt jede Millisekunde? → Stufe 4.
5. Sind die Teilaufgaben dynamisch, aber klar zerlegbar? → Stufe 5.
6. Ist der Pfad zum Ziel unbekannt und hängt von Zwischenergebnissen ab? → Stufe 6.
7. Werden unterschiedliche Rollen mit separatem Kontext benötigt und ist ein Einzelagent überlastet? → Stufe 7 (Details in Kapitel 5).

---

## 1.3. Der System-Prompt als Vertrag

Ein System-Prompt ist kein informeller «Text für die KI», sondern ein **architektonischer Verhaltensvertrag**. Er wird im Git-Repository versioniert und wie Programmcode durch automatisierte Tests abgedeckt.

### Vier obligatorische Abschnitte

1. **Rolle und explizite Grenzen:** Was der Agent tut und was er strikt verweigert. Ohne negative Abgrenzung neigen Modelle dazu, auf beliebige Off-Topic-Themen einzugehen.
2. **Priorisierte Regeln:** Nummerierte Liste. Bei Zielkonflikten muss die Hierarchie feststehen: **Sicherheit > Budget > Nutzerkomfort**.
3. **Antwortformat:** Strikte Syntax: JSON-Schema, Markdown etc.
4. **Beispiel (Few-Shot):** Ein konkreter, repräsentativer Beispieldialog.

### Beispiel für den Barista Agent

```text
# Rolle
Du bist der Barista-Assistent des Cafés «Kaffeebohne». Du berätst bei der Auswahl,
beantwortest Fragen zur Speisekarte und nimmst Bestellungen entgegen.
Du diskutierst KEINE Themen außerhalb des Café-Betriebs, gibst KEINE medizinischen
Ratschläge (bei Allergien nennst du rein die Zutaten laut Menü) und versprichst KEINE Rabatte.

# Regeln (bei Konflikten hat die Regel mit der niedrigeren Nummer Vorrang)
1. Zeige dem Gast vor create_order immer die Gesamtauswahl und den Preis zur Bestätigung.
2. Preise und Zutaten stammen ausschließlich aus get_menu, niemals aus dem Gedächtnis.
3. Ist ein Artikel nicht verfügbar, schlage die passendste Alternative vor.
4. Antworte prägnant: 1–3 Sätze.

# Format
Antworte in JSON: {"reply": "<Text für den Gast>", "suggested_items": ["<id>"]}

# Beispiel
Gast: Habt ihr etwas ohne Koffein?
Antwort: {"reply": "Ja: heißen Kakao und Kräutertee. Kakao kostet 220 ₽.", "suggested_items": ["cocoa", "herbal_tea"]}
```

> **Hinweis zum Stil:** In *Werkzeugbeschreibungen* funktionieren positive Formulierungen am besten (Abschnitt 1.7). Im System-Prompt sind negative Rollengrenzen jedoch unverzichtbar — sie sollten knapp, präzise und unmissverständlich sein.

**Best Practices für Verträge:**

- Die Prompt-Datei liegt im Repository (`prompts/barista.v3.md`); die Versionskennung wird bei jedem Request geloggt.
- Eine standardisierte Testsuite (Eval-Set) aus 20–50 typischen sowie bösartigen Anfragen wird bei jeder Prompt-Änderung in der CI ausgeführt.
- Im System-Prompt befinden sich **keine dynamischen Daten** (Uhrzeit, Benutzername) — diese würden den KV-Cache invalidieren (Abschnitt 1.5).

---

## 1.4. Pipeline-Skelett: «Früh blockieren, spät ausgeben»

Die Pipeline ist das Fließband zur Kontextaufbereitung vor dem Modellaufruf und zur Resultatsprüfung danach. Leitprinzip: **Günstige Prüfungen zuerst, teure LLM-Aufrufe so spät wie möglich**. Jede frühzeitig abgewiesene Anfrage verursacht null Token-Kosten.

```
Anfrage
  │
  1. Validierung ($0) ─────────────────► Abweisung: leer / Binärmüll / zu lang
  2. Eingangssicherheit ───────────────► Abweisung: Prompt Injection
  3. Kontextanreicherung                 (Kalender, aktive Aufgaben, Status)
  4. Gedächtnis & Retrieval              (Präferenzen, RAG aus Wissensdatenbank)
  5. Inhaltsfilterung ─────────────────► Bereinigung: beschädigte Docs, Leaks, indirekte Injections
  6. Kontextassemblierung & Budget     (Greedy-Cut / Zusammenfassung / Compaction)
  7. Routing & Modus                   (günstiges/teures Modell, Thinking Mode)
  8. Generierung                        (LLM-Aufruf oder Agent Loop — Abschnitt 1.6)
  9. Ausgangssicherheit ───────────────► Sperre: PII-Leak oder System-Prompt-Leak
 10. Formatierung & Validierung ───────► Retry / Fehler: ungültiges JSON
  │
Antwort
```

| # | Stufe | Aufgabe | Begründung | Code |
|---|---|---|---|---|
| 1 | Validation | Leere Strings, Binärzeichen, Maximallänge, Sprache | Kostenlos. Filtert laut Video ca. 5% des Müll-Traffics | [`Validate`](../code/ch01/pipeline.py#L53-L66 "code:Validate") |
| 2 | Input Security | Prüfung auf direkte Prompt Injections | Vor teuren RAG- und Modellaufrufen | [`InputSecurity`](../code/ch01/pipeline.py#L82-L87 "code:InputSecurity") |
| 3 | Context Enrichment | Anreicherung um externe Daten | Modell weiß nicht, wer anfragt oder welcher Tag ist | [`Enrich`](../code/ch01/pipeline.py#L92-L98 "code:Enrich") |
| 4 | Memory & Retrieval | Benutzerhistorie + RAG über FAQ und Dokumente | Details in Kapitel 2 und 6 | [`Retrieve`](../code/ch01/pipeline.py#L101-L108 "code:Retrieve") |
| 5 | Content Filtering | Bereinigung gefundener Daten: Leaks, **Injections in Dokumenten** | Gefundene Texte sind nicht vertrauenswürdige Eingaben | [`ContentFilter`](../code/ch01/pipeline.py#L113-L119 "code:ContentFilter") |
| 6 | Context Assembly | System-Prompt + RAG + Memory + Verlauf im Token-Budget | Strategien: Greedy, Compaction, Summary | [`Assemble`](../code/ch01/pipeline.py#L124-L154 "code:Assemble") |
| 7 | Routing & Mode | Modell- und Reasoning-Auswahl | Details in Kapitel 3 | [`Route`](../code/ch01/pipeline.py#L159-L167 "code:Route") |
| 8 | Generation | LLM-Call oder Agent Loop | Die einzige teure Komponente | [`Generate`](../code/ch01/pipeline.py#L172-L178 "code:Generate") |
| 9 | Output Security | Erkennung von PII und System-Prompt-Leaks (Canary) | Modell-Outputs sind ebenfalls unvertrauenswürdig | [`OutputSecurity`](../code/ch01/pipeline.py#L186-L196 "code:OutputSecurity") |
| 10 | Formatting & Validation | JSON-Validierung gegen Schema | Ungültiges JSON zerstört Client-Applikationen | [`ValidateOutput`](../code/ch01/pipeline.py#L201-L214 "code:ValidateOutput") |

> **Stufenanzahl: 6, 10 oder 12?** In der Videobeschreibung werden 12 Stufen genannt, im Vortrag 10 erläutert und im initialen Entwurf 6 aggregiert. Die exakte Zahl ist sekundär: Das Prinzip bleibt identisch. Entscheidend ist die Abfolge «günstig → teuer» und die Möglichkeit jeder Stufe, die Kette sofort abzubrechen.

### Wichtige Details zur Pipeline

- **Indirekte Prompt Injections** stammen nicht vom Endnutzer, sondern aus Datenbanken, externen Webseiten, PDFs oder **Tool-Rückgabewerten**. Stufe 2 erkennt diese nicht — daher ist Stufe 5 zwingend erforderlich ([`ContentFilter`](../code/ch01/pipeline.py#L113-L119 "code:ContentFilter"), Test [`test_indirect_injection_in_docs_filtered`](../code/ch01/test_ch01.py#L133-L136 "code:PipelineTests.test_indirect_injection_in_docs_filtered")). Auch im Agent Loop müssen Tool-Rückgaben validiert werden.
- **Canary-Token** ([`OutputSecurity`](../code/ch01/pipeline.py#L186-L196 "code:OutputSecurity")): Ein geheimer, zufälliger String im System-Prompt (z. B. `CANARY-7f3a`). Erscheint dieser im Output, liegt ein Prompt-Leak vor und die Antwort wird sofort blockiert.
- **Fehlerhaftes JSON:** In der Praxis werden 1–2 automatisierte Retries mit dem Validierungsfehler ausgeführt, bevor eine Fehlermeldung an den Nutzer geht. Viele Provider bieten zudem Structured Outputs an.
- **Aussagekräftige Abweisungen:** Jeder Abbruch liefert dem Anwender eine verständliche Erklärung und schreibt einen maschinenlesbaren Fehlercode ins Log.

### Code-Umsetzung

In `pipeline.py` implementiert jede Stufe die Methode `run(ctx)`, reichert den zentralen [`RequestContext`](../code/ch01/pipeline.py#L19-L33 "code:RequestContext") an oder wirft eine [`StageReject`](../code/ch01/pipeline.py#L36-L39 "code:StageReject"). Die Funktion [`run_pipeline`](../code/ch01/pipeline.py#L219-L230 "code:run_pipeline") führt die Kette sequentiell aus und erfasst Trace-Daten. Der Test [`test_block_early_spends_nothing`](../code/ch01/test_ch01.py#L125-L131 "code:PipelineTests.test_block_early_spends_nothing") verifiziert, dass bei Injections kein teurer Generierungsaufruf erfolgt.

```python
stages = [Validate(), InputSecurity(), Enrich(...), Retrieve(...), ContentFilter(),
          Assemble(SYSTEM, budget_tokens=3000), Route(...), Generate(...),
          OutputSecurity(SYSTEM, canary="CANARY-7f3a"), ValidateOutput({"reply": str})]
ctx = run_pipeline(stages, RequestContext(user_id="u1", query="Wie viel kostet ein Latte?"))
```

---

## 1.5. Ökonomie des Cachings: KV-Cache

In einem Agent Loop liest das Modell **in jeder Iteration den gesamten bisherigen Kontext erneut ein**, während die Ausgabe oft nur aus einem Funktionsaufruf oder wenigen Zeilen besteht. Das Manus-Team beziffert das Verhältnis von Eingabe- zu Ausgabetoken in seinem Agenten auf rund **100 : 1** [4]. Das bedeutet: Die Betriebskosten werden primär durch Eingabetoken bestimmt.

Der **KV-Cache** (Prompt Caching) verhindert das wiederholte Neuberechnen bereits verarbeiteter Präfixe. Bleibt das Präfix unverändert, sind gecachte Token rund 90% günstiger als reguläre Eingabetoken (bei Anthropic kostet der Cache-Read nur 10% des Basispreises) und werden deutlich schneller verarbeitet. Der Cache greift jedoch nur bei **exakter Übereinstimmung des Präfixes**: Ändert sich auch nur ein einziges Token, verfallen alle nachfolgenden gecachten Daten.

### Typische KV-Cache-Fallen

| Fehler | Warum der Cache bricht | Korrekte Lösung |
|---|---|---|
| Zeitstempel (`timestamp`) am Anfang des System-Prompts | Präfix ändert sich im Sekundentakt | Zeitangaben ans Ende des Kontexts in den Dynamik-Block setzen |
| Änderungen in der Mitte der Historie (Kürzungen früherer Nachrichten) | Alles hinter der modifizierten Stelle muss neu berechnet werden | Verlauf strikt **nur am Ende erweitern (append-only)** |
| Instabile JSON-Schlüsselreihenfolge | `{"a":1,"b":2}` ≠ `{"b":2,"a":1}` für den Cache | Serialisierung immer mit `sort_keys=True` durchführen |

> **Hauptregel:** Daten im Loop werden ausschließlich am Ende angefügt. Statische Blöcke an den Anfang, dynamische Blöcke ans Ende.

### Warum Kosten schneller wachsen als die Dialoglänge

Kommen pro Runde rund $k$ Token hinzu, liest das Modell in Schritt $i$ ca. $i \cdot k$ Token. Über $N$ Schritte summiert sich dies auf $\sim k \cdot N^2 / 2$. Ohne Caching **steigen die Kosten einer Aufgabe quadratisch** mit der Schrittanzahl. Der Cache eliminiert diese quadratische Kurve zwar nicht vollständig, teilt ihren Hauptteil aber durch ca. 10. Daher sind Caching und strikte Iterationslimits gleichermaßen unverzichtbar.

Beispiel: System-Prompt 2.000 Token, jeder Schritt fügt 500 Token hinzu, 20 Schritte → kumuliert ca. 135.000 Eingabetoken. Mit funktionierendem Caching wird der überwiegende Teil davon zum günstigen Cache-Tarif abgerechnet.

Im Code: [`Assemble`](../code/ch01/pipeline.py#L124-L154 "code:Assemble") platziert dynamische Daten ans Ende und nutzt `sort_keys=True`; [`run_agent`](../code/ch01/agent_loop.py#L112-L191 "code:run_agent") arbeitet rein inkrementell. Die Tests [`test_system_prompt_prefix_stable_for_kv_cache`](../code/ch01/test_ch01.py#L138-L141 "code:PipelineTests.test_system_prompt_prefix_stable_for_kv_cache") und [`test_messages_only_appended`](../code/ch01/test_ch01.py#L97-L101 "code:AgentLoopTests.test_messages_only_appended") sichern dieses Verhalten automatisiert ab.

---

## 1.6. Das Herz des Agenten: Schutz des Loops im Code

Der Agent Loop (ReAct: Reason + Act) folgt dem Kreislauf: Kontext analysieren → Tool-Aufruf wählen → Ergebnis empfangen → bis zum Ziel wiederholen. Ohne Schutzmechanismen drohen fatale Szenarien:

- **Endlosschleifen:** Exakt identische Werkzeugaufrufe in Dauerschleife.
- **Kostenexplosionen:** Quadratisches Wachstum der Tokenmengen.
- **Halluzinationskaskaden:** Ein Modellfehler in Schritt N wird in Schritt N+1 als gesicherte Tatsache interpretiert.

### Schutzbarrieren gehören in den Code, nicht in den Prompt

Ein Satz wie «Rufe das Tool maximal dreimal auf» im Prompt ist lediglich eine Bitte. Ein `if repeats >= 3:` im Code ist eine harte Garantie.

| # | Barriere | Standardwert | Reaktion bei Auslösung | Test |
|---|---|---|---|---|
| 1 | **Iterationslimit** | ~5 für Support, ~25 für Coding | Letzter Aufruf ohne Tools: «Antworte anhand vorliegender Daten» | [Test](../code/ch01/test_ch01.py#L64-L69 "code:AgentLoopTests.test_iteration_limit_forces_final_answer") |
| 2 | **Token-Budget** pro Task/Session | Gemäß Produktökonomie | Abbruch, Rückgabe des Zwischenstands, Logging | [Test](../code/ch01/test_ch01.py#L83-L86 "code:AgentLoopTests.test_token_budget") |
| 3 | **Limit identischer Aufrufe in Folge** | 2–3 | Konstruktive Anweisung; bei Missachtung harter Stopp | [Test](../code/ch01/test_ch01.py#L71-L76 "code:AgentLoopTests.test_repeat_guard_sends_constructive_instruction") |
| 4 | **Human-in-the-Loop** | Alle irreversiblen Aktionen | Unterbrechung bis zur menschlichen Bestätigung | [Test](../code/ch01/test_ch01.py#L58-L62 "code:AgentLoopTests.test_hitl_rejects_irreversible") |
| 5 | **Zeitlimit** (Task & Tool) | 60 s Task / 10 s Tool | Task bricht ab; blockiertes Tool liefert verständlichen Fehler | [Test](../code/ch01/test_ch01.py#L88-L95 "code:AgentLoopTests.test_tool_errors_are_instructions") |

Die Standardwerte werden in [`LoopLimits`](../code/ch01/agent_loop.py#L61-L66 "code:LoopLimits") definiert.

Die 5. Barriere (Timeouts) wird im Video nicht explizit aufgeführt, ist in der Praxis jedoch essenziell, da ein Agent sonst an hängenden Drittsystemen blockieren kann.

**Wichtig zu Barriere 3:** Wiederholungen sind nicht zwingend fehlerhaft (z. B. Polling eines Auftragsstatus). Daher werden gezielt *aufeinanderfolgende, identische Aufrufe* (Name + identische Parameter) gezählt. Für Statusabfragen empfiehlt sich ein dediziertes Tool mit serverseitigem Warten.

### Konstruktive Instruktionen bei Limit-Überschreitung

Erreicht der Agent eine Barriere, erhält das Modell keinen kryptischen Fehlercode, sondern eine klare Handlungsanweisung:

> ❌ `Error: tool call limit exceeded`  
> ✅ «Das Tool `get_menu` wurde bereits 3 Mal mit denselben Argumenten aufgerufen. Es gibt keine neuen Daten. Erstelle jetzt die finale Antwort für den Gast.»

Nach dieser Anweisung übergibt der Code **keine Tools mehr an das Modell** — dem LLM bleibt technisch keine andere Wahl, als mit reinem Text zu antworten.

### Human-in-the-Loop: Die Mechanik

Menschliche Freigaben sind mehr als ein einfaches Konsolen-`input()`:

1. Ein Werkzeug ist als irreversibel deklariert (`irreversible=True`): Löschoperationen, Bezahlvorgänge, E-Mail-Versand, Rechtevergaben.
2. Der Loop pausiert, der Zustand (Nachrichtenverlauf, Zähler) wird **persistent gespeichert** (Datenbank oder Message Queue).
3. Dem Operator wird transparent visualisiert, *was* ausgeführt werden soll: Tool, Argumente, Betrag.
4. Nach der Entscheidung setzt der Loop an exakt derselben Stelle fort. Bei Ablehnung erhält das Modell die Meldung: «Vom Menschen abgelehnt; frage den Nutzer nach Anpassungen». Bei Timeout gilt die Aktion als abgelehnt.

### Code-Referenz

In `agent_loop.py` kapselt die Funktion [`run_agent(llm, tools, messages, limits, approve, trace)`](../code/ch01/agent_loop.py#L112-L191 "code:run_agent") diese Logik. Das Modell wird über das Protokoll [`LLM`](../code/ch01/agent_loop.py#L43-L44 "code:LLM") angebunden und liefert [`LLMResponse`](../code/ch01/agent_loop.py#L35-L40 "code:LLMResponse") zurück.

```python
result = run_agent(
    llm=my_llm,
    tools=[get_menu_tool, create_order_tool],
    messages=[{"role": "user", "content": "Ich möchte einen Latte"}],
    limits=LoopLimits(max_iterations=5, max_total_tokens=20_000,
                      max_same_call_repeats=2, max_wall_seconds=60),
    approve=ask_human,          # irreversible Aktionen nur mit Bestätigung
    trace=log_event,            # Observability (Abschnitt 1.8)
)
print(result.stop_reason)       # final | max_iterations | token_budget | wall_clock | stuck
```

Jede Barriere wird in [`AgentLoopTests`](../code/ch01/test_ch01.py#L48-L101 "code:AgentLoopTests") umfassend getestet.

---

## 1.7. Die Hände des Agenten: Tool-Design und MCP

### Die Werkzeuganzahl

Wächst die Anzahl der verfügbaren Werkzeuge (im Video wird der Bereich von 20–50 genannt), droht der **«Tod durch tausend Instruktionen»**: Die Trennschärfe schwindet und das Modell wählt unpassende Werkzeuge. Gegenmaßnahmen:

- Beschränkung auf das minimal notwendige Set pro Agent.
- Zusammenfassung verwandter Tools über Parametrisierung.
- Bei umfangreichen Tool-Sets: Routing (Stufe 3) oder Aufteilung auf Sub-Agenten (Kapitel 5).

### Tool-Definitionen

Im Code repräsentiert die Klasse [`Tool`](../code/ch01/agent_loop.py#L51-L58 "code:Tool") ein Werkzeug: Name, Beschreibung, Argument-Schema, Funktion, Irreversibilitäts-Flag und Timeout (Beispiel: [`TOOLS`](../code/ch01/test_ch01.py#L39-L44 "code:TOOLS")).

1. **Name:** Konvention `verb_substantiv`: `get_user_preferences`, `create_order`.
2. **Positive Beschreibungen:** Präzise erklären, was das Tool tut und *wann* es einzusetzen ist — statt zu beschreiben, was verboten ist.
3. **Striktes JSON-Schema:** Typen, Enums, Pflichtfelder und Standardwerte.

```json
{
  "name": "create_order",
  "description": "Erstellt und bezahlt die Bestellung des Gastes. Nur aufrufen, nachdem der Gast Auswahl und Preis bestätigt hat. Positionen müssen aus get_menu stammen.",
  "input_schema": {
    "type": "object",
    "properties": {
      "item_id": {"type": "string", "description": "ID der Position aus get_menu"},
      "size": {"type": "string", "enum": ["S", "M", "L"], "default": "M"},
      "milk": {"type": "string", "enum": ["regular", "oat", "almond", "none"], "default": "regular"}
    },
    "required": ["item_id"]
  }
}
```

### Fehler als Handlungsanweisungen

Tool-Fehlermeldungen sind **Instruktionen für das Modell**. Ein technischer Stacktrace oder ein HTTP 500 nützen dem Modell nichts — es wird den Aufruf blind wiederholen oder Daten erfinden. Die Fehlermeldung muss erklären, was passiert ist und wie das Problem gelöst werden kann.

| Problemfall | Schlecht | Gut |
|---|---|---|
| Artikel fehlt | `KeyError: 'raf'` | «Artikel 'raf' existiert nicht im Menü. Rufe get_menu auf und schlage eine Alternative vor.» |
| Timeout | `TimeoutError` | «Dienst antwortete nicht innerhalb von 10 s. Nicht sofort wiederholen; Gast informieren, dass Zahlung temporär nicht möglich ist.» |
| Falsche Argumente | `TypeError: ...` | «Ungültige Argumente für create_order. Schema: {...}» |
| Unbekanntes Tool | Loop stürzt ab | «Tool X existiert nicht. Verfügbar sind: get_menu, create_order.» |

Im Code übernimmt dies [`_run_tool`](../code/ch01/agent_loop.py#L91-L107 "code:_run_tool") zusammen mit der Exception [`ToolError`](../code/ch01/agent_loop.py#L47-L48 "code:ToolError") (Test [`test_tool_errors_are_instructions`](../code/ch01/test_ch01.py#L88-L95 "code:AgentLoopTests.test_tool_errors_are_instructions")): Exceptions werden in instruktive Fehlermeldungen transformiert.

### Die Falle automatischer MCP-Wrapper

Das Model Context Protocol (MCP) standardisiert Werkzeugschnittstellen. Automatische Konverter von OpenAPI-Spezifikationen zu MCP-Servern decken jedoch nur den **Happy Path** ab:

- Endpunktbeschreibungen wurden für Entwickler geschrieben, nicht für Sprachmodelle.
- 40 REST-Endpunkte erzeugen 40 Tools → Informationsüberflutung.
- Spezifische Fehler, Retries und Timeouts bleiben unbehandelt — Rohfehler gelangen direkt ins Modell.
- Kritische Operationen (`DELETE`, Zahlungen) besitzen keine Kennzeichnung für Human-in-the-Loop.

Fazit: Werkzeuge müssen nach den **Aufgaben des Agenten** modelliert werden, nicht nach bestehenden REST-Routen.

---

## 1.8. Observability ab Tag 1

Observability beantwortet die zentrale Frage: **Warum hat diese eine Anfrage 12 Dollar statt 2 Cent gekostet und 8 Sekunden gedauert?**

Folgende Daten müssen lückenlos getrackt werden:

- Start, Ende, Dauer und Status jeder Pipelinestufe.
- Gewähltes Modell, Reasoning-Modus und Prompt-Version.
- Jede Loop-Iteration: Tokenverbrauch (Input/Output) und Abbruchgrund.
- Werkzeugaufrufe: Name, Parameter, Laufzeit, Fehlermeldungen.
- **KV Cache Hit Rate:** Anteil der aus dem Cache bedienten Eingabetoken.
- Gesamtkosten pro Geschäftsaufgabe.

Im Code schreibt [`run_pipeline`](../code/ch01/pipeline.py#L219-L230 "code:run_pipeline") Messdaten in `ctx.trace`, während [`run_agent`](../code/ch01/agent_loop.py#L112-L191 "code:run_agent") den `trace`-Callback nutzt. In Produktivumgebungen wird dies an OpenTelemetry oder spezialisierte LLM-Tracing-Plattformen angebunden (Kapitel 7).

---

## Zusammenfassung des Kapitels

1. Die Herausforderung produktiver Agenten liegt im Harness Engineering, nicht im Basismodell.
2. Beginnen Sie auf der untersten Komplexitätsstufe; Autonomie ist nur dort gerechtfertigt, wo Workflows versagen.
3. Der System-Prompt ist ein versionierter Vertrag: Rollengrenzen, Prioritäten, Format und Few-Shot-Beispiele.
4. Pipeline: Günstig → Teuer; externe Dokumente und Tool-Outputs sind prinzipiell unvertrauenswürdig.
5. KV-Cache: Statisches nach vorn, Dynamisches ans Ende, Daten nur anhängen, deterministisches JSON.
6. Loop-Schutz im Code: Iterationslimits, Token-Budgets, Wiederholungssperren, Human-in-the-Loop, Timeouts.
7. Tools: Wenige, präzise Schnittstellen, instruktive Fehlermeldungen; Finger weg von unreflektierten automatischen MCP-Wrappern.
8. Observability und Tracing gehören in das MVP ab Tag 1.

## Fragen zur Selbstkontrolle

1. Worin unterscheidet sich ein Workflow fundamental von einem Agenten? Wer steuert die Schrittfolge?
2. Auf welcher Stufe der Komplexitätsleiter würden Sie «Öffnungszeiten des Cafés ausgeben» umsetzen? Und «Ein passendes Getränk zur Stimmung des Gastes empfehlen und bestellen»?
3. Warum muss die Prioritätsregel «Sicherheit > Budget > Komfort» explizit im Prompt stehen?
4. Warum schützt die Eingangssicherheitsprüfung in Stufe 2 nicht vor Injections in RAG-Dokumenten?
5. Ohne System-Prompt kostet ein Agent mit 20 Schritten etwa viermal so viel wie einer mit 10 Schritten. Warum nicht doppelt so viel? Welche Rolle spielt der KV-Cache?
6. Nennen Sie drei Ursachen für KV-Cache-Misses und wie man sie verhindert.
7. Warum sind Iterationslimits im Prompt unzureichend?
8. Weshalb ist ein `500 Internal Server Error` als Tool-Rückgabe schädlich für den Loop?
9. Welche Barriere hätte den Vorfall bei DataTalks.Club verhindert?

## Praxisaufgabe: Barista Agent v0.1

Basiscode unter `code/ch01/`.

1. **Ausführen:** Starten Sie `python -m unittest test_ch01.py -v`. Untersuchen Sie jeden Test und identifizieren Sie die zugehörige Barriere im Code.
2. **Reales Modell anbinden:** Implementieren Sie den [`LLM`](../code/ch01/agent_loop.py#L43-L44 "code:LLM")-Adapter für einen API-Provider Ihrer Wahl (z. B. Anthropic oder OpenAI), um `messages` und `tools` zu übergeben und [`LLMResponse`](../code/ch01/agent_loop.py#L35-L40 "code:LLMResponse") zu erzeugen (Referenz: [`ScriptedLLM`](../code/ch01/test_ch01.py#L11-L17 "code:ScriptedLLM")).
3. **System-Prompt & Evals:** Formulieren Sie `prompts/barista.v1.md` nach dem 4-Sektionen-Muster und definieren Sie 10 Testfälle (5 reguläre, 5 Edge-Cases wie Prompt Injections oder unbestätigte Bestellungen).
4. **Cache-Messung:** Erfassen Sie `cache_read_tokens` und berechnen Sie die Cache Hit Rate über 10 Dialogrunden. Fügen Sie anschließend testweise einen Zeitstempel in den Prompt-Header ein und vergleichen Sie die Kosten.
5. **Persistentes HITL:** Modifizieren Sie `approve` so, dass der Zustand bei Freigabeanforderungen als JSON auf Festplatte gesichert und erst nach manuellem Trigger wieder aufgenommen wird.

## Glossar

| Begriff | Definition |
|---|---|
| **Agent Loop** | Autonome Schleife, in der das Modell selbstständig über Werkzeuge und Folgeschritte entscheidet |
| **Canary-Token** | Eindeutige Zeichenkette im Prompt zur Detektion unautorisierter Prompt-Extraktion |
| **Compaction** | Reduzierung des Kontextumfangs durch Auslagerung, Filterung oder Zusammenfassung |
| **Few-Shot** | Bereitstellung exemplarischer Ein-/Ausgabe-Paare im Prompt |
| **Harness** | Die Ausführungsumgebung um das LLM: Pipeline, Schutzbarrieren, Tool-Management und Observability |
| **HITL (Human-in-the-Loop)** | Zwingend erforderliche menschliche Freigabe vor Ausführung kritischer Aktionen |
| **KV-Cache / Prompt Caching** | Wiederverwendung vorberechneter Aufmerksamkeitszustände identischer Prompt-Präfixe |
| **MCP** | Model Context Protocol — offener Standard zur Anbindung von Tools und Datenquellen |
| **PII** | Personally Identifiable Information — sensible personenbezogene Daten |
| **Prompt Injection** | Manipulation von Modellanweisungen; direkt über Nutzereingaben oder indirekt über Dokumente/Tools |
| **RAG** | Retrieval-Augmented Generation — dynamisches Nachladen relevanter Wissensfragmente |
| **ReAct** | Reason + Act: Interaktives Zusammenspiel aus Denkschritten und Tool-Handlungen |
| **Workflow** | Systemarchitektur mit deterministisch im Code programmierter Ablaufsteuerung |

## Quellen

1. Anthropic. *Building effective agents* (2024). <https://www.anthropic.com/research/building-effective-agents>
2. Gartner: Mehr als 40% der agentischen KI-Projekte vor dem Aus bis Ende 2027 (Juni 2025). <https://w.media/over-40-percent-of-agentic-ai-projects-could-face-the-axe-by-end-of-2027-gartner/>
3. A. Grigorev. *How I Dropped Our Production Database and Now Pay 10% More for AWS* (2026). <https://alexeyondata.substack.com/p/how-i-dropped-our-production-database>
4. Manus. *Context Engineering for AI Agents: Lessons from Building Manus* (2025).
5. D. Beresnitski. *Anatomy of an AI Agent for Seniors: Pipeline, Cycle, Tools, Traps* (2026). <https://www.youtube.com/watch?v=rN4_Y67Tr8I>
