# Anatomy of an AI Agent — Lehrbuch für Production Agent Engineering

Lehrbuch und Praxiskurs zum Kurs von Dmitri Beresnitski «Anatomie eines KI-Agenten» (Videoreihe «System Design eines KI-Agenten von innen»). Ziel ist es, den Entwurf und Bau von Produktivsystemen auf Basis von LLMs zu erlernen: Laufzeitumgebungen (Harness), Agentenschleifen (Loops), Werkzeuge (Tools), Speicher (Memory), Multi-Agenten-Systeme sowie Sicherheit und Observability.

Zielgruppe: Senior Backend- und AI/ML-Engineers, Software-Architekten, Tech Leads.

## Repository-Struktur

| Pfad | Inhalt |
| :--- | :--- |
| [`metodichka-anatomy-ai-agent.md`](metodichka-anatomy-ai-agent.md) | Kompakte Zusammenfassung des Gesamtkurses: 7 Module |
| [`textbook/`](textbook/) | Ausführliche Lehrbuchkapitel: Theorie, durchgängiges Beispiel, Kontrollfragen, Praxisteil, Glossar, Quellen |
| [`code/`](code/) | Begleitender Code zu den Kapiteln: Ausschließlich Python-Standardbibliothek, inklusive Tests |

## Kapitel

| # | Thema | Status |
| :--- | :--- | :--- |
| 1 | [Anatomie eines KI-Agenten: Pipeline, Loop, Tools, Fallstricke](textbook/01-anatomy-of-an-ai-agent.md) · [Code](code/ch01/) | ✅ fertig, abgeglichen mit Video #1 |
| 2 | [Architektur von Intelligenz und Gedächtnis: Routing, Wissen, Speicher, Kontextbudget](textbook/02-intelligence-and-memory.md) · [Code](code/ch02/) | ✅ fertig, nach Zusammenfassung von Video #2 |
| 3 | Ökonomie und Auswahl von LLMs in der Produktion | ⏳ Zusammenfassung im Leitfaden |
| 4 | Harness Engineering: Entwurf der Ausführungsumgebung | ⏳ Zusammenfassung im Leitfaden |
| 5 | Multi-Agenten-Systeme und Entwurfsmuster | ⏳ Zusammenfassung im Leitfaden |
| 6 | Langzeitgedächtnis und Wissensmanagement | ⏳ Zusammenfassung im Leitfaden |
| 7 | Produktion: Sicherheit, Ausfallsicherheit, Observability | ⏳ Zusammenfassung im Leitfaden |

## Durchgängiges Beispiel: Barista Agent

Durch alle Kapitel zieht sich ein zentraler Beispiel-Agent: Der Café-Assistent. Er beantwortet Fragen zur Speisekarte, merkt sich Gästepräferenzen, nimmt Bestellungen auf und wickelt Zahlungen ab. Das Beispiel ist kompakt, enthält jedoch alle kritischen Aspekte produktiver Systeme: Geldtransaktionen, externe Dienste, personenbezogene Daten und irreversible Aktionen.

## Code ausführen

Erfordert Python 3.10+, keine externen Abhängigkeiten nötig.

```bash
cd code/ch01 && python3 -m unittest test_ch01.py -v
cd ../ch02 && python3 -m unittest test_ch02.py -v
```

Das Modell wird über ein simples Protokoll (`LLM` in `agent_loop.py`) angebunden: Ein Adapter für die API eines beliebigen Anbieters ist lediglich eine Funktion. Die Tests laufen gegen ein Mock-Modell ohne Netzwerk und ohne API-Keys.

## Code-Links im Text

Die Bezeichnungen von Funktionen, Klassen und Tests in den Kapiteln sind anklickbar: Auf GitHub öffnet der Link den jeweiligen Codeabschnitt mit hervorgehobenen Zeilen. Zeilennummern werden nicht manuell gepflegt — im Link-Titel wird der Symbolname angegeben:

```markdown
[`run_agent`](../code/ch01/agent_loop.py "code:run_agent")
[Test](../code/ch01/test_ch01.py "code:AgentLoopTests.test_token_budget")
```

Nach Änderungen an Code oder Text können die Zeilennummern aktualisiert werden:

```bash
python3 tools/update_code_links.py          # aktuelle #Lstart-Lend einsetzen
python3 tools/update_code_links.py --check  # nur prüfen (wie in der CI)
```

GitHub Actions ([`.github/workflows/code-links.yml`](.github/workflows/code-links.yml)) überprüft bei jedem Push die Aktualität der Links und führt die Tests aus.

## Prinzipien

1. Das Problem produktiver Agenten liegt nicht im Modell, sondern im Engineering drumherum.
2. Minimal notwendige Komplexität: Autonomie wird nur dort eingeführt, wo ein deterministischer Workflow nicht ausreicht.
3. Einschränkungen gehören in den Code, nicht in den Prompt.
4. Jede Aussage mit Quelle. Zahlen ohne Primärquelle sind gekennzeichnet.

## Quellen

- D. Beresnitski, [Anatomy of an AI Agent for Seniors: Pipeline, Cycle, Tools, Traps](https://www.youtube.com/watch?v=rN4_Y67Tr8I) (Video #1)
- D. Beresnitski, [Дорогая модель ≠ умный агент. Анатомия мозга AI-агента](https://youtu.be/vjMxeQ3aIGM) (Video #2)
- Anthropic, [Building effective agents](https://www.anthropic.com/research/building-effective-agents)
- Vollständige Quellenverzeichnisse am Ende jedes Kapitels und im Leitfaden.
