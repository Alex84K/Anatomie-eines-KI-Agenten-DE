"""Kapitel 2. Agenten-Gedächtnis: Vier Typen, Hintergrund-Lesen/-Schreiben, Kontextkompaktierung.

Ohne externe Abhängigkeiten.
  - Procedural / Semantic / Episodic / Working — vier Gedächtnistypen nach CoALA,
    jeder mit eigenem Lebenszyklus und festem Platz im Kontext;
  - prefetch — Auswahl relevanter Speicherdateien anhand von Headern im Hintergrund,
    ohne die Hauptanfrage zu blockieren;
  - EpisodicWriter — asynchrones Speichern von Erfahrungswerten nach der Antwort mit Cursor
    und At-Least-Once-Semantik (daher idempotente Speicherung);
  - compact — Kompaktierungsleiter für den Kontext: von günstigen zu teuren Schritten.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from retrieval import tokens


# ---------- Vier Gedächtnistypen ----------

@dataclass
class Procedural:
    """Wie gehandelt wird: System-Prompt und Tool-Beschreibungen. Wird vom Entwickler zwischen Releases geändert."""
    system_prompt: str
    tool_descriptions: list[str]

    def render(self) -> str:
        return self.system_prompt + "\n\n# Tools\n" + "\n".join(self.tool_descriptions)


@dataclass
class MemoryFile:
    """Einheit des episodischen/Benutzergedächtnisses: Header getrennt vom Inhalt.

    Anhand des Headers (description) wird entschieden, ob die Datei gelesen wird;
    der Inhalt wird nur bei Bedarf geladen.
    """
    name: str
    description: str
    body: str


@dataclass
class Episode:
    """Lektion aus Erfahrung: [Kontext] → [Aktion] → [Ergebnis] → [Erkenntnis]."""
    context: str
    action: str
    outcome: str
    lesson: str
    source: str = ""                  # Woher die Erkenntnis stammt (Nachrichten-ID) — Provenienz-Anker

    def key(self) -> str:
        # Idempotenzschlüssel: Dieselbe Erkenntnis doppelt protokolliert = ein Eintrag
        raw = json.dumps([self.context, self.action, self.lesson], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class Working:
    """Was gerade passiert: Nachrichten der aktuellen Sitzung. Lebt bis zum Sitzungsende."""
    messages: list[dict] = field(default_factory=list)

    def size(self) -> int:
        return sum(tokens(m.get("content", "")) for m in self.messages)


# ---------- Asynchroner Speicher-Prefetch ----------

Selector = Callable[[str, list[tuple[str, str]]], list[str]]
_pool = ThreadPoolExecutor(max_workers=2)


def start_prefetch(query: str, files: list[MemoryFile], selector: Selector, limit: int = 5) -> Future:
    """Startet die Auswahl relevanter Speicherdateien im Hintergrund und gibt sofort ein Future zurück.

    selector — günstiges Modell: sieht nur (Name, Beschreibung), gibt Dateinamen zurück.
    Die Hauptanfrage läuft währenddessen weiter (Pipeline, Routing).
    """
    manifest = [(f.name, f.description) for f in files]
    by_name = {f.name: f for f in files}

    def job() -> list[MemoryFile]:
        names = selector(query, manifest)
        # Modell könnte Unpassendes liefern — im Code filtern und begrenzen
        picked = [by_name[n] for n in dict.fromkeys(names) if n in by_name]
        return picked[:limit]

    return _pool.submit(job)


def collect_prefetch(fut: Future, wait_s: float) -> list[MemoryFile]:
    """Holt das Prefetch-Ergebnis ab. Bei Timeout oder Fehler ohne Speicher weitermachen statt zu blockieren."""
    try:
        return fut.result(timeout=wait_s)
    except FutureTimeout:
        return []
    except Exception:
        return []


# ---------- Asynchrones Schreiben des episodischen Gedächtnisses ----------

Extractor = Callable[[list[dict]], list[Episode]]


class EpisodicWriter:
    """Extrahiert Lektionen aus neuen Nachrichten und hängt sie an ein Append-only-Log an.

    Cursor — Index der ersten unverarbeiteten Nachricht. Wird erst nach erfolgreichem
    Schreiben weitergesetzt. Stürzt der Prozess dazwischen ab, werden dieselben Nachrichten
    erneut verarbeitet (at-least-once) — Duplikate werden per Key verworfen.
    """

    def __init__(self, log_path: Path, extractor: Extractor):
        self.log_path, self.extractor = log_path, extractor
        self.cursor = 0
        self._keys = set()
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                self._keys.add(json.loads(line)["key"])

    def run(self, messages: list[dict]) -> int:
        """Verarbeitet Nachrichten ab dem Cursor. Gibt die Anzahl neuer Einträge zurück."""
        fresh = messages[self.cursor:]
        if not fresh:
            return 0
        episodes = self.extractor(fresh)          # Kann Exception werfen → Cursor bleibt unverändert
        written = 0
        with self.log_path.open("a", encoding="utf-8") as f:
            for ep in episodes:
                k = ep.key()
                if k in self._keys:
                    continue
                f.write(json.dumps({"key": k, **ep.__dict__}, ensure_ascii=False) + "\n")
                self._keys.add(k)
                written += 1
        self.cursor = len(messages)               # Erst nach erfolgreichem Schreiben
        return written

    def lessons(self) -> list[Episode]:
        if not self.log_path.exists():
            return []
        out = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            d.pop("key")
            out.append(Episode(**d))
        return out


# ---------- Kontext-Kompaktierungsleiter ----------

@dataclass
class CompactConfig:
    window: int = 8_000               # Kontextfenstergröße des Modells (Token)
    offload_over_chars: int = 8_000   # Tool-Ergebnis länger als Schwellenwert — auf Platte auslagern
    keep_head_chars: int = 2_000      # Wie viel vom ausgelagerten Ergebnis im Prompt verbleibt
    micro_at: float = 0.6             # Schwellenwert für Bereinigung alter Tool-Ergebnisse
    auto_at: float = 0.8              # Schwellenwert für automatische Zusammenfassung der Historie
    keep_recent: int = 6              # Wie viele jüngste Nachrichten niemals verändert werden


Summarizer = Callable[[list[dict]], str]


def _size(msgs: list[dict]) -> int:
    return sum(tokens(m.get("content", "")) for m in msgs)


def compact(messages: list[dict], cfg: CompactConfig, store: Path,
            summarize: Summarizer) -> tuple[list[dict], list[str]]:
    """Komprimiert den Verlauf stufenweise, bis der Kontext passt. Gibt (Nachrichten, Schritte) zurück.

    Stufe 1 — Offload: Große Tool-Ergebnisse auf Festplatte, dem Modell nur Anfang und Pfad mitteilen.
             Wird sofort auf neue Tool-Ergebnisse angewandt, daher bleibt das Präfix stabil und der Cache intakt.
    Stufe 2 — Microcompact: Inhalte alter Tool-Ergebnisse durch Platzhalter ersetzen.
    Stufe 3 — Autocompact: Älterer Teil des Verlaufs wird durch Zusammenfassung ersetzt, jüngste Nachrichten bleiben.
    """
    steps: list[str] = []
    out = []
    store.mkdir(parents=True, exist_ok=True)

    # Stufe 1
    for i, m in enumerate(messages):
        c = m.get("content", "")
        if m.get("role") == "tool" and len(c) > cfg.offload_over_chars:
            path = store / f"tool_{i}.txt"
            path.write_text(c, encoding="utf-8")
            m = {**m, "content": c[:cfg.keep_head_chars] +
                 f"\n…[vollständiges Ergebnis: {path}, {len(c)} Zeichen; Datei bei Bedarf lesen]"}
            steps.append(f"offload:{i}")
        out.append(m)

    # Stufe 2
    if _size(out) > cfg.window * cfg.micro_at:
        cut = len(out) - cfg.keep_recent
        for i in range(max(0, cut)):
            if out[i].get("role") == "tool" and not out[i]["content"].startswith("[bereinigt"):
                out[i] = {**out[i], "content": "[bereinigt: altes Tool-Ergebnis]"}
                steps.append(f"micro:{i}")

    # Stufe 3
    if _size(out) > cfg.window * cfg.auto_at and len(out) > cfg.keep_recent:
        head, tail = out[:-cfg.keep_recent], out[-cfg.keep_recent:]
        summary = summarize(head)
        out = [{"role": "user", "content": "[Zusammenfassung des bisherigen Teils der Sitzung]\n" + summary}] + tail
        steps.append("auto")

    return out, steps


# ---------- Kontextzusammenstellung aus allen Gedächtnistypen ----------

def build_context(proc: Procedural, knowledge: str, recalled: list[MemoryFile],
                  lessons: list[Episode], work: Working) -> list[dict]:
    """Statisches nach vorn (wird gecacht), Dynamisches ans Ende.

    Reihenfolge: Prozedural (system) → Working (Sitzungsverlauf, append-only) →
    Block dieser Runde: Gefundenes Wissen, Erinnerungen, Lektionen. Gefundene Daten wechseln
    von Runde zu Runde, stehen daher hinter dem Verlauf und invalidieren dessen Cache nicht.
    """
    dynamic = []
    if knowledge:
        dynamic.append(knowledge)
    if recalled:
        dynamic.append("<memory>\n" + "\n".join(f"## {f.name}\n{f.body}" for f in recalled) + "\n</memory>")
    if lessons:
        dynamic.append("<lessons>\n" + "\n".join(f"- {e.lesson}" for e in lessons) + "\n</lessons>")
    msgs = [{"role": "system", "content": proc.render()}] + work.messages
    if dynamic:
        msgs.append({"role": "user", "content": "\n\n".join(dynamic)})
    return msgs
