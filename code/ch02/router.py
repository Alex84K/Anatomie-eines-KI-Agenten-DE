"""Kapitel 2. Intelligentes Routing: Welches Modell aufrufen und wie lange nachdenken lassen.

Ohne externe Abhängigkeiten. Drei Schichten der Modellauswahl:
  1. Modalitätenfilter — Modelle ausschließen, die die Eingabe nicht verarbeiten können;
  2. Komplexitätsklassifikator — kleines schnelles Modell (oder ML-Klassifikator);
  3. Fallback-Schicht — deterministische Regeln nach Schlüsselbegriffen, danach Standardmodell.
Darüber — Berücksichtigung des Prompt-Caches: Modellwechsel mitten im Dialog setzt den Cache zurück,
daher wird die Entscheidung «Modell wechseln» anhand der Kosten pro Runde getroffen, nicht nach Listenpreis.
Separat — Thinking Mode Policy: Reasoning ist teuer (wird als Ausgabe abgerechnet),
daher in der Planungsphase aktiv und in der Ausführungsphase deaktiviert.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


# ---------- Modelle und Preise ----------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: str                         # "small" | "frontier"
    price_in: float                   # $ pro 1M Eingabetoken
    price_out: float                  # $ pro 1M Ausgabe- (und Thinking-) Token
    price_cache_read: float           # $ pro 1M aus dem Cache gelesene Token
    price_cache_write: float          # $ pro 1M in den Cache geschriebene Token
    modalities: frozenset[str] = frozenset({"text"})


def turn_cost(m: ModelSpec, cached: int, uncached: int, output: int, cache_write: int = 0) -> float:
    """Kosten einer Dialogrunde in Dollar.

    cached      — Präfixtoken, die das Modell aus dem Cache liest;
    uncached    — Token, die das Modell ohne Cache liest;
    cache_write — wie viele der ungecachten Token dabei in den Cache geschrieben werden;
    output      — Ausgabetoken, einschließlich Thinking.
    """
    plain = uncached - cache_write
    return (cached * m.price_cache_read + plain * m.price_in
            + cache_write * m.price_cache_write + output * m.price_out) / 1_000_000


# ---------- Anfrage und Entscheidung ----------

@dataclass
class Request:
    text: str
    modalities: frozenset[str] = frozenset({"text"})
    current_model: str | None = None  # auf welchem Modell der Dialog läuft (None = 1. Runde)
    history_tokens: int = 0           # Länge des bereits aufgelaufenen Kontexts
    new_tokens: int = 200             # wie viel in dieser Runde hinzukommt
    expected_output: int = 300


@dataclass
class RouteDecision:
    model: ModelSpec
    reason: str                       # für Tracing: warum dieses Modell gewählt wurde
    trace: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    label: str                        # "simple" | "complex"
    confidence: float                 # 0..1


class Classifier(Protocol):
    def __call__(self, text: str) -> Verdict: ...


# ---------- Schicht 3: Deterministische Regeln ----------

COMPLEX_HINTS = re.compile(
    r"(vergleiche|analysiere|warum|plane|erstelle einen plan|berechne|erkläre den unterschied|"
    r"сравни|проанализируй|почему|спланируй|составь план|рассчитай|объясни разницу)",
    re.I
)
SIMPLE_HINTS = re.compile(
    r"(wie viel kostet|preis|bestellstatus|öffnungszeiten|adresse|gibt es|"
    r"сколько стоит|цена|статус заказа|часы работы|адрес|есть ли)",
    re.I
)


def rule_fallback(text: str) -> str | None:
    """Regeln nach Schlüsselbegriffen. None bedeutet: Regeln können nicht entscheiden."""
    if COMPLEX_HINTS.search(text):
        return "complex"
    if SIMPLE_HINTS.search(text):
        return "simple"
    return None


# ---------- Router ----------

def route(
    req: Request,
    models: list[ModelSpec],
    classifier: Classifier,
    default_tier: str = "frontier",
    min_confidence: float = 0.7,
) -> RouteDecision:
    trace: list[str] = []

    # Schicht 1: Modalitäten. Am günstigsten: kein Bild an ein reines Textmodell senden.
    capable = [m for m in models if req.modalities <= m.modalities]
    if not capable:
        raise ValueError(f"Kein Modell für Modalitäten {sorted(req.modalities)} verfügbar")
    trace.append("modality:" + ",".join(m.name for m in capable))

    # Schicht 2: Klassifikator. Fehler oder geringe Konfidenz → Schicht 3.
    tier: str | None = None
    try:
        v = classifier(req.text)
        trace.append(f"classifier:{v.label}:{v.confidence:.2f}")
        if v.confidence >= min_confidence:
            tier = "small" if v.label == "simple" else "frontier"
    except Exception as e:  # Klassifikator abgestürzt — Anfrage darf nicht mitabstürzen
        trace.append(f"classifier_error:{type(e).__name__}")

    reason = "classifier"
    if tier is None:
        label = rule_fallback(req.text)
        if label is not None:
            tier, reason = ("small" if label == "simple" else "frontier"), "rules"
        else:
            tier, reason = default_tier, "default"
        trace.append(f"fallback:{reason}:{tier}")

    pool = [m for m in capable if m.tier == tier] or capable
    chosen = min(pool, key=lambda m: m.price_in)

    # Cache-Beachtung: Läuft der Dialog bereits auf einem anderen Modell, Rundekosten vergleichen.
    current = next((m for m in capable if m.name == req.current_model), None)
    if current is not None and current.name != chosen.name and req.history_tokens:
        stay = turn_cost(current, req.history_tokens, req.new_tokens, req.expected_output)
        switch = turn_cost(chosen, 0, req.history_tokens + req.new_tokens, req.expected_output,
                           cache_write=req.history_tokens + req.new_tokens)
        trace.append(f"cache_check:stay=${stay:.4f}:switch=${switch:.4f}")
        if stay <= switch:
            return RouteDecision(current, "cache_stay", trace)

    return RouteDecision(chosen, reason, trace)


# ---------- Thinking Mode ----------

@dataclass
class ThinkingPolicy:
    """Reasoning in der Planungsphase nutzen, in der Ausführung ohne Reasoning arbeiten."""
    plan_iterations: int = 2          # wie viele erste Iterationen des Loops Thinking nutzen
    budget_tokens: int = 4_000        # Obergrenze an Thinking-Token pro Iteration
    replan_on_error: bool = True      # Tool-Fehler → einen Schritt lang erneut nachdenken

    def budget(self, iteration: int, last_step_failed: bool = False) -> int:
        """Wie viele Thinking-Token bei Iteration (1-basiert) erlaubt sind. 0 = Modus deaktiviert."""
        if iteration <= self.plan_iterations:
            return self.budget_tokens
        if self.replan_on_error and last_step_failed:
            return self.budget_tokens // 2
        return 0
