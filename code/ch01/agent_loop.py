"""Kapitel 1. Abgesicherter Agenten-Loop (Agent Loop) mit Code-Barrieren.

Ohne externe Abhängigkeiten. LLM wird über das Protokoll `LLM` angebunden:
Jeder Client (Anthropic, OpenAI, lokales Modell) wird in eine Funktion
eingebettet, die `LLMResponse` zurückgibt.

Barrieren (alle im Code, nicht im Prompt):
  1. Iterationslimit;
  2. Token-Budget pro Aufgabe;
  3. Limit für identische aufeinanderfolgende Tool-Aufrufe;
  4. Human-in-the-Loop für irreversible Aktionen;
  + Zeitlimit für die Gesamtaufgabe und Timeout für jedes Tool.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# ---------- Datenmodell ----------

@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]

    def key(self) -> str:
        # sort_keys — stabile Schlüsselreihenfolge (wichtig für KV-Cache und Vergleiche)
        return self.name + ":" + json.dumps(self.args, sort_keys=True, ensure_ascii=False)


@dataclass
class LLMResponse:
    text: str | None                 # finale Antwort (wenn kein Tool aufgerufen wird)
    tool_call: ToolCall | None       # Tool-Aufruf-Anforderung
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    def __call__(self, messages: list[dict], tools: list["Tool"] | None) -> LLMResponse: ...


class ToolError(Exception):
    """Fehler, dessen Text eine Instruktion für das Modell ist, kein Stacktrace."""


@dataclass
class Tool:
    name: str                        # verb_substantiv: get_menu, create_order
    description: str                 # Was es tut und wann anzuwenden (positive Formulierungen)
    schema: dict[str, Any]           # JSON Schema der Argumente
    fn: Callable[..., Any]
    irreversible: bool = False       # True → Bestätigung durch Menschen erforderlich
    timeout_s: float = 10.0


@dataclass
class LoopLimits:
    max_iterations: int = 5          # Support ~5, Coding-Agent ~25
    max_total_tokens: int = 20_000
    max_same_call_repeats: int = 2   # wie oft ein identischer Aufruf nacheinander erlaubt ist
    max_wall_seconds: float = 60.0


@dataclass
class AgentResult:
    answer: str | None
    stop_reason: str                 # final | max_iterations | token_budget | wall_clock | stuck
    iterations: int
    total_tokens: int
    messages: list[dict] = field(default_factory=list)


Approver = Callable[[ToolCall], bool]
Tracer = Callable[..., None]


def _noop_trace(event: str, **fields: Any) -> None:
    pass


# ---------- Hilfsfunktionen ----------

_executor = ThreadPoolExecutor(max_workers=4)


def _run_tool(tool: Tool, args: dict) -> str:
    """Führt ein Tool mit Timeout aus. Jeder Fehler wird zu einer verständlichen Instruktion."""
    future = _executor.submit(tool.fn, **args)
    try:
        result = future.result(timeout=tool.timeout_s)
        return json.dumps(result, sort_keys=True, ensure_ascii=False)
    except FutureTimeout:
        return (f"FEHLER: {tool.name} antwortete nicht innerhalb von {tool.timeout_s} s. "
                "Wiederhole den Aufruf nicht sofort; informiere den Nutzer, dass der Dienst temporär nicht erreichbar ist.")
    except ToolError as e:
        return f"FEHLER: {e}"
    except TypeError as e:  # ungültige Argumente
        return (f"FEHLER: Ungültige Argumente für {tool.name} ({e}). "
                f"Argumentschema: {json.dumps(tool.schema, ensure_ascii=False)}")
    except Exception as e:  # keinen Stacktrace an das Modell übergeben
        return (f"FEHLER: Interner Fehler in {tool.name} ({type(e).__name__}). "
                "Wiederhole den Aufruf nicht; bitte den Nutzer, es später erneut zu versuchen.")


# ---------- Loop ----------

def run_agent(
    llm: LLM,
    tools: list[Tool],
    messages: list[dict],
    limits: LoopLimits = LoopLimits(),
    approve: Approver = lambda call: False,   # standardmäßig sind irreversible Aktionen verboten
    trace: Tracer = _noop_trace,
) -> AgentResult:
    by_name = {t.name: t for t in tools}
    messages = list(messages)                 # Eingabe nicht mutieren; nur am Ende anfügen
    total_tokens = 0
    started = time.monotonic()
    last_key, repeats = None, 0
    finalize_requested = False

    def finish(answer: str | None, reason: str, i: int) -> AgentResult:
        trace("loop.end", stop_reason=reason, iterations=i, total_tokens=total_tokens)
        return AgentResult(answer, reason, i, total_tokens, messages)

    for i in range(1, limits.max_iterations + 1):
        if time.monotonic() - started > limits.max_wall_seconds:
            return finish(None, "wall_clock", i - 1)

        # Nach Aufforderung zum Abschluss werden keine Tools mehr übergeben — Modell muss mit Text antworten
        resp = llm(messages, None if finalize_requested else tools)
        total_tokens += resp.input_tokens + resp.output_tokens
        trace("loop.iteration", i=i, input_tokens=resp.input_tokens,
              output_tokens=resp.output_tokens, tool=resp.tool_call.name if resp.tool_call else None)

        if resp.tool_call is None:
            messages.append({"role": "assistant", "content": resp.text})
            return finish(resp.text, "final", i)

        if finalize_requested:               # Modell hat die Instruktion ignoriert
            return finish(None, "stuck", i)

        if total_tokens >= limits.max_total_tokens:
            return finish(None, "token_budget", i)

        call = resp.tool_call
        messages.append({"role": "assistant", "tool_call": {"name": call.name, "args": call.args}})

        # Barriere 3: identische Aufrufe in Folge
        key = call.key()
        repeats = repeats + 1 if key == last_key else 0
        last_key = key
        if repeats >= limits.max_same_call_repeats:
            finalize_requested = True
            messages.append({"role": "tool", "name": call.name, "content":
                f"Das Tool {call.name} wurde bereits {repeats + 1} Mal mit denselben Argumenten aufgerufen, "
                "es gibt keine neuen Daten. Die Daten sind gesammelt — erstelle jetzt die finale Antwort."})
            trace("guard.repeat", tool=call.name, repeats=repeats + 1)
            continue

        tool = by_name.get(call.name)
        if tool is None:
            content = (f"FEHLER: Tool {call.name} existiert nicht. "
                       f"Verfügbar: {', '.join(sorted(by_name))}.")
        elif tool.irreversible and not approve(call):   # Barriere 4: HITL
            content = (f"Die Aktion {call.name} wurde vom Menschen abgelehnt. Wiederhole sie nicht; "
                       "frage beim Nutzer nach, was geändert werden soll.")
            trace("guard.hitl_rejected", tool=call.name)
        else:
            t0 = time.monotonic()
            content = _run_tool(tool, call.args)
            trace("tool.call", tool=call.name, ms=round((time.monotonic() - t0) * 1000),
                  error=content.startswith("FEHLER"))
        messages.append({"role": "tool", "name": call.name, "content": content})

    # Barriere 1: Iterationslimit. Dem Modell eine letzte Chance geben, ohne Tools zu antworten —
    # aber nur, wenn das Token-Budget es erlaubt.
    if total_tokens < limits.max_total_tokens:
        messages.append({"role": "user", "content":
            "Schrittlimit erreicht. Antworte dem Nutzer anhand der bereits gesammelten Daten, ohne Tools."})
        resp = llm(messages, None)
        total_tokens += resp.input_tokens + resp.output_tokens
        if resp.tool_call is None and resp.text:
            messages.append({"role": "assistant", "content": resp.text})
            return finish(resp.text, "max_iterations", limits.max_iterations)
    return finish(None, "max_iterations", limits.max_iterations)
