"""Kapitel 1. Pipeline-Skelett «Früh blockieren, spät ausgeben».

Jede Stufe ist ein Objekt mit der Methode run(ctx). Eine Stufe kann:
  * den Anforderungskontext (RequestContext) anreichern;
  * die Anfrage abweisen, indem sie StageReject auslöst — danach fallen keine Kosten mehr an.

Die Implementierungen der Stufen hier sind didaktisch (Regex, grobe Tokenzählung).
In Produktionsprojekten werden sie durch reale Komponenten ersetzt: Injection-Klassifikator, RAG, Tokenizer des Modells.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class RequestContext:
    user_id: str
    query: str
    now_iso: str = ""                        # Anfragezeitpunkt — Dynamik, gehört ans ENDE des Kontexts
    enrichment: dict[str, Any] = field(default_factory=dict)
    memory: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    model: str = ""
    thinking: bool = False
    prompt: list[dict] = field(default_factory=list)   # zusammengestellter Kontext für LLM
    raw_output: str = ""
    output: dict[str, Any] | None = None
    trace: list[dict] = field(default_factory=list)


class StageReject(Exception):
    def __init__(self, stage: str, reason: str, user_message: str):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason, self.user_message = stage, reason, user_message


class Stage(Protocol):
    name: str
    def run(self, ctx: RequestContext) -> RequestContext: ...


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)            # grobe Schätzung; in Prod: Tokenizer des Modells


# ---------- 1. Validierung ($0) ----------

class Validate:
    name = "validate"
    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars
    def run(self, ctx):
        q = ctx.query.strip()
        if not q:
            raise StageReject(self.name, "empty", "Bitte geben Sie Ihre Frage ein.")
        if len(q) > self.max_chars:
            raise StageReject(self.name, "too_long", "Nachricht ist zu lang.")
        if re.search(r"[\x00-\x08\x0e-\x1f]", q):
            raise StageReject(self.name, "binary", "Nachricht enthält unzulässige Zeichen.")
        ctx.query = q
        return ctx


# ---------- 2. Eingangssicherheit ----------

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"ignoriere (alle )?(vorherigen|bisherigen) anweisungen",
    r"игнорируй (все )?(предыдущие|прошлые) инструкции",
    r"(zeige|reveal|print|покажи|выведи).{0,20}(system prompt|systemprompt|системн\w* промпт)",
]

def looks_like_injection(text: str) -> bool:
    return any(re.search(p, text, re.I) for p in INJECTION_PATTERNS)


class InputSecurity:
    name = "input_security"
    def run(self, ctx):
        if looks_like_injection(ctx.query):
            raise StageReject(self.name, "prompt_injection", "Diese Anfrage kann nicht ausgeführt werden.")
        return ctx


# ---------- 3–4. Anreicherung, Speicher, RAG ----------

class Enrich:
    name = "enrich"
    def __init__(self, fetch: Callable[[RequestContext], dict]):
        self.fetch = fetch
    def run(self, ctx):
        ctx.enrichment = self.fetch(ctx)
        return ctx


class Retrieve:
    name = "retrieve"
    def __init__(self, memory: Callable[[str], list[str]], rag: Callable[[str], list[str]]):
        self.memory, self.rag = memory, rag
    def run(self, ctx):
        ctx.memory = self.memory(ctx.user_id)
        ctx.retrieved = self.rag(ctx.query)
        return ctx


# ---------- 5. Filterung gefundener Inhalte (indirekte Injections, Müll) ----------

class ContentFilter:
    name = "content_filter"
    def run(self, ctx):
        kept = [d for d in ctx.retrieved if d.strip() and not looks_like_injection(d)]
        ctx.trace.append({"stage": self.name, "dropped": len(ctx.retrieved) - len(kept)})
        ctx.retrieved = kept
        return ctx


# ---------- 6. Kontext-Zusammenstellung und Budget ----------

class Assemble:
    """Reihenfolge ist entscheidend für den KV-Cache: Statisches nach vorn, Dynamisches ans Ende."""
    name = "assemble"
    def __init__(self, system_prompt: str, budget_tokens: int = 3000):
        self.system_prompt, self.budget = system_prompt, budget_tokens

    def run(self, ctx):
        used = approx_tokens(self.system_prompt) + approx_tokens(ctx.query)
        # Greedy: Dokumente und Verlauf aufnehmen, solange ins Budget passend
        docs = []
        for d in ctx.retrieved:
            if used + approx_tokens(d) > self.budget:
                break
            docs.append(d); used += approx_tokens(d)
        history = []
        for m in reversed(ctx.history):                  # neuere Nachrichten sind wichtiger
            t = approx_tokens(json.dumps(m, ensure_ascii=False))
            if used + t > self.budget:
                break
            history.insert(0, m); used += t

        dynamic = json.dumps({"now": ctx.now_iso, "context": ctx.enrichment,
                              "memory": ctx.memory, "documents": docs},
                             sort_keys=True, ensure_ascii=False)   # stabile Schlüsselreihenfolge
        ctx.prompt = (
            [{"role": "system", "content": self.system_prompt}]   # ohne Zeit und Variablen!
            + history
            + [{"role": "user", "content": f"<context>{dynamic}</context>\n\n{ctx.query}"}]
        )
        ctx.trace.append({"stage": self.name, "tokens": used, "docs": len(docs)})
        return ctx


# ---------- 7. Modellauswahl und Modus ----------

class Route:
    name = "route"
    def __init__(self, classify: Callable[[str], str], models: dict[str, str]):
        self.classify, self.models = classify, models
    def run(self, ctx):
        kind = self.classify(ctx.query)            # "simple" | "complex"
        ctx.model = self.models[kind]
        ctx.thinking = kind == "complex"
        return ctx


# ---------- 8. Generierung ----------

class Generate:
    name = "generate"
    def __init__(self, call: Callable[[RequestContext], str]):
        self.call = call                           # hier kann auch run_agent(...) stehen
    def run(self, ctx):
        ctx.raw_output = self.call(ctx)
        return ctx


# ---------- 9. Ausgangssicherheit ----------

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\+?\d[\d\s\-()]{8,}\d")

class OutputSecurity:
    name = "output_security"
    def __init__(self, system_prompt: str, canary: str):
        self.system_prompt, self.canary = system_prompt, canary
    def run(self, ctx):
        out = ctx.raw_output
        # Canary-Token — eindeutiger String im System-Prompt; taucht er in Ausgabe auf = Leak
        if self.canary in out or self.system_prompt[:80] in out:
            raise StageReject(self.name, "prompt_leak", "Diese Antwort kann nicht angezeigt werden.")
        ctx.raw_output = PHONE.sub("[Telefon verborgen]", EMAIL.sub("[E-Mail verborgen]", out))
        return ctx


# ---------- 10. Formatierung und Validierung ----------

class ValidateOutput:
    name = "validate_output"
    def __init__(self, required: dict[str, type]):
        self.required = required
    def run(self, ctx):
        try:
            data = json.loads(ctx.raw_output)
        except json.JSONDecodeError:
            raise StageReject(self.name, "invalid_json", "Antwort konnte nicht formatiert werden, bitte erneut versuchen.")
        for k, t in self.required.items():
            if not isinstance(data.get(k), t):
                raise StageReject(self.name, f"bad_field:{k}", "Antwort konnte nicht formatiert werden, bitte erneut versuchen.")
        ctx.output = data
        return ctx


# ---------- Ausführung mit Tracing ----------

def run_pipeline(stages: list[Stage], ctx: RequestContext) -> RequestContext:
    for stage in stages:
        t0 = time.monotonic()
        try:
            ctx = stage.run(ctx)
            ctx.trace.append({"stage": stage.name, "ok": True,
                              "ms": round((time.monotonic() - t0) * 1000, 2)})
        except StageReject as r:
            ctx.trace.append({"stage": stage.name, "ok": False, "reason": r.reason})
            ctx.output = {"error": r.reason, "message": r.user_message}
            return ctx
    return ctx
