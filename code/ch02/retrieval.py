"""Kapitel 2. Hybrid aus RAG + Kontextfenster: Wenig finden, aber das Richtige.

Ohne externe Abhängigkeiten. Didaktische Implementierung:
  - Dokumenten-Chunking mit Überlappung;
  - lexikalische Suche (vereinfachtes BM25) anstelle einer Vektordatenbank;
  - Kontext-Zusammenstellung innerhalb des Token-Budgets mit Rand-Anordnung (Edge-Order)
    gegen den «Lost in the Middle»-Effekt;
  - Kostenvergleich: «Ganzes Dokument im Kontext» vs. «RAG + Kontextfenster».
In Produktionsprojekten wird die Suche durch Embeddings/Hybridsuche + Reranker ersetzt, ohne die Schnittstelle zu ändern.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

WORD = re.compile(r"\w+", re.U)


def tokens(text: str) -> int:
    """Grobe Schätzung der Tokenanzahl. In Produktionssystemen: Tokenizer des Providers."""
    return max(1, len(text) // 4)


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str


def chunk_text(source: str, text: str, size: int = 400, overlap: int = 80) -> list[Chunk]:
    """Teilt Text in Chunks von ca. size Zeichen mit Überlappung.

    Die Überlappung verhindert, dass Fakten an Chunk-Grenzen verloren gehen.
    Trennung an Wortgrenzen, um Wörter nicht zu zerschneiden.
    """
    if overlap >= size:
        raise ValueError("overlap muss kleiner als size sein")
    out, start, n = [], 0, 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > start else end
        out.append(Chunk(f"{source}#{n}", source, text[start:end].strip()))
        if end >= len(text):
            break
        start, n = end - overlap, n + 1
    return out


class Index:
    """Vereinfachtes BM25 über Chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = [Counter(words(c.text)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if chunks else 1.0
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        q = words(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > min_score:
                scored.append((self.chunks[i], s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


def edge_order(items: list):
    """Anordnung an den Rändern: 1. nach Relevanz an den Anfang, 2. ans Ende, 3. an zweite Stelle usw.

    Modelle verwerten Informationen aus der Mitte langer Kontexte schlechter (Lost in the Middle),
    daher platzieren wir das Wichtigste dort, wo die Aufmerksamkeit am höchsten ist.
    """
    head, tail = [], []
    for i, x in enumerate(items):
        (head if i % 2 == 0 else tail).append(x)
    return head + tail[::-1]


def assemble(query: str, found: list[tuple[Chunk, float]], budget_tokens: int) -> tuple[str, list[str]]:
    """Stellt einen Wissensblock für den Prompt innerhalb des Budgets zusammen. Gibt (Text, Chunk-IDs) zurück.

    Chunks werden gierig nach absteigender Relevanz ausgewählt, solange das Budget reicht,
    und danach an den Rändern verteilt.
    Jeder Chunk erhält eine Quellenangabe: Das Modell kann referenzieren, wir können prüfen.
    """
    picked, used = [], 0
    for chunk, _ in found:
        cost = tokens(chunk.text) + 10
        if used + cost > budget_tokens:
            continue
        picked.append(chunk)
        used += cost
    ordered = edge_order(picked)
    body = "\n\n".join(f"[{c.id}]\n{c.text}" for c in ordered)
    return f"<knowledge query={query!r}>\n{body}\n</knowledge>", [c.id for c in ordered]


@dataclass
class CostCompare:
    long_context: float
    rag: float

    @property
    def ratio(self) -> float:
        return self.long_context / self.rag if self.rag else math.inf


def compare_costs(doc_tokens: int, rag_tokens: int, turns: int, price_in: float,
                  price_cache_read: float | None = None, embed_price: float = 0.02) -> CostCompare:
    """Berechnet die Kosten eines Dialogs über `turns` Runden: Gesamtes Dokument im Kontext vs. RAG.

    long context: Dokument wird in jeder Runde gelesen (aus dem Cache, falls vorhanden);
    RAG: Dokument wird einmal durch Embeddings verarbeitet + in jeder Runde gefundene Chunks gelesen.
    Preise in $ pro 1M Token.
    """
    if price_cache_read is None:
        long_ctx = doc_tokens * turns * price_in
    else:  # erste Runde ohne Cache, danach Lesen aus dem Cache
        long_ctx = doc_tokens * price_in + doc_tokens * (turns - 1) * price_cache_read
    rag = doc_tokens * embed_price + rag_tokens * turns * price_in
    return CostCompare(long_ctx / 1_000_000, rag / 1_000_000)
