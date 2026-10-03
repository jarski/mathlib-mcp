"""Retrieval over API cards and the symbol index.

Keyword retrieval is BM25 with a small math-vocabulary query expansion ("conjugate
gradient" -> ``cg``, "GEMM" -> ``matmul``). If the optional ``fastembed`` extra is
installed, a dense embedding ranking is fused in with reciprocal rank fusion, which
helps with paraphrased task descriptions ("solve a sparse SPD system iteratively").
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache

from mathlib_mcp.cards import ApiCard, load_cards, load_symbols

_TOKEN = re.compile(r"[a-z0-9]+")

# query term -> extra terms (lower case, already tokenized)
SYNONYMS: dict[str, list[str]] = {
    "gemm": ["matmul", "matrix", "multiply"],
    "multiplication": ["matmul", "multiply", "product"],
    "multiply": ["matmul", "product"],
    "product": ["matmul", "dot"],
    "inverse": ["inv"],
    "invert": ["inv"],
    "eigenvalue": ["eig", "eigh", "eigenvalues", "eigsh"],
    "eigenvalues": ["eig", "eigh", "eigvalsh", "eigsh"],
    "eigenvector": ["eig", "eigh", "eigenvectors"],
    "eigendecomposition": ["eig", "eigh"],
    "conjugate": ["cg"],
    "gradient": ["cg"],
    "least": ["lstsq"],
    "squares": ["lstsq"],
    "regression": ["lstsq"],
    "fourier": ["fft"],
    "spectrum": ["fft", "rfft"],
    "convolution": ["fft", "rfft", "irfft"],
    "frequency": ["fft", "fftfreq"],
    "decomposition": ["factorization", "qr", "svd", "cholesky", "lu"],
    "factorization": ["decomposition", "lu", "qr", "cholesky"],
    "factorize": ["lu", "factor", "cholesky"],
    "triangular": ["solve", "triangular", "trsm"],
    "system": ["solve", "linear"],
    "equations": ["solve", "linear"],
    "spd": ["cholesky", "cg", "positive", "definite"],
    "symmetric": ["eigh", "cholesky", "cg"],
    "sparse": ["csr", "cusparse"],
    "transfer": ["asarray", "asnumpy"],
    "host": ["asnumpy", "asarray", "get"],
    "device": ["asarray"],
    "copy": ["asarray", "asnumpy"],
    "einstein": ["einsum", "contraction"],
    "contraction": ["einsum", "tensordot"],
    "bias": ["epilog", "matmul"],
    "relu": ["epilog", "matmul"],
    "gelu": ["epilog", "matmul"],
    "fused": ["epilog"],
    "fusion": ["epilog"],
    "determinant": ["det", "slogdet"],
    "norm": ["norm"],
    "rank": ["matrix_rank"],
    "pseudoinverse": ["pinv"],
    "batched": ["batched", "batch"],
    "iterative": ["cg", "gmres", "bicgstab"],
    "solver": ["solve", "cg", "gmres", "spsolve"],
    "error": ["errstate", "linalgerror"],
    "singular": ["errstate", "svd"],
    "tf32": ["compute", "type"],
    "precision": ["compute", "type", "tf32"],
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # camelCase -> camel case
    tokens = _TOKEN.findall(text.lower())
    out = []
    for token in tokens:
        out.append(token)
        if len(token) > 4 and token.endswith("s"):  # crude plural folding
            out.append(token[:-1])
    return out


@dataclass
class Hit:
    id: str
    score: float
    kind: str  # function | class
    summary: str
    signature: str
    curated: bool
    has_card: bool

    def to_line(self) -> str:
        star = " [curated]" if self.curated else ""
        return f"- `{self.id}`{star}: {self.summary or '(no summary)'}\n  `{self.signature}`"


@dataclass
class _Doc:
    id: str
    tokens: Counter
    length: int
    boost: float
    hit: Hit


class SearchIndex:
    def __init__(self, cards: dict[str, ApiCard], symbols: dict, *, embeddings: bool | None = None):
        self.docs: list[_Doc] = []
        alias_ids = {alias for card in cards.values() for alias in card.aliases}
        for card in cards.values():
            hit = Hit(card.id, 0.0, card.kind, card.summary, card.signature, card.curated, True)
            self._add(card.id, card.search_text(), 1.3 if card.curated else 1.0, hit)
        excluded = [p for lib in symbols.get("libraries", {}).values()
                    for p in lib.get("exclude", [])]
        for qualname, entry in symbols["symbols"].items():
            if qualname in cards or qualname in alias_ids:
                continue
            if entry["kind"] not in ("function", "class") or "summary" not in entry:
                continue
            if any(part.startswith("_") for part in qualname.split(".")):
                continue
            if any(qualname.startswith(prefix + ".") for prefix in excluded):
                continue
            name = qualname.rsplit(".", 1)[1]
            params = entry.get("params")
            signature = f"{name}({', '.join(p[0] for p in params)})" if params else f"{name}(...)"
            hit = Hit(qualname, 0.0, entry["kind"], entry["summary"], signature, False, False)
            text = f"{qualname} {qualname.replace('.', ' ').replace('_', ' ')} {entry['summary']}"
            self._add(qualname, text, 0.7, hit)
        self.n = len(self.docs)
        self.avg_len = sum(d.length for d in self.docs) / max(self.n, 1)
        df: Counter = Counter()
        for doc in self.docs:
            df.update(doc.tokens.keys())
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        use_embeddings = (embeddings if embeddings is not None
                          else os.environ.get("MATHLIB_MCP_EMBEDDINGS", "auto") != "off")
        self._dense = _DenseIndex.try_create(self.docs) if use_embeddings else None

    def _add(self, doc_id: str, text: str, boost: float, hit: Hit) -> None:
        tokens = Counter(tokenize(text))
        # the identifier itself is the strongest signal
        for token in tokenize(doc_id.rsplit(".", 1)[1]):
            tokens[token] += 3
        self.docs.append(_Doc(doc_id, tokens, sum(tokens.values()), boost, hit))

    def bm25(self, query: str, k1: float = 1.4, b: float = 0.75) -> list[tuple[float, _Doc]]:
        terms = tokenize(query)
        expanded = list(terms)
        for term in terms:
            expanded += SYNONYMS.get(term, [])
        weights = Counter({t: 1.0 for t in terms})
        for term in expanded:
            weights.setdefault(term, 0.5)
        scored = []
        for doc in self.docs:
            score = 0.0
            for term, weight in weights.items():
                tf = doc.tokens.get(term)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc.length / self.avg_len))
                score += weight * idf * norm
            if score > 0:
                scored.append((score * doc.boost, doc))
        scored.sort(key=lambda item: -item[0])
        return scored

    def search(self, query: str, k: int = 8, library: str | None = None) -> list[Hit]:
        ranked = self.bm25(query)
        if self._dense is not None:
            ranked = _rrf(ranked, self._dense.rank(query))
        hits = []
        for score, doc in ranked:
            if library and not doc.id.startswith(library):
                continue
            hit = Hit(**{**doc.hit.__dict__, "score": round(score, 4)})
            hits.append(hit)
            if len(hits) == k:
                break
        return hits


def _rrf(*rankings: list[tuple[float, _Doc]], k: int = 60) -> list[tuple[float, _Doc]]:
    scores: dict[str, float] = {}
    docs: dict[str, _Doc] = {}
    for ranking in rankings:
        for position, (_, doc) in enumerate(ranking[:200]):
            scores[doc.id] = scores.get(doc.id, 0.0) + doc.boost / (k + position + 1)
            docs[doc.id] = doc
    return sorted(((s, docs[i]) for i, s in scores.items()), key=lambda item: -item[0])


class _DenseIndex:
    """Optional embedding ranking (``pip install mathlib-mcp[embeddings]``)."""

    MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model, docs: list[_Doc], vectors):
        self.model, self.docs, self.vectors = model, docs, vectors

    @classmethod
    def try_create(cls, docs: list[_Doc]) -> _DenseIndex | None:
        try:
            import numpy as np
            from fastembed import TextEmbedding
        except ImportError:
            return None
        model = TextEmbedding(cls.MODEL)
        card_docs = [d for d in docs if d.hit.has_card]  # embed cards only: fast start-up
        texts = [f"{d.id}: {d.hit.summary}" for d in card_docs]
        vectors = np.array(list(model.embed(texts)))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return cls(model, card_docs, vectors)

    def rank(self, query: str) -> list[tuple[float, _Doc]]:
        import numpy as np

        q = np.array(list(self.model.query_embed([query]))[0])
        q /= np.linalg.norm(q)
        sims = self.vectors @ q
        order = np.argsort(-sims)
        return [(float(sims[i]), self.docs[i]) for i in order]


@cache
def default_search_index() -> SearchIndex:
    return SearchIndex(load_cards(), load_symbols())
