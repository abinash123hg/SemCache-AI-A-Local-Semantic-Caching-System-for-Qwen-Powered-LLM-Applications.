"""Embedding helpers: call Ollama, validate the vector, normalize it.

Normalized vectors make cosine similarity a plain dot product, which keeps
the cache lookup cheap.
"""

import math
import re

from backend import config, ollama_client

# nomic-embed-text's model card recommends task prefixes. I first tried the
# asymmetric pair (search_query: for the new question, search_document: for
# the stored one) and it hurt: measured cosine for "What is the capital of
# France?" vs "Name the capital city of France." was 0.830 asymmetric but
# 0.960 with search_query: on both sides. A cache compares question to
# question, not question to document, so both sides get the same prefix.
QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_query: "  # same prefix on purpose; see comment above

_whitespace = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """Lowercase, trim, collapse whitespace. Used for exact-match hashing so
    'What is AI?' and '  what is   ai? ' are the same question."""
    return _whitespace.sub(" ", text.strip().lower())


def _validate(vec: list[float], expected_dim: int | None) -> None:
    if expected_dim is not None and len(vec) != expected_dim:
        raise ValueError(f"embedding dim {len(vec)} != expected {expected_dim}")
    if any(math.isnan(x) or math.isinf(x) for x in vec):
        raise ValueError("embedding contains NaN/Inf, refusing to store")
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        raise ValueError("embedding is all zeros, refusing to store")


def embed_query(question: str, expected_dim: int | None = None) -> tuple[list[float], int]:
    """Embed a user question (query side). Returns (unit_vector, dim)."""
    vectors, dim = ollama_client.embed([QUERY_PREFIX + question])
    vec = vectors[0]
    _validate(vec, expected_dim)
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec], dim


def embed_document(question: str, expected_dim: int | None = None) -> tuple[list[float], int]:
    """Embed a question for storage (document side). Returns (unit_vector, dim)."""
    vectors, dim = ollama_client.embed([DOCUMENT_PREFIX + question])
    vec = vectors[0]
    _validate(vec, expected_dim)
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec], dim


def expected_dim() -> int:
    """Ask the model once for its real dimension instead of hardcoding 768."""
    _, dim = ollama_client.embed(["dim check"])
    return dim
