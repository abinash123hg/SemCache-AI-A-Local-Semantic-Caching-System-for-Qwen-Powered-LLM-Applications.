"""Talks to the local Ollama server: status, embeddings, generation, pulls."""

import logging
from typing import Any, Iterator

import httpx

from backend import config

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns a bad response."""


def _client(timeout: float | None = None) -> httpx.Client:
    return httpx.Client(
        base_url=config.OLLAMA_BASE_URL,
        timeout=timeout or config.REQUEST_TIMEOUT_SECONDS,
    )


def is_running() -> bool:
    try:
        with _client(timeout=3.0) as c:
            return c.get("/api/tags").status_code == 200
    except httpx.HTTPError:
        return False


def list_models() -> list[str]:
    """Return installed model tags, e.g. ['qwen2.5:3b', 'nomic-embed-text:latest']."""
    try:
        with _client(timeout=5.0) as c:
            r = c.get("/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
    except httpx.HTTPError as e:
        raise OllamaError(f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}: {e}") from e


def missing_models() -> list[str]:
    """Which of the two models we need are not installed yet.
    Tags from Ollama may carry ':latest', so compare base names too."""
    installed = list_models()

    def have(want: str) -> bool:
        base = want.split(":")[0]
        return any(t == want or t.split(":")[0] == base for t in installed)

    return [m for m in (config.CHAT_MODEL, config.EMBED_MODEL) if not have(m)]


def pull_stream(model: str) -> Iterator[dict[str, Any]]:
    """Yield progress dicts while pulling a model. Caller decides what to show."""
    try:
        with _client(timeout=None) as c:
            with c.stream("POST", "/api/pull", json={"model": model, "stream": True}) as r:
                if r.status_code != 200:
                    body = r.read().decode(errors="replace")
                    raise OllamaError(f"Ollama pull failed ({r.status_code}): {body}")
                import json
                for line in r.iter_lines():
                    if line.strip():
                        yield json.loads(line)
    except httpx.HTTPError as e:
        raise OllamaError(f"Pull of {model} failed: {e}") from e


def embed(texts: list[str], model: str | None = None) -> tuple[list[list[float]], int]:
    """Embed a batch of texts. Returns (vectors, dim).
    We never hardcode the dimension; we read it from the response."""
    model = model or config.EMBED_MODEL
    payload = {"model": model, "input": texts}
    try:
        with _client() as c:
            r = c.post("/api/embed", json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        raise OllamaError(f"Embedding request failed: {e}") from e

    vectors = data.get("embeddings")
    if not vectors:
        raise OllamaError(f"Ollama returned no embeddings for model {model}")
    return vectors, len(vectors[0])


def generate(
    question: str,
    system_prompt: str,
    temperature: float,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate an answer. Returns the raw Ollama response so the caller can
    read timings and token counts (prompt_eval_count, eval_count)."""
    model = model or config.CHAT_MODEL
    payload = {
        "model": model,
        "prompt": question,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        with _client() as c:
            r = c.post("/api/generate", json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise OllamaError(f"Generation with {model} failed: {e}") from e


def generate_stream(
    question: str,
    system_prompt: str,
    temperature: float,
    model: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream a generation chunk by chunk. Each chunk carries a 'response'
    fragment; the final chunk has done=True plus the token counts."""
    model = model or config.CHAT_MODEL
    payload = {
        "model": model,
        "prompt": question,
        "system": system_prompt,
        "stream": True,
        "options": {"temperature": temperature},
    }
    try:
        with _client(timeout=None) as c:
            with c.stream("POST", "/api/generate", json=payload) as r:
                if r.status_code != 200:
                    body = r.read().decode(errors="replace")
                    raise OllamaError(f"Generation with {model} failed ({r.status_code}): {body}")
                import json
                for line in r.iter_lines():
                    if line.strip():
                        yield json.loads(line)
    except httpx.HTTPError as e:
        raise OllamaError(f"Generation with {model} failed: {e}") from e
