"""SQLite-backed semantic cache.

Tables:
  entries   - cached question/answer pairs with their embeddings
  stats     - one row per /chat request, used by analytics and benchmarks
  feedback  - "flag as wrong" reports
  settings  - runtime-tunable tau/delta/ttl (persisted across restarts)

Embeddings are stored as float32 blobs and compared with NumPy brute-force
cosine similarity. At this scale (hundreds to thousands of entries) that is
faster to reason about than any vector database, and vectors are normalized
at insert so similarity is just a dot product.
"""

import hashlib
import json
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend import config

_write_lock = threading.Lock()


def _now() -> float:
    return time.time()


def namespace_of(chat_model: str, embed_model: str, system_prompt: str, temperature: float) -> str:
    """Entries are never reused across a different model/prompt/temperature.
    A hash keeps the key short and avoids storing prompts in every row."""
    raw = json.dumps(
        [chat_model, embed_model, system_prompt, round(temperature, 3)],
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def exact_hash(normalized_question: str) -> str:
    return hashlib.sha256(normalized_question.encode()).hexdigest()


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"<{n}f", blob), dtype=np.float64)


@dataclass
class LookupResult:
    hit: bool
    exact: bool = False
    entry_id: int | None = None
    question: str | None = None       # matched (stored) question
    answer: str | None = None
    similarity: float = 0.0
    margin: float = 0.0
    age_seconds: float = 0.0
    reason: str = ""                  # why it missed, in plain words


class Cache:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or config.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._load_settings()

    def close(self) -> None:
        # On Windows an open sqlite handle locks the file, so tests and any
        # clean shutdown must close it explicitly.
        self._conn.close()

    # ---------- setup ----------

    def _init_schema(self) -> None:
        with _write_lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    question TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    exact_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    answer TEXT NOT NULL,
                    chat_model TEXT NOT NULL,
                    embed_model TEXT NOT NULL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    flagged INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_entries_ns ON entries(namespace);
                CREATE INDEX IF NOT EXISTS idx_entries_hash ON entries(exact_hash);

                CREATE TABLE IF NOT EXISTS stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    namespace TEXT NOT NULL,
                    hit INTEGER NOT NULL,
                    exact_hit INTEGER NOT NULL,
                    similarity REAL,
                    margin REAL,
                    embedding_ms REAL,
                    retrieval_ms REAL,
                    generation_ms REAL,
                    total_ms REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    question TEXT
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    entry_id INTEGER,
                    question TEXT,
                    reason TEXT
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def _load_settings(self) -> None:
        rows = {r["key"]: r["value"] for r in self._conn.execute("SELECT key, value FROM settings")}
        self.tau = float(rows.get("tau", config.SIMILARITY_THRESHOLD))
        self.delta = float(rows.get("delta", config.MARGIN_THRESHOLD))
        self.ttl_hours = float(rows.get("ttl_hours", config.TTL_HOURS))

    def update_settings(self, tau: float | None = None, delta: float | None = None,
                        ttl_hours: float | None = None) -> dict:
        if tau is not None:
            if not (0.0 <= tau <= 1.0):
                raise ValueError("tau must be between 0 and 1")
            self.tau = tau
        if delta is not None:
            if not (0.0 <= delta <= 1.0):
                raise ValueError("delta must be between 0 and 1")
            self.delta = delta
        if ttl_hours is not None:
            if ttl_hours < 0:
                raise ValueError("ttl_hours cannot be negative")
            self.ttl_hours = ttl_hours
        with _write_lock, self._conn:
            for k, v in (("tau", self.tau), ("delta", self.delta), ("ttl_hours", self.ttl_hours)):
                self._conn.execute(
                    "INSERT INTO settings(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (k, str(v)),
                )
        return self.get_settings()

    def get_settings(self) -> dict:
        return {"tau": self.tau, "delta": self.delta, "ttl_hours": self.ttl_hours}

    # ---------- lookup ----------

    def lookup(self, namespace: str, normalized: str,
               query_vec: np.ndarray | None) -> LookupResult:
        """Exact match first (cheap hash), then semantic top-1 with margin rule.
        query_vec may be None for the exact-only mode used by benchmarks."""
        now = _now()
        row = self._conn.execute(
            "SELECT * FROM entries WHERE namespace = ? AND exact_hash = ? "
            "AND flagged = 0 AND expires_at > ? ORDER BY created_at DESC LIMIT 1",
            (namespace, exact_hash(normalized), now),
        ).fetchone()
        if row:
            self._bump_hit(row["id"])
            return LookupResult(
                hit=True, exact=True, entry_id=row["id"], question=row["question"],
                answer=row["answer"], similarity=1.0, margin=1.0,
                age_seconds=now - row["created_at"], reason="exact match",
            )

        if query_vec is None:
            return LookupResult(hit=False, reason="no exact match (semantic search skipped)")

        # Semantic search over all live entries in this namespace.
        rows = self._conn.execute(
            "SELECT id, question, answer, embedding, created_at FROM entries "
            "WHERE namespace = ? AND flagged = 0 AND expires_at > ?",
            (namespace, now),
        ).fetchall()
        if not rows:
            return LookupResult(hit=False, reason="cache is empty for this namespace")

        ids = [r["id"] for r in rows]
        mat = np.vstack([_blob_to_vec(r["embedding"]) for r in rows])
        # Vectors are unit-normalized at insert, so cosine == dot product.
        sims = mat @ query_vec
        order = np.argsort(-sims)
        best = int(order[0])
        s1 = float(sims[best])
        s2 = float(sims[order[1]]) if len(order) > 1 else 0.0
        margin = s1 - s2
        best_row = rows[best]

        if s1 < self.tau:
            return LookupResult(hit=False, similarity=s1, margin=margin,
                                reason=f"best similarity {s1:.4f} below tau {self.tau:.2f}")
        if margin < self.delta:
            return LookupResult(hit=False, similarity=s1, margin=margin,
                                reason=f"margin {margin:.4f} below delta {self.delta:.2f} (ambiguous)")

        self._bump_hit(ids[best])
        return LookupResult(
            hit=True, entry_id=ids[best], question=best_row["question"],
            answer=best_row["answer"], similarity=s1, margin=margin,
            age_seconds=now - best_row["created_at"], reason="semantic match",
        )

    def _bump_hit(self, entry_id: int) -> None:
        with _write_lock, self._conn:
            self._conn.execute("UPDATE entries SET hit_count = hit_count + 1 WHERE id = ?", (entry_id,))

    # ---------- insert / delete ----------

    def insert(self, namespace: str, question: str, normalized: str, vec: list[float],
               answer: str, chat_model: str, embed_model: str,
               prompt_tokens: int = 0, completion_tokens: int = 0,
               ttl_hours: float | None = None) -> int:
        now = _now()
        ttl = ttl_hours if ttl_hours is not None else self.ttl_hours
        blob = _vec_to_blob(vec)
        with _write_lock, self._conn:
            # Duplicate detection: concurrent misses on the same question all
            # try to insert; keep only the newest entry per exact hash so the
            # cache does not fill up with copies.
            self._conn.execute(
                "DELETE FROM entries WHERE namespace = ? AND exact_hash = ?",
                (namespace, exact_hash(normalized)),
            )
            cur = self._conn.execute(
                "INSERT INTO entries (namespace, question, normalized, exact_hash, embedding, "
                "dim, answer, chat_model, embed_model, prompt_tokens, completion_tokens, "
                "created_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (namespace, question, normalized, exact_hash(normalized), blob, len(vec),
                 answer, chat_model, embed_model, prompt_tokens, completion_tokens,
                 now, now + ttl * 3600),
            )
            return int(cur.lastrowid)

    def delete_entry(self, entry_id: int) -> bool:
        with _write_lock, self._conn:
            cur = self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            return cur.rowcount > 0

    def clear_all(self, confirm: bool) -> int:
        # Safety flag: clearing everything by accident would waste all the
        # stored answers, so the caller must pass confirm=True.
        if not confirm:
            raise ValueError("clear_all requires confirm=True")
        with _write_lock, self._conn:
            cur = self._conn.execute("DELETE FROM entries")
            return cur.rowcount

    def flag_wrong(self, entry_id: int | None, question: str, reason: str) -> None:
        """Flagged answers must never be served again, so we delete the entry
        and keep a feedback row for the false-positive count in analytics."""
        with _write_lock, self._conn:
            self._conn.execute(
                "INSERT INTO feedback (ts, entry_id, question, reason) VALUES (?,?,?,?)",
                (_now(), entry_id, question, reason),
            )
            if entry_id is not None:
                self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))

    # ---------- reads ----------

    def list_entries(self, namespace: str | None = None, search: str = "",
                     limit: int = 200) -> list[dict]:
        q = "SELECT id, namespace, question, answer, chat_model, embed_model, dim, " \
            "created_at, expires_at, hit_count, flagged, prompt_tokens, completion_tokens " \
            "FROM entries"
        args: list = []
        conds = []
        if namespace:
            conds.append("namespace = ?")
            args.append(namespace)
        if search:
            conds.append("question LIKE ?")
            args.append(f"%{search}%")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self._conn.execute(q, args)]

    def record_request(self, namespace: str, hit: bool, exact_hit: bool,
                       similarity: float | None, margin: float | None,
                       embedding_ms: float, retrieval_ms: float,
                       generation_ms: float, total_ms: float,
                       prompt_tokens: int | None, completion_tokens: int | None,
                       question: str) -> None:
        with _write_lock, self._conn:
            self._conn.execute(
                "INSERT INTO stats (ts, namespace, hit, exact_hit, similarity, margin, "
                "embedding_ms, retrieval_ms, generation_ms, total_ms, prompt_tokens, "
                "completion_tokens, question) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (_now(), namespace, int(hit), int(exact_hit), similarity, margin,
                 embedding_ms, retrieval_ms, generation_ms, total_ms,
                 prompt_tokens, completion_tokens, question),
            )

    def analytics(self, namespace: str | None = None) -> dict:
        conds, args = [], []
        if namespace:
            conds.append("namespace = ?")
            args.append(namespace)
        where = (" WHERE " + " AND ".join(conds)) if conds else ""

        rows = [dict(r) for r in self._conn.execute(
            f"SELECT hit, exact_hit, similarity, margin, embedding_ms, retrieval_ms, "
            f"generation_ms, total_ms, prompt_tokens, completion_tokens FROM stats{where}", args)]

        n = len(rows)
        hits = sum(r["hit"] for r in rows)
        exact_hits = sum(r["exact_hit"] for r in rows)

        def p95(vals: list[float]) -> float | None:
            if not vals:
                return None
            s = sorted(vals)
            idx = max(0, int(round(0.95 * len(s))) - 1)
            return s[idx]

        totals = [r["total_ms"] for r in rows if r["total_ms"] is not None]
        gen_tokens_saved = sum(
            (r["completion_tokens"] or 0) for r in rows if r["hit"] and r["completion_tokens"]
        )
        prompt_tokens_saved = sum(
            (r["prompt_tokens"] or 0) for r in rows if r["hit"] and r["prompt_tokens"]
        )
        feedback_count = self._conn.execute("SELECT COUNT(*) c FROM feedback").fetchone()["c"]
        entry_count = self._conn.execute(
            "SELECT COUNT(*) c FROM entries" + (" WHERE namespace = ?" if namespace else ""),
            [namespace] if namespace else [],
        ).fetchone()["c"]

        # HYPOTHETICAL external-API savings; local Ollama cost is zero.
        hyp_usd = (prompt_tokens_saved / 1e6 * config.PRICE_PER_1M_INPUT_TOKENS
                   + gen_tokens_saved / 1e6 * config.PRICE_PER_1M_OUTPUT_TOKENS)

        return {
            "requests": n,
            "hits": hits,
            "exact_hits": exact_hits,
            "semantic_hits": hits - exact_hits,
            "misses": n - hits,
            "hit_rate": (hits / n) if n else None,
            "miss_rate": ((n - hits) / n) if n else None,
            "mean_total_ms": (sum(totals) / len(totals)) if totals else None,
            "p95_total_ms": p95(totals),
            "mean_embedding_ms": _mean([r["embedding_ms"] for r in rows if r["embedding_ms"] is not None]),
            "mean_retrieval_ms": _mean([r["retrieval_ms"] for r in rows if r["retrieval_ms"] is not None]),
            "mean_generation_ms": _mean([r["generation_ms"] for r in rows if r["generation_ms"] is not None]),
            "llm_calls_avoided": hits,
            "false_positive_reports": feedback_count,
            "entries": entry_count,
            "tokens_saved": {"prompt": prompt_tokens_saved, "completion": gen_tokens_saved},
            "hypothetical_api_savings_usd": round(hyp_usd, 6),
            "actual_local_cost_usd": 0.0,
            "similarity_histogram": _histogram([r["similarity"] for r in rows if r["similarity"] is not None]),
        }

    def similarity_histogram_data(self, namespace: str | None = None) -> list[dict]:
        rows = self._conn.execute(
            "SELECT similarity FROM stats WHERE similarity IS NOT NULL"
            + (" AND namespace = ?" if namespace else ""),
            [namespace] if namespace else [],
        ).fetchall()
        return _histogram([r["similarity"] for r in rows])


def _mean(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def _histogram(sims: list[float], bins: int = 15) -> list[dict]:
    if not sims:
        return []
    out = []
    lo, hi = 0.5, 1.0  # similarities below 0.5 are never hits; keep the view useful
    width = (hi - lo) / bins
    counts = [0] * bins
    for s in sims:
        idx = min(bins - 1, max(0, int((s - lo) / width)))
        counts[idx] += 1
    for i, c in enumerate(counts):
        out.append({"from": round(lo + i * width, 3), "to": round(lo + (i + 1) * width, 3), "count": c})
    return out
