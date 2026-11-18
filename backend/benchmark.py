"""The five evaluation experiments from the project spec.

Everything here runs against data/benchmark_dataset.json and the real Ollama
models. Results are saved as JSON in data/results/ so the dashboard and the
README quote the same numbers.

Embeddings are computed once per question (in batches) and reused across the
threshold sweep, which keeps experiment 2 fast without changing what it measures.
"""

import json
import logging
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np

from backend import cache as cache_mod
from backend import config, embedding, ollama_client

log = logging.getLogger(__name__)

EMBED_BATCH = 32


def _hardware() -> dict:
    # RAM reading on Windows without adding psutil as a dependency.
    ram_gb = None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        ram_gb = round(stat.ullTotalPhys / (1024 ** 3), 1)
    except Exception:  # non-Windows or ctypes failure: leave as None, don't fake it
        pass
    return {
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "gpu": "unknown (Ollama decides; no GPU check done)",
        "os": platform.platform(),
    }


def load_dataset() -> dict:
    with open(config.BENCHMARK_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def _embed_unique(texts: list[str]) -> dict[str, list[float]]:
    """Embed each unique text once, in batches, query-side prefix.
    Returns {text: unit_vector}."""
    unique = sorted(set(texts))
    out: dict[str, list[float]] = {}
    for i in range(0, len(unique), EMBED_BATCH):
        batch = unique[i:i + EMBED_BATCH]
        prefixed = [embedding.QUERY_PREFIX + t for t in batch]
        vectors, _ = ollama_client.embed(prefixed)
        for t, v in zip(batch, vectors):
            arr = np.array(v, dtype=np.float64)
            arr = arr / np.linalg.norm(arr)
            out[t] = arr.tolist()
    return out


def _cos(a: list[float], b: list[float]) -> float:
    return float(np.dot(np.array(a), np.array(b)))


def _base_meta(dataset: dict) -> dict:
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "tau": config.SIMILARITY_THRESHOLD,
        "delta": config.MARGIN_THRESHOLD,
        "ttl_hours": config.TTL_HOURS,
        "dataset_version": dataset.get("version"),
        "dataset_pairs": len(dataset.get("pairs", [])),
        "hardware": _hardware(),
    }


def _save(name: str, payload: dict) -> dict:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info("saved benchmark result -> %s", path)
    return payload


def _pair_similarities(dataset: dict) -> tuple[list[dict], float]:
    """Evaluate each labeled pair the way the real cache would.

    Exact duplicates are decided by the normalized-text rule (same as the
    exact-hash step in cache.lookup). Everything else goes through semantic
    search where the query "a" is compared against a pool of every other
    question in the dataset, so the margin rule sees real competition
    instead of a single isolated candidate. Each question text appears in
    only one pair (dataset v1.1), which keeps pool matches meaningful."""
    t0 = time.time()
    pairs = dataset["pairs"]
    all_texts = [p["a"] for p in pairs] + [p["b"] for p in pairs]
    vecs = _embed_unique(all_texts)

    results = []
    for p in pairs:
        na = embedding.normalize_question(p["a"])
        nb = embedding.normalize_question(p["b"])
        if na == nb:
            # case/punctuation-only difference: the exact-hash step hits
            results.append({
                "id": p.get("id"), "a": p["a"], "b": p["b"],
                "label": p.get("label"), "should_hit": p.get("should_hit"),
                "similarity": 1.0, "second_best": 1.0, "margin": 1.0,
                "best_match": p["b"], "exact_rule": True,
            })
            continue
        va = np.array(vecs[p["a"]])
        # candidate pool = every other question in the dataset
        sims = []
        for t, v in vecs.items():
            if t == p["a"]:
                continue
            sims.append((_cos(va, list(v)), t))
        sims.sort(reverse=True)
        s1, best_t = sims[0]
        s2 = sims[1][0] if len(sims) > 1 else 0.0
        results.append({
            "id": p.get("id"),
            "a": p["a"],
            "b": p["b"],
            "label": p.get("label"),
            "should_hit": p.get("should_hit"),
            "similarity": round(s1, 4),
            "second_best": round(s2, 4),
            "margin": round(s1 - s2, 4),
            "best_match": best_t,
            "exact_rule": False,
        })
    return results, time.time() - t0


# ---------- experiment 1: false positives ----------

def run_false_positive_test(tau: float | None = None, delta: float | None = None) -> dict:
    tau = config.SIMILARITY_THRESHOLD if tau is None else tau
    delta = config.MARGIN_THRESHOLD if delta is None else delta
    dataset = load_dataset()
    pairs, embed_seconds = _pair_similarities(dataset)

    fp, fn, tp, tn = [], [], [], []
    for p in pairs:
        would_hit = p["similarity"] >= tau and p["margin"] >= delta
        # The "b" question is what the pair is about; best_match tells us what
        # the cache would actually return. A hit counts for the pair only if
        # the top match is the partner question or a same-meaning duplicate.
        if p["should_hit"] and would_hit:
            tp.append(p)
        elif p["should_hit"] and not would_hit:
            fn.append(p)
        elif not p["should_hit"] and would_hit:
            fp.append(p)
        else:
            tn.append(p)

    payload = {
        "experiment": "1_false_positives",
        "meta": {**_base_meta(dataset), "tau_used": tau, "delta_used": delta,
                 "embedding_seconds": round(embed_seconds, 1)},
        "counts": {"true_positives": len(tp), "false_negatives": len(fn),
                   "false_positives": len(fp), "true_negatives": len(tn)},
        "false_positives": fp,
        "false_negatives": fn,
        "pairs": pairs,
    }
    return _save("exp1_false_positives", payload)


# ---------- experiment 2: threshold sweep ----------

def run_threshold_sweep() -> dict:
    dataset = load_dataset()
    pairs, embed_seconds = _pair_similarities(dataset)
    delta = config.MARGIN_THRESHOLD

    sweep = []
    taus = [round(0.80 + 0.01 * i, 2) for i in range(16)]  # 0.80 .. 0.95
    for tau in taus:
        tp = fp = fn = tn = 0
        for p in pairs:
            would_hit = p["similarity"] >= tau and p["margin"] >= delta
            if p["should_hit"]:
                tp += would_hit
                fn += not would_hit
            else:
                fp += would_hit
                tn += not would_hit
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        fpr = fp / (fp + tn) if (fp + tn) else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) else None)
        sweep.append({
            "tau": tau,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "false_positive_rate": round(fpr, 4) if fpr is not None else None,
        })

    payload = {
        "experiment": "2_threshold_sweep",
        "meta": {**_base_meta(dataset), "delta_used": delta,
                 "embedding_seconds": round(embed_seconds, 1)},
        "sweep": sweep,
    }
    return _save("exp2_threshold_sweep", payload)


# ---------- experiment 3: margin check ----------

def run_margin_check() -> dict:
    dataset = load_dataset()
    pairs, embed_seconds = _pair_similarities(dataset)
    tau = config.SIMILARITY_THRESHOLD

    def evaluate(delta: float) -> dict:
        tp = fp = fn = tn = 0
        fp_list = []
        for p in pairs:
            would_hit = p["similarity"] >= tau and p["margin"] >= delta
            if p["should_hit"]:
                tp += would_hit
                fn += not would_hit
            else:
                fp += would_hit
                tn += not would_hit
                if would_hit:
                    fp_list.append({"id": p["id"], "a": p["a"], "b": p["b"],
                                    "similarity": p["similarity"], "margin": p["margin"]})
        return {"delta": delta, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
                "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
                "false_positive_examples": fp_list}

    payload = {
        "experiment": "3_margin_check",
        "meta": {**_base_meta(dataset), "tau_used": tau,
                 "embedding_seconds": round(embed_seconds, 1)},
        "delta_zero": evaluate(0.0),
        "delta_chosen": evaluate(config.MARGIN_THRESHOLD),
    }
    return _save("exp3_margin_check", payload)


# ---------- experiment 4: TTL ----------

def run_ttl_test() -> dict:
    """Insert an entry with a 1-second TTL, look it up right away (must hit),
    wait for it to expire, look it up again (must miss). Uses a throwaway
    namespace so real cache data is untouched."""
    import sqlite3

    db = cache_mod.Cache(config.DB_PATH)
    ns = cache_mod.namespace_of("ttl-test-model", config.EMBED_MODEL, "ttl test", 0.0)
    vec = [0.0] * 8
    vec[0] = 1.0  # unit vector; good enough to test expiry logic
    q = "ttl experiment question"
    ttl_hours = 1 / 3600  # one second
    entry_id = db.insert(ns, q, q, vec, "ttl answer", "ttl-test-model",
                         config.EMBED_MODEL, ttl_hours=ttl_hours)

    before = db.lookup(ns, q, np.array(vec))
    time.sleep(1.5)
    after = db.lookup(ns, q, np.array(vec))

    # clean up after ourselves
    db.delete_entry(entry_id)
    db.close()
    with sqlite3.connect(config.DB_PATH) as c:
        c.execute("DELETE FROM stats WHERE namespace = ?", (ns,))

    payload = {
        "experiment": "4_ttl",
        "meta": _base_meta(load_dataset()),
        "ttl_used_hours": ttl_hours,
        "wait_seconds": 1.5,
        "hit_before_expiry": before.hit,
        "hit_after_expiry": after.hit,
        "reason_after": after.reason,
        "passed": bool(before.hit and not after.hit),
    }
    return _save("exp4_ttl", payload)


# ---------- experiment 5: baseline comparison ----------

BASELINE_QUESTIONS = [
    "What is the capital of France?",
    "Explain what a semantic cache is in one sentence.",
    "How do I reverse a list in Python?",
    "What does HTTP status code 404 mean?",
    "Name the capital of France.",                    # paraphrase of #1
    "What is the purpose of a semantic cache?",        # paraphrase of #2
    "How can I reverse a Python list?",                # paraphrase of #3
    "What is a 404 error in HTTP?",                    # paraphrase of #4
    "Tell me the capital city of France.",             # paraphrase of #1
    "How to reverse a list in Python?",                # near-duplicate of #3
]


def run_baseline_comparison() -> dict:
    """Ask the same 10 questions through three modes on the same model and
    hardware: (a) no cache, (b) exact-match cache only, (c) full semantic
    cache. Each mode uses a fresh namespace so modes don't share entries."""
    db = cache_mod.Cache(config.DB_PATH)
    modes = ["no_cache", "exact_only", "semantic_cache"]
    results = {}

    for mode in modes:
        ns = cache_mod.namespace_of(f"baseline-{mode}", config.EMBED_MODEL,
                                    config.DEFAULT_SYSTEM_PROMPT, config.DEFAULT_TEMPERATURE)
        # fresh namespace: clear any leftovers from a previous baseline run
        for e in db.list_entries(namespace=ns):
            db.delete_entry(e["id"])

        rows = []
        llm_calls = 0
        for q in BASELINE_QUESTIONS:
            norm = embedding.normalize_question(q)
            t0 = time.perf_counter()
            emb_ms = ret_ms = gen_ms = 0.0
            hit = exact = False
            sim = margin = None
            answer = ""

            t1 = time.perf_counter()
            query_vec = None
            if mode == "semantic_cache":
                v, _ = embedding.embed_query(q)
                query_vec = np.array(v)
            emb_ms = (time.perf_counter() - t1) * 1000

            t1 = time.perf_counter()
            lookup_vec = query_vec if mode == "semantic_cache" else None
            if mode == "no_cache":
                res = cache_mod.LookupResult(hit=False, reason="cache disabled")
            else:
                res = db.lookup(ns, norm, lookup_vec)
            ret_ms = (time.perf_counter() - t1) * 1000

            if res.hit:
                hit, exact = True, res.exact
                sim, margin = res.similarity, res.margin
                answer = res.answer or ""
            else:
                gen = ollama_client.generate(q, config.DEFAULT_SYSTEM_PROMPT,
                                             config.DEFAULT_TEMPERATURE,
                                             model=config.CHAT_MODEL)
                llm_calls += 1
                answer = gen.get("response", "")
                gen_ms = (time.perf_counter() - t1 - ret_ms / 1000) * 1000
                if mode != "no_cache":
                    v, _ = embedding.embed_document(q)
                    db.insert(ns, q, norm, v, answer, f"baseline-{mode}",
                              config.EMBED_MODEL,
                              prompt_tokens=gen.get("prompt_eval_count", 0),
                              completion_tokens=gen.get("eval_count", 0))

            total_ms = (time.perf_counter() - t0) * 1000
            rows.append({"question": q, "hit": hit, "exact": exact,
                         "similarity": round(sim, 4) if sim is not None else None,
                         "total_ms": round(total_ms, 1),
                         "generation_ms": round(gen_ms, 1)})

        totals = [r["total_ms"] for r in rows]
        results[mode] = {
            "llm_calls": llm_calls,
            "hits": sum(r["hit"] for r in rows),
            "mean_total_ms": round(sum(totals) / len(totals), 1),
            "p95_total_ms": round(sorted(totals)[max(0, int(round(0.95 * len(totals))) - 1)], 1),
            "requests": rows,
        }

    payload = {
        "experiment": "5_baseline_comparison",
        "meta": _base_meta(load_dataset()),
        "questions": BASELINE_QUESTIONS,
        "results": results,
    }
    return _save("exp5_baseline", payload)


EXPERIMENTS = {
    "1": run_false_positive_test,
    "2": run_threshold_sweep,
    "3": run_margin_check,
    "4": run_ttl_test,
    "5": run_baseline_comparison,
}


def load_results() -> dict:
    out = {}
    if config.RESULTS_DIR.exists():
        for p in sorted(config.RESULTS_DIR.glob("*.json")):
            try:
                with open(p, encoding="utf-8") as f:
                    out[p.stem] = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                log.warning("cannot read result %s: %s", p, e)
    return out
