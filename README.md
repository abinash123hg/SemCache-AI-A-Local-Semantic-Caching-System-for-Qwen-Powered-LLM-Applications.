# SemCache AI

A **semantic cache** in front of a local LLM. Repeated or re-phrased questions
are answered from a SQLite cache instead of re-running the model, and every
lookup is scored so you can see exactly when a cache hit is trustworthy.

Everything runs on your own machine through **[Ollama](https://ollama.com/)** —
no API keys, no cloud calls, no network access at runtime.

**Highlights**

- **Streaming answers** — `POST /chat/stream` (SSE) streams the model output
  token by token; the UI shows what the cache is doing while it waits.
- **Bounded, self-cleaning store** — WAL-mode SQLite, TTL expiry, an entry cap
  with least-recently-used eviction, and near-duplicate merge so re-phrasings do
  not pile up.
- **Per-namespace tuning** — override `tau` / `delta` / `ttl` for one
  model+prompt+temperature combination without touching the global defaults.
- **Backup & restore** — export the whole cache to JSON and import it back
  (`/cache/export`, `/cache/import`).

---

## How a request is handled

For each incoming question the cache tries, in order:

1. **Normalise** the text (trim, collapse whitespace, lowercase for the exact key).
2. **Exact-hit** — a SHA-256 hash of the normalised text. Always safe.
3. **Embed** the question with `nomic-embed-text` and run a **cosine top-1** search.
4. **Semantic-hit** if both rules pass:
   - `score >= tau` (similarity threshold, default `0.90`), **and**
   - `score - second_best >= delta` (margin rule, default `0.02`) — the match must
     clearly beat the runner-up, not just edge past the threshold.
5. **Miss** → generate with `qwen2.5:3b`, then store the answer for next time.

Answers are scoped by **namespace**, which is derived from the system prompt +
temperature, so changing generation settings does not serve stale answers.

---

## Requirements

| Item | Detail |
|------|--------|
| OS | Windows 10/11 (the launcher is a `.bat`) |
| Python | 3.11 or newer |
| Ollama | installed and running on `127.0.0.1:11434` |
| Models | `qwen2.5:3b` (~1.9 GB), `nomic-embed-text` (~274 MB) |

Runtime Python deps are in `requirements.txt` (fastapi, uvicorn, pydantic, numpy, httpx).

---

## Quick start

Double-click **`start.bat`**. It will:

1. find a usable Python,
2. create `venv\` and install `requirements.txt` (only when the file changed),
3. start Ollama if it is not already running,
4. pull the two models if they are missing,
5. check ports, launch the API (`:8000`) and the UI (`:5500`) in their own windows,
6. open the UI in your browser.

Use **`stop.bat`** to shut both servers down.

### Manual start (no launcher)

```bat
venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then serve the UI from a second terminal:

```bat
venv\Scripts\python.exe -m http.server 5500 --directory frontend
```

UI: <http://127.0.0.1:5500>  ·  API docs: <http://127.0.0.1:8000/docs>

---

## Project layout

```
5b4870b1/
├─ start.bat / stop.bat        launcher + shutdown
├─ requirements.txt
├─ LICENSE                     MIT
├─ .github/workflows/ci.yml    runs the unit tests on push / PR
├─ backend/
│  ├─ main.py                  FastAPI routes
│  ├─ cache.py                 SQLite store + hit rules + analytics
│  ├─ embedding.py             vectors, cosine, normalisation
│  ├─ ollama_client.py         thin HTTP client for Ollama
│  ├─ benchmark.py             5 evaluation experiments
│  └─ config.py                thresholds, models, paths, prices
├─ frontend/                   index.html · app.js · style.css
├─ tests/test_cache.py         offline unit tests (no Ollama needed)
└─ data/
   ├─ semcache.db              cache + stats (created at runtime)
   ├─ benchmark_dataset.json   102 labelled pairs (v1.1)
   └─ results/                 experiment output (exp1…exp5)
```

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | backend status + which models are present |
| `POST` | `/chat` | ask a question (cached); returns answer, `hit`, similarity, timings |
| `POST` | `/chat/stream` | same decision as `/chat`, but streams the answer as Server-Sent Events |
| `GET` | `/cache` | list cache entries (optionally by namespace) |
| `DELETE` | `/cache/{id}` | delete one entry |
| `POST` | `/cache/clear` | clear the cache (requires `confirm: true`) |
| `GET` | `/cache/export` | download entries as JSON (backup / move machines) |
| `POST` | `/cache/import` | load entries from a previous export |
| `POST` | `/cache/cleanup` | sweep expired entries now |
| `POST` | `/feedback` | flag a wrong hit (bad entry is removed, counts as a miss) |
| `GET` | `/analytics` | hit rate, token/latency savings, similarity histogram |
| `GET` | `/config` · `POST` | read/update global `tau`, `delta`, `ttl`; `POST` with a `namespace` sets a per-namespace override |
| `GET` | `/config/namespaces` | list the per-namespace overrides that are set |
| `GET` | `/models/status` · `POST /models/pull` | model presence / pull |
| `POST` | `/benchmark/run` · `GET /benchmark/results` | run / read the experiments |

---

## Configuration

Edit `backend/config.py`, or use the **Settings** tab. Key values:

| Setting | Default | Meaning |
|---------|---------|---------|
| `SIMILARITY_THRESHOLD` (`tau`) | `0.90` | minimum cosine score for a semantic hit |
| `MARGIN_THRESHOLD` (`delta`) | `0.02` | how far the best match must beat the second best |
| `TTL_HOURS` | `24` | entry lifetime; expired rows are ignored then swept |
| `MAX_ENTRIES` | `5000` | hard size cap; past this, the least-recently-used entries are dropped |
| `NEAR_DUPLICATE_THRESHOLD` | `0.99` | on insert, a question this close to an existing one replaces it instead of adding a twin |
| `PRICE_PER_1M_INPUT_TOKENS` | `0.10` | used to turn "tokens saved" into a cost figure |
| `PRICE_PER_1M_OUTPUT_TOKENS` | `0.20` | as above |

`tau`, `delta` and `ttl` can also be set **per namespace** (one namespace = one
combination of chat model, embed model, system prompt and temperature) from the
**Settings** tab: leave a field blank to fall back to the global value.

`start.bat` keeps its own copy of the model names and ports — keep those in sync
with `config.py` if you change them.

The cache is stored in SQLite in **WAL mode** with `busy_timeout`, so concurrent
requests do not block each other or throw "database is locked".

---

## Tests

Offline unit tests need no model running — they use fake unit vectors, because
they test the hit rules, not the embeddings. **27 tests** cover exact/semantic
hits, the margin rule, namespace isolation, TTL, persistence, settings
validation, punctuation normalisation, near-duplicate merge, LRU eviction,
export/import and per-namespace overrides:

```bat
venv\Scripts\python.exe -m unittest tests.test_cache -v
```

CI runs the same suite on Python 3.11 and 3.12 (see `.github/workflows/ci.yml`).

### Live verification

The live path was checked against a running Ollama on this machine. With the
backend up:

```bat
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"What is the capital of France?\"}"
```

The first call is a miss (`source: "llm"`), and re-asking the same or a
re-phrased question returns `source: "cache"`. `server_smoke.log` /
`frontend_smoke.log` in the project root capture a full pass (`/health`,
`/chat`, `/cache`, `/analytics` all `200 OK`, Ollama reachable).

---

## Benchmark results

`backend/benchmark.py` runs five experiments against
`data/benchmark_dataset.json` — **102 hand-labelled pairs**:

| Label | Pairs | Expected |
|-------|-------|----------|
| `exact_duplicate` | 12 | hit |
| `paraphrase` | 35 | hit |
| `unrelated` | 19 | no hit |
| `ambiguous` | 10 | no hit (a hit would be a guess) |
| `tricky_lookalike` | 26 | no hit (e.g. "Java" vs "JavaScript") |

So 47 pairs *should* hit and 55 *should not*. Results below are the real
output from `data/results/` (12-core Intel, 15.6 GB RAM, Windows 11).

**Exp 1 — false positives** (`tau=0.90`, `delta=0.02`)

| | |
|---|---|
| True positives | 19 |
| False negatives | 28 |
| **False positives** | **5** |
| True negatives | 50 |
| Embedding time | 8.3 s for the whole pool |

**Exp 2 — threshold sweep** (tau 0.80 → 0.95). F1 peaks at **~0.73 around
tau ≈ 0.82**; the shipped default `0.90` trades recall for safety
(F1 ≈ 0.54 there) — far fewer wrong hits, more missed reuses.

**Exp 3 — margin rule** (`delta = 0` vs `0.02`). Counts are **identical**
(19 / 5 / 28 / 50): on this dataset every false positive already had a margin
above `0.02` (smallest FP margin ≈ 0.058), so the margin rule removed nothing
here. It stays enabled as cheap insurance for closer look-alikes.

**Exp 4 — TTL expiry.** Entry inserted, clock advanced past the TTL, entry
correctly stops being served and is swept. **Passed.**

**Exp 5 — baseline comparison** (10 questions)

| Strategy | LLM calls |
|----------|-----------|
| No cache | 10 |
| Exact-match cache only | 10 (0 semantic hits) |
| **Semantic cache** | **6** (4 hits) |

i.e. the semantic layer avoided 4 model calls on re-phrased questions that an
exact-match cache would have missed entirely.

---

## Storage notes

- Models are pulled once by Ollama and live outside the project (~2.2 GB total).
- `venv\` holds the Python deps; it is **git-ignored** so it never bloats a repo.
- `data/semcache.db` grows with the cache and is git-ignored too.
- Cache entries can be removed any time from the **Cache** tab or `/cache/clear`.
