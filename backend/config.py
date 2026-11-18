"""Central configuration. Ports, model names and cache thresholds live here
and nowhere else, so start.bat and the code always agree."""

from pathlib import Path

# Paths are relative to the project root so nothing is machine-specific.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "semcache.db"
BENCHMARK_DATASET_PATH = DATA_DIR / "benchmark_dataset.json"
RESULTS_DIR = DATA_DIR / "results"

# Ports. start.bat uses these same numbers (8000 backend, 5500 frontend).
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_PORT = 5500

# CORS: only the local frontend page is allowed, nothing broader.
ALLOWED_ORIGINS = [
    f"http://127.0.0.1:{FRONTEND_PORT}",
    f"http://localhost:{FRONTEND_PORT}",
]

# Ollama
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"
REQUEST_TIMEOUT_SECONDS = 120.0  # local generation can be slow on CPU

# Cache defaults (changeable at runtime via /config)
SIMILARITY_THRESHOLD = 0.90   # tau: minimum cosine similarity for a hit
MARGIN_THRESHOLD = 0.02       # delta: min gap between best and second-best
TTL_HOURS = 24                # entries older than this are not served

# Input limits. Long questions cost embedding time and are rarely cache hits.
MAX_QUESTION_CHARS = 2000
MAX_SYSTEM_PROMPT_CHARS = 1000

# Generation defaults
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question clearly and concisely."
)

# Hypothetical external-API prices for the savings estimate only.
# These are NOT what we pay; Ollama runs locally for free.
PRICE_PER_1M_INPUT_TOKENS = 0.10   # USD, hypothetical
PRICE_PER_1M_OUTPUT_TOKENS = 0.20  # USD, hypothetical
