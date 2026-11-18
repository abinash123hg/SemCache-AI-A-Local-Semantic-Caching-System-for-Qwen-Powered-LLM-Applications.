"""Offline unit tests for the cache logic. No Ollama needed: we use fake
unit vectors, because these tests are about the hit rules, namespaces, TTL
and persistence, not about embeddings.

Run from the project root:
    venv\\Scripts\\python.exe -m unittest tests.test_cache -v

Live tests that need Ollama (exact/semantic hit through /chat, etc.) are
run separately and reported in the README with real output.
"""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from backend import cache as cache_mod


def unit(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float64)
    return (arr / np.linalg.norm(arr)).tolist()


class CacheTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = cache_mod.Cache(Path(self.tmp.name) / "test.db")
        self.ns = cache_mod.namespace_of("chat-model", "embed-model", "sys prompt", 0.7)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def insert(self, question: str, vec: list[float], ns: str | None = None,
               ttl_hours: float | None = None, answer: str = "ans") -> int:
        return self.db.insert(
            ns or self.ns, question, question.lower(), unit(vec), answer,
            "chat-model", "embed-model", prompt_tokens=10, completion_tokens=20,
            ttl_hours=ttl_hours,
        )


class TestExactMatch(CacheTestBase):
    def test_exact_hit_after_normalization(self):
        # lookup normalizes: case + whitespace must not matter
        self.insert("How do I sort a list?", [1, 0, 0, 0])
        res = self.db.lookup(self.ns, "how do i sort a list?", None)
        self.assertTrue(res.hit)
        self.assertTrue(res.exact)
        self.assertEqual(res.answer, "ans")

    def test_exact_miss_on_different_text(self):
        self.insert("How do I sort a list?", [1, 0, 0, 0])
        res = self.db.lookup(self.ns, "how do I sort a dictionary?", None)
        self.assertFalse(res.hit)


class TestSemanticMatch(CacheTestBase):
    def test_hit_above_tau(self):
        self.insert("capital of France", [1, 0, 0, 0])
        q = unit([0.99, 0.141, 0, 0])  # cos ~0.99 with stored
        res = self.db.lookup(self.ns, "which city is the capital of france", np.array(q))
        self.assertTrue(res.hit)
        self.assertGreaterEqual(res.similarity, self.db.tau)

    def test_miss_below_tau(self):
        self.insert("capital of France", [1, 0, 0, 0])
        q = unit([0.5, 0.866, 0, 0])  # cos = 0.5
        res = self.db.lookup(self.ns, "how do I bake a cake", np.array(q))
        self.assertFalse(res.hit)
        self.assertIn("below tau", res.reason)

    def test_margin_rule_blocks_ambiguous(self):
        # two entries almost equally close: high similarity but tiny margin
        self.insert("sort ascending", [1, 0, 0, 0])
        self.insert("sort descending", [0.999, 0.045, 0, 0])
        q = unit([1, 0.01, 0, 0])
        res = self.db.lookup(self.ns, "how do I sort", np.array(q))
        self.assertFalse(res.hit)
        self.assertIn("margin", res.reason)

    def test_empty_cache_misses(self):
        res = self.db.lookup(self.ns, "anything", np.array(unit([1, 0, 0, 0])))
        self.assertFalse(res.hit)
        self.assertIn("empty", res.reason)


class TestNamespace(CacheTestBase):
    def test_no_reuse_across_namespaces(self):
        other_ns = cache_mod.namespace_of("chat-model", "embed-model", "DIFFERENT prompt", 0.7)
        self.insert("capital of France", [1, 0, 0, 0])
        q = unit([1, 0, 0, 0])
        res_other = self.db.lookup(other_ns, "capital of france", np.array(q))
        self.assertFalse(res_other.hit, "entry from another namespace must never be served")
        res_same = self.db.lookup(self.ns, "capital of france", np.array(q))
        self.assertTrue(res_same.hit)

    def test_namespace_hash_stable_and_distinct(self):
        a = cache_mod.namespace_of("m", "e", "p", 0.7)
        b = cache_mod.namespace_of("m", "e", "p", 0.7)
        c = cache_mod.namespace_of("m", "e", "p", 0.8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class TestTTL(CacheTestBase):
    def test_expired_entry_not_served(self):
        self.insert("capital of France", [1, 0, 0, 0], ttl_hours=1 / 3600)  # ~1 second
        q = np.array(unit([1, 0, 0, 0]))
        res_now = self.db.lookup(self.ns, "capital of france", q)
        self.assertTrue(res_now.hit, "entry should hit before it expires")
        time.sleep(1.2)
        res = self.db.lookup(self.ns, "capital of france", q)
        self.assertFalse(res.hit, "expired entry must not be served")
        # exact lookup must also ignore it
        res_exact = self.db.lookup(self.ns, "capital of france", None)
        self.assertFalse(res_exact.hit)


class TestDeleteClearFlag(CacheTestBase):
    def test_delete_entry(self):
        eid = self.insert("q1", [1, 0, 0, 0])
        self.assertTrue(self.db.delete_entry(eid))
        self.assertFalse(self.db.delete_entry(eid))  # second delete finds nothing

    def test_clear_requires_confirm(self):
        self.insert("q1", [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            self.db.clear_all(confirm=False)
        removed = self.db.clear_all(confirm=True)
        self.assertEqual(removed, 1)

    def test_flag_removes_entry_from_rotation(self):
        eid = self.insert("capital of France", [1, 0, 0, 0])
        self.db.flag_wrong(eid, "capital of France", "wrong answer")
        res = self.db.lookup(self.ns, "capital of france", np.array(unit([1, 0, 0, 0])))
        self.assertFalse(res.hit)
        fb = self.db._conn.execute("SELECT COUNT(*) c FROM feedback").fetchone()["c"]
        self.assertEqual(fb, 1)


class TestPersistence(CacheTestBase):
    def test_survives_reopen(self):
        db_path = Path(self.tmp.name) / "persist.db"
        db1 = cache_mod.Cache(db_path)
        ns = cache_mod.namespace_of("m", "e", "p", 0.7)
        db1.insert(ns, "capital of France", "capital of france", unit([1, 0, 0, 0]),
                   "Paris", "m", "e")
        db1.update_settings(tau=0.85)
        db1.close()  # simulate process exit; Windows keeps the file locked otherwise
        db2 = cache_mod.Cache(db_path)
        self.assertEqual(db2.tau, 0.85, "settings must persist")
        res = db2.lookup(ns, "capital of france", None)
        self.assertTrue(res.hit, "entries must persist across restarts")
        self.assertEqual(res.answer, "Paris")
        db2.close()


class TestSettingsValidation(CacheTestBase):
    def test_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            self.db.update_settings(tau=1.5)
        with self.assertRaises(ValueError):
            self.db.update_settings(ttl_hours=-1)

    def test_update_round_trip(self):
        out = self.db.update_settings(tau=0.93, delta=0.01, ttl_hours=6)
        self.assertEqual(out, {"tau": 0.93, "delta": 0.01, "ttl_hours": 6.0})


if __name__ == "__main__":
    unittest.main()
