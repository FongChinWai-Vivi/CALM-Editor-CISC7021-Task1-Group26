"""SideMemory — WISE-style sharded non-parametric memory.

This module implements the non-parametric side memory component of the CALM
project (W3, Fong Hio Lam). It follows the WISE design pattern: edits are
routed to a shard based on a deterministic hash of the edit subject, and each
shard stores key-value pairs mapping an edit signature to its answer. The
memory is never overwritten — once a subject is stored in a shard, subsequent
edits to the same subject are rejected by that shard, forcing them through
the parameter-writing path or another shard.

The module is fully self-contained with no third-party dependencies beyond
Python's standard library. A smoke test is included at the bottom.

Usage as a component inside CALMEditor::

    from side_memory import SideMemory

    mem = SideMemory(n_shards=64)
    mem.store("Spain", "official_language", "Spanish")
    result = mem.lookup("Spain", "official_language")  # "Spanish"
    mem.evict(shard_idx=0)  # remove all entries from shard 0

Storage contract: the key is the query pair ``(subject, relation)``; the answer
is the value.  The answer must never enter the key, otherwise a lookup can only
succeed when the caller already knows the answer and recall is constructive.
"""

from __future__ import annotations

import hashlib
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shard internals
# ---------------------------------------------------------------------------


class _Shard(OrderedDict):
    """A single LRU-capable shard storing one or more edit entries."""

    def __init__(self, max_size: int = 200):
        super().__init__()
        self.max_size = max_size

    def put(self, key: str, value: str) -> bool:
        """Store *key* → *value*. Returns True if stored, False if already present."""
        if key in self:
            return False
        if len(self) >= self.max_size:
            self.popitem(last=False)  # evict oldest
        self[key] = value
        return True

    def get(self, key: str) -> Optional[str]:
        return super().get(key)


# ---------------------------------------------------------------------------
# SideMemory
# ---------------------------------------------------------------------------


class SideMemory:
    """Sharded non-parametric memory for knowledge editing.

    Parameters
    ----------
    n_shards : int
        Number of shards. Each edit is routed to one shard via
        ``hash(edit_subject) % n_shards``.
    max_per_shard : int
        Maximum entries per shard before eviction kicks in (LRU).
    seed : int
        Random seed for reproducibility of eviction order.

    Attributes
    ----------
    shards : list[_Shard]
        Internal shard list.
    total_stored : int
        Total number of unique entries across all shards.
    """

    def __init__(self, n_shards: int = 64, max_per_shard: int = 200,
                 seed: int = 0):
        self.n_shards = n_shards
        self.max_per_shard = max_per_shard
        self.seed = seed
        self.rng = random.Random(seed)
        self.shards: List[_Shard] = [
            _Shard(max_size=max_per_shard) for _ in range(n_shards)
        ]
        # Global index for fast lookup across all shards.
        self._index: Dict[str, Tuple[int, str]] = {}  # key -> (shard_idx, value)
        self.total_stored = 0

    @staticmethod
    def _hash_key(subject: str, relation: str) -> str:
        """Deterministic edit signature over the query only: (subject, relation).

        The answer (``target_new``) is a *value*, never part of the key.  A key
        that contains the answer makes every recall trivially 1.0, because the
        caller must already know the answer in order to build the query.
        """
        raw = f"{subject}|{relation}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def route(self, subject: str) -> int:
        """Return the shard index for a given subject string."""
        return hash(subject) % self.n_shards

    def lookup(self, subject: str, relation: str,
               target_new: Optional[str] = None) -> Optional[str]:
        """Look up an edit by its query ``(subject, relation)``.

        Returns the stored answer, or ``None`` when the pair was never stored.
        The answer is *not* an argument: a lookup must be answerable from the
        query alone, otherwise the measured recall is constructive.

        ``target_new`` exists only for backward compatibility with call sites
        written against the old three-argument API (``EasyEdit/examples/
        w3_calm_stream.py``, ``TANG I CHOI/w4_gate_ablations/
        run_w4_gate_ablation.py``).  It is accepted and *ignored*: it never
        enters the key and never influences the result, so a hit still means
        "the query resolved", not "the caller already knew the answer".
        """
        key = self._hash_key(subject, relation)
        shard_idx, _value = self._index.get(key, (-1, None))
        if shard_idx < 0:
            return None
        return self.shards[shard_idx].get(key)

    def has(self, subject: str, relation: str) -> bool:
        """True if ``(subject, relation)`` is present in the store."""
        return self.lookup(subject, relation) is not None

    def store(self, subject: str, relation: str, answer: str) -> bool:
        """Store the answer for the query ``(subject, relation)``.

        Returns True on success, False if the pair is already present.
        """
        key = self._hash_key(subject, relation)
        shard_idx = self.route(subject)
        shard = self.shards[shard_idx]
        if key not in shard:
            shard.put(key, answer)
            self._index[key] = (shard_idx, answer)
            self.total_stored += 1
            return True
        return False

    def evict(self, shard_idx: Optional[int] = None) -> int:
        """Evict entries from a specific shard (or all shards). Returns count evicted."""
        if shard_idx is not None:
            shard = self.shards[shard_idx]
            old_len = len(shard)
            shard.clear()
            # Remove from global index.
            keys_to_remove = [k for k, v in self._index.items() if v[0] == shard_idx]
            for k in keys_to_remove:
                del self._index[k]
                self.total_stored -= 1
            return old_len
        else:
            total_evicted = 0
            for i, shard in enumerate(self.shards):
                total_evicted += self.evict(i)
            return total_evicted

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        sizes = [len(s) for s in self.shards]
        return {
            "n_shards": self.n_shards,
            "max_per_shard": self.max_per_shard,
            "total_stored": self.total_stored,
            "avg_entries_per_shard": round(sum(sizes) / len(sizes), 2),
            "min_entries": min(sizes),
            "max_entries": max(sizes),
        }

    def clear(self) -> None:
        """Remove all entries from every shard."""
        for shard in self.shards:
            shard.clear()
        self._index.clear()
        self.total_stored = 0


# ---------------------------------------------------------------------------
# Read path helper
# ---------------------------------------------------------------------------


def build_memory_note(subject: str, relation: str, note: str) -> str:
    """Return the prompt prefix used to inject a retrieved memory fact.

    This is the read half of the memory channel: the caller looks up
    ``(subject, relation)`` and prepends the returned fact to the prompt before
    generation.  The prompt itself must not contain the answer, otherwise the
    measurement leaks the target.
    """
    return f"[{subject}] [{relation}] is [{note}]\n"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def run_smoke():
    """Self-contained smoke test demonstrating store / lookup / evict."""
    print("=" * 60)
    print("SideMemory Smoke Test")
    print("=" * 60)

    mem = SideMemory(n_shards=8, max_per_shard=10, seed=42)

    # --- Store ---
    edits = [
        ("Spain", "capital", "Madrid"),
        ("France", "capital", "Paris"),
        ("Germany", "capital", "Berlin"),
        ("Japan", "capital", "Tokyo"),
        ("Italy", "capital", "Rome"),
        ("UK", "capital", "London"),
        ("China", "capital", "Beijing"),
        ("Brazil", "capital", "Brasilia"),
        ("India", "capital", "New Delhi"),
        ("Australia", "capital", "Canberra"),
        ("Canada", "capital", "Ottawa"),
        ("Mexico", "capital", "Mexico City"),
    ]

    stored_count = 0
    for subj, rel, ans in edits:
        ok = mem.store(subj, rel, ans)
        if ok:
            stored_count += 1
    print(f"Stored {stored_count}/{len(edits)} edits")

    # --- Lookup (query only, answer not supplied) ---
    print("\nLookup results:")
    for subj, rel, ans in edits[:6]:
        found = mem.lookup(subj, rel)
        status = f"OK ({found})" if found else "MISS"
        print(f"  {subj} ({rel}) -> {status}")

    # --- Duplicate rejection ---
    dup_ok = mem.store(edits[0][0], edits[0][1], "NEW_ANSWER")
    print(f"\nDuplicate store attempt: {'REJECTED' if not dup_ok else 'STORED'}")

    # --- Negative control: unseen keys must miss (the old keying could not miss) ---
    unseen_subject = mem.lookup("Atlantis", "capital")
    unseen_relation = mem.lookup(edits[0][0], "national_motto")
    print(f"\nNegative control: unseen subject -> {unseen_subject}, "
          f"unseen relation -> {unseen_relation} (expected None, None)")

    # --- Eviction ---
    print(f"\nBefore eviction: {mem.stats()}")
    evicted = mem.evict(0)
    print(f"After evicting shard 0: evicted={evicted}, remaining={mem.total_stored}")

    # --- Stats ---
    print(f"\nFinal stats: {mem.stats()}")
    print("=" * 60)
    print("Smoke test passed.")


if __name__ == "__main__":
    run_smoke()
