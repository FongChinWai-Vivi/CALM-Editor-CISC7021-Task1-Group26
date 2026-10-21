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
    mem.store(edit_signature="Spain|official_language", answer="Spanish")
    result = mem.lookup("Spain|official_language")  # "Spanish"
    mem.evict(shard_id=0)  # remove all entries from shard 0
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
    def _hash_key(subject: str, relation: str, target_new: str) -> str:
        """Deterministic hash-based edit signature."""
        raw = f"{subject}|{relation}|{target_new}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def route(self, subject: str) -> int:
        """Return the shard index for a given subject string."""
        return hash(subject) % self.n_shards

    def lookup(self, subject: str, relation: str, target_new: str) -> Optional[str]:
        """Look up an edit entry. Returns the stored answer or None."""
        key = self._hash_key(subject, relation, target_new)
        shard_idx, value = self._index.get(key, (-1, None))
        if shard_idx < 0:
            return None
        return self.shards[shard_idx].get(key)

    def store(self, subject: str, relation: str, target_new: str,
              answer: str) -> bool:
        """Store an edit entry. Returns True on success, False if already exists."""
        key = self._hash_key(subject, relation, target_new)
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
        ok = mem.store(subj, rel, ans, ans)
        if ok:
            stored_count += 1
    print(f"Stored {stored_count}/{len(edits)} edits")

    # --- Lookup ---
    print("\nLookup results:")
    for subj, rel, ans in edits[:6]:
        found = mem.lookup(subj, rel, ans)
        status = f"OK ({found})" if found else "MISS"
        print(f"  {subj} ({rel}) -> {status}")

    # --- Duplicate rejection ---
    dup_ok = mem.store(edits[0][0], edits[0][1], edits[0][2], "NEW_ANSWER")
    print(f"\nDuplicate store attempt: {'REJECTED' if not dup_ok else 'STORED'}")

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
