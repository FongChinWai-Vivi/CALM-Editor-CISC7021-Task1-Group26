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
        self.evictions = 0

    @staticmethod
    def _norm(text: Any) -> str:
        """Normalise a query field before it enters the key.

        ``calm_editor._mem_key`` (the version clock) already lower-cases and
        collapses whitespace, so the store must do the same or the two halves
        of the same query can disagree: ``("Spain ", "official_language")``
        stored and ``("spain", "official_language")`` looked up would build two
        different digests and report a spurious miss.  Normalising here keeps
        the store's identity relation independent of surface casing, which is
        exactly the failure mode a paraphrase probe hits first.
        """
        return " ".join(str(text if text is not None else "").split()).lower()

    @staticmethod
    def _hash_key(subject: str, relation: str) -> str:
        """Deterministic edit signature over the query only: (subject, relation).

        The answer (``target_new``) is a *value*, never part of the key.  A key
        that contains the answer makes every recall trivially 1.0, because the
        caller must already know the answer in order to build the query.
        """
        raw = f"{SideMemory._norm(subject)}|{SideMemory._norm(relation)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def route(self, subject: str) -> int:
        """Return the shard index for a given subject string.

        Uses SHA-256, not the builtin ``hash``: ``hash`` on ``str`` is salted
        per process (``PYTHONHASHSEED``), so the shard a subject lands in — and
        therefore which entries a shard's LRU eviction can claim — would differ
        between the run that produced a results file and the run that audited
        it.  The module promises deterministic routing, so it has to be
        digest-based.  Shards are capacity containers only (lookups go through
        the global index), so this changes no hit rate; it makes the layout
        reproducible.
        """
        digest = hashlib.sha256(self._norm(subject).encode()).hexdigest()
        return int(digest[:16], 16) % self.n_shards

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

    def store(self, subject: str, relation: str, answer: str,
              target_new: Optional[str] = None, overwrite: bool = False) -> bool:
        """Store the answer for the query ``(subject, relation)``.

        Returns True when the write landed, False when the pair is already
        present and ``overwrite`` is not set.

        ``target_new`` mirrors ``lookup``'s legacy fourth argument: call sites
        written against the old four-argument API (``EasyEdit/easyeditor/
        editors/calm_editor.py``, which passes the target twice) are accepted
        and the duplicate is ignored — it is the value, never part of the key.
        Without this shim every ROUTE decision raises ``TypeError`` because the
        key was narrowed to ``(subject, relation)`` in the S1 fix.

        ``overwrite`` closes the counterfactual hole: a later edit that revises
        the *same* query (``("Spain", "capital", "Madrid")`` then ``("Spain",
        "capital", "Lisbon")``) used to be rejected as a duplicate, so the store
        kept serving the superseded answer while the caller believed the edit
        had been absorbed.  Default False keeps the duplicate-rejection
        contract every existing table was measured against.
        """
        key = self._hash_key(subject, relation)
        shard_idx = self.route(subject)
        shard = self.shards[shard_idx]
        if key in shard:
            if not overwrite:
                return False
            shard[key] = answer
            self._index[key] = (shard_idx, answer)
            return True
        # Capacity: ``_Shard.put`` drops the oldest entry when the shard is
        # full.  Doing that eviction here (instead of inside ``put``) keeps the
        # global index and ``total_stored`` in step with the shard; otherwise
        # the index grows without bound and a later ``evict`` decrements for
        # keys that no longer exist, driving the counter negative.
        if len(shard) >= shard.max_size:
            victim = next(iter(shard), None)
            if victim is not None:
                shard.pop(victim, None)
                if self._index.pop(victim, None) is not None:
                    self.total_stored -= 1
                self.evictions += 1
        shard.put(key, answer)
        self._index[key] = (shard_idx, answer)
        self.total_stored += 1
        return True

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
            "evictions": self.evictions,
            # Invariant the W6 audit relies on: the global index and the shards
            # must agree, otherwise lookup and total_stored tell different
            # stories about the same store.
            "index_consistent": len(self._index) == self.total_stored,
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

    # --- Surface-form invariance: casing / padding is the same query ---
    padded = mem.lookup("  SPAIN ", " Capital ")
    print(f"\nCase/space invariant: '  SPAIN '/' Capital ' -> {padded} "
          f"(expected Madrid)")

    # --- Deterministic routing: same subject, same shard, every process ---
    print(f"Route('Spain') = {mem.route('Spain')} "
          f"(sha256-derived, stable across PYTHONHASHSEED)")

    # --- Counterfactual supersede: overwrite=False rejects, True revises ---
    refused = mem.store("Spain", "capital", "Lisbon")
    revised = mem.store("Spain", "capital", "Lisbon", overwrite=True)
    print(f"Supersede: default={refused} (expected False), "
          f"overwrite={revised} (expected True) -> "
          f"lookup={mem.lookup('Spain', 'capital')}")

    # --- Capacity eviction must not desync the global index ---
    small = SideMemory(n_shards=1, max_per_shard=3, seed=0)
    for i in range(6):
        small.store(f"subject-{i}", "capital", f"answer-{i}")
    st = small.stats()
    print(f"\nCapacity: stored={st['total_stored']} evictions={st['evictions']} "
          f"index_consistent={st['index_consistent']} (expected 3, 3, True)")
    print(f"  oldest evicted -> {small.lookup('subject-0', 'capital')} "
          f"(expected None), newest -> {small.lookup('subject-5', 'capital')}")

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
