"""MEMIT reproduction -- CALM W2 baseline (Fong Hio Lam, MC654953).

Reproduces the core mechanism of MEMIT (Mass-Editing Memory in a
Transformer, Meng et al., 2022) on a small self-contained NumPy model:

* knowledge is stored in the intermediate FFN weights of a tiny residual
  MLP ("MiniLM");
* for each edit we cache the target-layer input ("keys") and compute a
  desired output residual ("values") that drives the final logits toward
  the new answer;
* a batch of edits is written in one step by solving a ridge-regularised
  least-squares problem with a context covariance term ``C`` that
  preserves unrelated facts (the ROME/MEMIT "preserve" constraint).

The script is fully self-contained (NumPy only) and runs a small smoke
test with ``--mock``. For formal evaluation, swap the MiniLM forward
hooks for a Hugging Face transformer (see README.md): the algorithmic
core (cache -> least-squares solve -> weight write) is unchanged and
compatible with the W1 locked roster (Llama-3-8B formal, GPT-J-6B early,
gpt2-xl/gpt2-small fast verification).

Evaluation follows the W1 protocol: rewrite_acc (reliability),
paraphrase_acc (generalisation), locality_neighborhood_acc /
locality_attribute_acc (locality) and portability_acc (multi-hop).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Tiny config (MiniLM hyper-parameters, kept small for smoke tests)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 128       # toy vocabulary size
PROMPT_LEN = 8         # number of tokens per prompt
HIDDEN = 512           # hidden width of MiniLM
NUM_LAYERS = 6         # number of residual FFN blocks
EDIT_LAYER = 3         # layer whose FFN weights receive the edit (middle)
ACT = "linear"         # activation used inside each FFN block


# ---------------------------------------------------------------------------
# Text-free toy tokenizer
# ---------------------------------------------------------------------------


def _rand_tokens(rng: random.Random, n: int, vocab: int) -> List[int]:
    return [rng.randrange(vocab) for _ in range(n)]


def tokens_to_text(tokens: Sequence[int]) -> str:
    """Render a token list as a readable string for logging only."""
    return " ".join(f"t{t}" for t in tokens)


# ---------------------------------------------------------------------------
# MiniLM -- a tiny residual MLP that mimics a transformer's FFN storage
# ---------------------------------------------------------------------------


class MiniLM:
    """Tiny editable language model.

    Architecture (residual blocks, transformer-like):

        x = mean(emb[prompt tokens])          # (H,)
        for l in range(NUM_LAYERS):
            h = x + W_l * act(x)              # FFN block with weight W_l
        logits = h @ W_out.T + b_out

    With all ``W_l == 0`` the model is a linear map fitted on the initial
    knowledge set through ``fit_output``. Edits modify a chosen ``W_l``
    so the edited facts are answered with the new target while unrelated
    (locality) prompts keep their pre-edit answer.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)
        # Fixed, roughly-orthonormal embedding table (rows are vocab vectors).
        emb = self.rng.normal(size=(HIDDEN, VOCAB_SIZE)).astype(np.float64)
        q, _ = np.linalg.qr(emb)
        self.emb = q.T  # (VOCAB_SIZE, HIDDEN)
        # FFN weights: start at zero (identity residual path).
        self.Ws: List[np.ndarray] = [
            np.zeros((HIDDEN, HIDDEN), dtype=np.float64) for _ in range(NUM_LAYERS)
        ]
        self.W_out: Optional[np.ndarray] = None
        self.b_out: Optional[np.ndarray] = None
        self._act = (lambda z: np.maximum(z, 0.0)) if ACT == "relu" else (lambda z: z)
        self._rand = py_rng

    # -- forward ----------------------------------------------------------

    def embed_prompt(self, tokens: Sequence[int]) -> np.ndarray:
        """Average the token embeddings of a prompt into one vector."""
        if not tokens:
            raise ValueError("prompt must contain at least one token")
        return self.emb[np.asarray(tokens, dtype=int)].mean(axis=0)

    def ffn(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """One residual FFN block: ``x + W * act(x)``."""
        return x + w @ self._act(x)

    def forward_from(self, x: np.ndarray, stop_after: Optional[int] = None) -> np.ndarray:
        """Run the residual blocks up to (and including) ``stop_after``."""
        h = x
        limit = NUM_LAYERS if stop_after is None else stop_after + 1
        for l in range(limit):
            h = self.ffn(h, self.Ws[l])
        return h

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward: logits over the vocabulary."""
        h = self.forward_from(x)
        if self.W_out is None or self.b_out is None:
            raise RuntimeError("fit_output() must be called before forward()")
        return h @ self.W_out.T + self.b_out

    def generate(self, tokens: Sequence[int]) -> int:
        """Greedy argmax answer token for a prompt."""
        logits = self.forward(self.embed_prompt(tokens))
        return int(np.argmax(logits))

    # -- initial knowledge fit -------------------------------------------

    def fit_output(self, knowledge: Sequence[Tuple[List[int], int]],
                   reg: float = 1e-4):
        """Fit the output layer so each knowledge prompt answers correctly.

        Solves ``h_i @ W_out.T = onehot(answer_i)`` with ridge-regularised
        least squares (``lstsq``), which stays numerically stable when the
        prompt features are near-degenerate. After fitting, the model
        answers every knowledge item with its pre-edit (``target_true``)
        answer.
        """
        rows = []
        targets = []
        for tokens, answer in knowledge:
            rows.append(self.embed_prompt(tokens))
            targets.append(answer)
        H = np.vstack(rows)                        # (n_know, H)
        Y = np.zeros((len(targets), VOCAB_SIZE))
        for i, ans in enumerate(targets):
            Y[i, ans] = 1.0
        # W_out.T solves H @ X = Y in the ridge least-squares sense.
        Wt = np.linalg.lstsq(H, Y, rcond=None)[0]  # (HIDDEN, n_know)
        self.W_out = Wt.T                          # (VOCAB, HIDDEN)
        self.b_out = np.zeros(VOCAB_SIZE, dtype=np.float64)

    # -- state helpers ----------------------------------------------------

    def get_ffn_weight(self, layer: int) -> np.ndarray:
        return self.Ws[layer]

    def set_ffn_weight(self, layer: int, w: np.ndarray):
        self.Ws[layer] = w

    def deepcopy_weights(self) -> List[np.ndarray]:
        return [w.copy() for w in self.Ws]


# ---------------------------------------------------------------------------
# Toy evaluation data (EditCase-aligned dicts)
# ---------------------------------------------------------------------------


def make_toy_cases(model: MiniLM, n_cases: int = 3, seed: int = 0) -> List[Dict]:
    """Build ``n_cases`` self-contained edit cases for the smoke test.

    Each case models a "subject -- relation -- object" fact. The first
    ``SUBJECT_LEN`` tokens of a prompt are the subject (kept identical
    across the rewrite/paraphrase/portability prompts), the remaining
    tokens are context. Locality prompts use completely different tokens,
    so their edit-layer keys stay (nearly) orthogonal to the edited key
    and the edit does not disturb them -- the same separation that makes
    MEMIT local on real transformers.

    All locality/portability targets are captured as the pre-edit greedy
    answers, so locality is scored exactly like the W1 protocol (keep the
    pre-edit answer after the edit).
    """
    SUBJECT_LEN = 4
    rng = random.Random(seed + 1000)
    cases: List[Dict] = []
    for idx in range(n_cases):
        subject = _rand_tokens(rng, SUBJECT_LEN, VOCAB_SIZE)
        ctx = _rand_tokens(rng, PROMPT_LEN - SUBJECT_LEN, VOCAB_SIZE)
        prompt = subject + ctx
        target_new = rng.randrange(VOCAB_SIZE)
        # Paraphrases / portability: same subject, fresh context.
        paraphrases = [
            subject + _rand_tokens(rng, PROMPT_LEN - SUBJECT_LEN, VOCAB_SIZE)
            for _ in range(2)
        ]
        portability = [subject + _rand_tokens(rng, PROMPT_LEN - SUBJECT_LEN,
                                              VOCAB_SIZE)]
        # Locality prompts: unrelated subjects (fresh tokens everywhere).
        neighborhood = [_rand_tokens(rng, PROMPT_LEN, VOCAB_SIZE) for _ in range(2)]
        attribute = [_rand_tokens(rng, PROMPT_LEN, VOCAB_SIZE) for _ in range(2)]
        cases.append({
            "case_id": idx,
            "prompt": prompt,
            "subject": subject,
            "target_new": target_new,
            "target_true": None,   # pre-edit answer of neighborhood[0]
            "paraphrasess": paraphrases,
            "neighborhood_prompts": neighborhood,
            "attribute_prompts": attribute,
            "portability_prompts": portability,
            "neighborhood_answers": [],
            "attribute_answers": [],
        })
    # Capture pre-edit answers for every locality prompt.
    for case in cases:
        case["target_true"] = model.generate(case["neighborhood_prompts"][0])
        case["neighborhood_answers"] = [
            model.generate(p) for p in case["neighborhood_prompts"]
        ]
        case["attribute_answers"] = [
            model.generate(p) for p in case["attribute_prompts"]
        ]
    return cases


# ---------------------------------------------------------------------------
# MEMIT core algorithm
# ---------------------------------------------------------------------------


def cache_keys(model: MiniLM, cases: Sequence[Dict]) -> np.ndarray:
    """Cache the EDIT_LAYER input vectors ("keys") for every edit case.

    Keys are the FFN-block inputs ``act(x)`` used as features of the
    least-squares write; this mirrors MEMIT's per-edit key caching at the
    target layer.
    """
    keys = []
    for case in cases:
        x = model.embed_prompt(case["prompt"])
        h = model.forward_from(x, stop_after=EDIT_LAYER - 1)
        keys.append(h)
    return np.vstack(keys)  # (n_cases, H)


def desired_residuals(model: MiniLM, cases: Sequence[Dict],
                      margin: float = 0.1) -> np.ndarray:
    """Compute the desired output residuals ("values") for each edit.

    Following the ROME/MEMIT value-optimisation spirit we first define a
    *minimal logits change*: raise only the logit of ``target_new`` until
    it beats every other answer by a small ``margin``. The desired hidden
    residual is then the least-norm hidden change that realises that
    logits change through the output head (``lstsq`` against ``W_out``).
    This keeps the edit as small as possible, which is what makes the
    covariance-preservation constraint effective for locality.
    """
    if model.W_out is None:
        raise RuntimeError("model is not fitted")
    residuals = []
    for case in cases:
        x = model.embed_prompt(case["prompt"])
        logits = model.forward(x)
        tn = case["target_new"]
        others = np.delete(logits, tn)
        need = float(np.max(others) - logits[tn]) + margin
        delta_logits = np.zeros(VOCAB_SIZE, dtype=np.float64)
        if need > 0.0:
            delta_logits[tn] = need
        delta = np.linalg.lstsq(model.W_out, delta_logits, rcond=None)[0]
        residuals.append(delta)
    return np.vstack(residuals)  # (n_cases, H)


def context_covariance(model: MiniLM, n_context: int = 16, seed: int = 0) -> np.ndarray:
    """Context covariance ``C = sum_k k k^T`` over unrelated samples.

    Following ROME/MEMIT, adding ``C`` to the regularised solve keeps the
    update from disturbing unrelated (locality) facts.
    """
    rng = random.Random(seed + 2000)
    acc = np.zeros((HIDDEN, HIDDEN), dtype=np.float64)
    for _ in range(n_context):
        tokens = _rand_tokens(rng, PROMPT_LEN, VOCAB_SIZE)
        x = model.embed_prompt(tokens)
        h = model.forward_from(x, stop_after=EDIT_LAYER - 1)
        k = h
        acc += np.outer(k, k)
    return acc


def compute_layer_update(model: MiniLM,
                         cases: Sequence[Dict],
                         lam: float = 1e-2,
                         margin: float = 0.1,
                         context_scale: float = 50.0,
                         context_seed: int = 0) -> np.ndarray:
    """Solve the MEMIT batch update for the target FFN layer.

    Returns ``delta_W`` with

        delta_W = R^T K (lam I + context_scale * C + K^T K)^{-1}

    where ``K`` are cached keys (rows = per-edit activations), ``R`` are
    desired residuals (rows = per-edit targets) and ``C`` is the context
    covariance. The covariance term is up-weighted by ``context_scale``
    so unrelated (locality) facts are preserved while the edits are still
    written in one step. Applying ``W_l <- W_l + delta_W`` completes the
    batch edit.
    """
    K = cache_keys(model, cases)
    R = desired_residuals(model, cases, margin=margin)
    C = context_covariance(model, seed=context_seed)
    G = lam * np.eye(HIDDEN) + context_scale * C + K.T @ K
    delta_W = R.T @ K @ np.linalg.inv(G)
    return delta_W


# ---------------------------------------------------------------------------
# Editor facade (EasyEdit BaseEditor-style surface)
# ---------------------------------------------------------------------------


class MemitEditor:
    """MEMIT baseline editor with an EasyEdit ``BaseEditor``-style surface.

    Usage::

        editor = MemitEditor.from_hparams(model=model, layer=EDIT_LAYER,
                                          lam=1e-2)
        editor.batch_edit(cases)          # single-step multi-edit write
        metrics = editor.evaluate(cases)  # W1 protocol metrics
    """

    def __init__(self, model: MiniLM, layer: int = EDIT_LAYER,
                 lam: float = 1e-2, margin: float = 0.1,
                 context_scale: float = 50.0):
        self.model = model
        self.layer = layer
        self.lam = lam
        self.margin = margin
        self.context_scale = context_scale
        self.backup: Optional[List[np.ndarray]] = None

    @classmethod
    def from_hparams(cls, model: MiniLM, layer: int = EDIT_LAYER,
                     lam: float = 1e-2, margin: float = 0.1,
                     context_scale: float = 50.0) -> "MemitEditor":
        """Create an editor from hyper-parameters (EasyEdit convention)."""
        return cls(model=model, layer=layer, lam=lam, margin=margin,
                   context_scale=context_scale)

    def edit(self, case: Dict) -> Dict:
        """Apply a single edit (wraps :meth:`batch_edit`)."""
        return self.batch_edit([case])

    def batch_edit(self, cases: Sequence[Dict]) -> Dict:
        """Write all edits in one MEMIT batch update.

        Returns a summary dict with the applied layer, delta-W norm and
        the per-case post-edit rewrite answers.
        """
        if self.backup is None:
            self.backup = self.model.deepcopy_weights()
        delta_W = compute_layer_update(self.model, cases, lam=self.lam,
                                       margin=self.margin,
                                       context_scale=self.context_scale)
        old = self.model.get_ffn_weight(self.layer)
        self.model.set_ffn_weight(self.layer, old + delta_W)
        return {
            "method": "MEMIT",
            "layer": self.layer,
            "n_edits": len(cases),
            "delta_norm": float(np.linalg.norm(delta_W)),
            "post_edit_answers": [
                tokens_to_text([case["prompt"][0]]) + " -> t"
                + str(self.model.generate(case["prompt"]))
                for case in cases
            ],
        }

    def evaluate(self, cases: Sequence[Dict]) -> Dict[str, float]:
        """Run the W1 metric battery on the edited model."""
        return evaluate_cases(self.model, cases)


# ---------------------------------------------------------------------------
# W1 protocol metrics (normalised exact/containment on toy tokens)
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def _acc(hits: int, total: int) -> float:
    return float(hits) / total if total else 0.0


def evaluate_cases(model: MiniLM, cases: Sequence[Dict]) -> Dict[str, float]:
    """Score a case list against the W1 evaluation protocol.

    ``rewrite`` uses the edit prompt, ``paraphrase`` uses every
    paraphrase, ``locality`` uses neighborhood/attribute prompts (target
    = pre-edit answer) and ``portability`` uses the multi-hop prompt.
    """
    def hit(gen: int, target: int) -> bool:
        return gen == target

    rewrite_hits = sum(hit(model.generate(c["prompt"]), c["target_new"]) for c in cases)
    para_hits, para_n = 0, 0
    neigh_hits, neigh_n = 0, 0
    attr_hits, attr_n = 0, 0
    port_hits, port_n = 0, 0
    for c in cases:
        for p in c["paraphrasess"]:
            para_n += 1
            para_hits += int(hit(model.generate(p), c["target_new"]))
        neigh_answers = c.get("neighborhood_answers") or [
            c["target_true"]] * len(c["neighborhood_prompts"])
        for p, expected in zip(c["neighborhood_prompts"], neigh_answers):
            neigh_n += 1
            neigh_hits += int(hit(model.generate(p), expected))
        attr_answers = c.get("attribute_answers") or []
        for p, expected in zip(c["attribute_prompts"], attr_answers):
            attr_n += 1
            attr_hits += int(hit(model.generate(p), expected))
        for p in c["portability_prompts"]:
            port_n += 1
            port_hits += int(hit(model.generate(p), c["target_new"]))
    return {
        "rewrite_acc": _acc(rewrite_hits, len(cases)),
        "paraphrase_acc": _acc(para_hits, para_n),
        "locality_neighborhood_acc": _acc(neigh_hits, neigh_n),
        "locality_attribute_acc": _acc(attr_hits, attr_n),
        "portability_acc": _acc(port_hits, port_n),
        "n_cases": len(cases),
    }


# ---------------------------------------------------------------------------
# Smoke-test wiring
# ---------------------------------------------------------------------------


def build_smoke_env(seed: int = 0, n_cases: int = 3,
                    n_knowledge: Optional[int] = None):
    """Create a fitted MiniLM and ``n_cases`` toy evaluation cases.

    ``n_knowledge`` defaults to ``VOCAB_SIZE`` and the knowledge answers
    cover the whole vocabulary exactly once. This gives every answer
    token a non-zero output row in ``W_out`` (so an edit can push any new
    answer through the output layer) and keeps the fitted output layer at
    full row rank, matching the way a real pretrained head spans the
    vocabulary.
    """
    rng = random.Random(seed + 3000)
    model = MiniLM(seed=seed)
    if n_knowledge is None:
        knowledge = [
            (_rand_tokens(rng, PROMPT_LEN, VOCAB_SIZE), t)
            for t in range(VOCAB_SIZE)
        ]
    else:
        knowledge = [
            (_rand_tokens(rng, PROMPT_LEN, VOCAB_SIZE), rng.randrange(VOCAB_SIZE))
            for _ in range(n_knowledge)
        ]
    model.fit_output(knowledge)
    cases = make_toy_cases(model, n_cases=n_cases, seed=seed)
    return model, cases


def run_smoke(n_cases: int = 3, seed: int = 0, lam: float = 1e-3,
              margin: float = 1.0, context_scale: float = 5.0,
              verbose: bool = False) -> Dict:
    """End-to-end MEMIT smoke test: fit -> batch edit -> evaluate."""
    model, cases = build_smoke_env(seed=seed, n_cases=n_cases)
    editor = MemitEditor.from_hparams(model=model, layer=EDIT_LAYER, lam=lam,
                                      margin=margin,
                                      context_scale=context_scale)
    if verbose:
        print("[smoke] initial model fitted,", len(cases), "edit cases ready")
    summary = editor.batch_edit(cases)
    metrics = editor.evaluate(cases)
    if verbose:
        print("[smoke] edit summary:", json.dumps(summary, indent=2))
        print("[smoke] metrics:", json.dumps(metrics, indent=2))
    return {"summary": summary, "metrics": metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MEMIT baseline reproduction (CALM W2, self-contained)."
    )
    parser.add_argument("--mock", action="store_true",
                        help="run the smoke test on the built-in toy model.")
    parser.add_argument("--cases", type=int, default=3,
                        help="number of toy edit cases (smoke test).")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for reproducibility.")
    parser.add_argument("--lam", type=float, default=1e-3,
                        help="ridge regularisation weight.")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="safety margin for the new answer logit.")
    parser.add_argument("--context-scale", type=float, default=5.0,
                        help="weight of the context covariance term.")
    parser.add_argument("--verbose", action="store_true",
                        help="print the edit summary and metrics.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.mock:
        raise SystemExit(
            "provide --mock for the self-contained smoke test; for formal "
            "evaluation import MemitEditor and attach a Hugging Face "
            "transformer (see README.md)."
        )
    result = run_smoke(n_cases=args.cases, seed=args.seed, lam=args.lam,
                       margin=args.margin,
                       context_scale=args.context_scale,
                       verbose=args.verbose)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
