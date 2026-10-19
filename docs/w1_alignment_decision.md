---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_08431d5cc9b811f196f9525400248c00
    ReservedCode1: o5SthRu2vD6Se4sx7K7NcEuc94AX/rz9RyHFbF4wTgHMsNdblhR1bIoIu/hQF0jR40d3mZqdz2dl5RHLcZZNpYfn9NC8+H5t9+5YDJJ2Ui3ugdcPMEzZMtFBDBf6Md8MDkjEGV/J3YJdX2MoEtUzOFthUoa03/hhboTGIya7+w8Ofxowp8jC2tXyYKc=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_08431d5cc9b811f196f9525400248c00
    ReservedCode2: o5SthRu2vD6Se4sx7K7NcEuc94AX/rz9RyHFbF4wTgHMsNdblhR1bIoIu/hQF0jR40d3mZqdz2dl5RHLcZZNpYfn9NC8+H5t9+5YDJJ2Ui3ugdcPMEzZMtFBDBf6Md8MDkjEGV/J3YJdX2MoEtUzOFthUoa03/hhboTGIya7+w8Ofxowp8jC2tXyYKc=
---

# W1 Alignment Decision — CALM Project

**Status**: Draft for the W1 team alignment meeting (~30 min)
**Date**: 2026-10-17
**Scope**: Locked model roster, interface decisions, Gate 1 checklist, and team onboarding actions.
**Related docs**: `docs/evaluation_protocol.md`, `docs/w1_hypernetwork_verification.md`, `README.md` (Collaboration Guidelines).

---

## 1. Locked Model Roster

| Model | HF id | Role | Cache path | Notes |
|---|---|---|---|---|
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B` | **Formal evaluation** — primary model for reliability / generalisation / locality measurements | `EasyEdit/hugging_cache/llama-3-8b` | Requires gated access on Hugging Face; 4-bit quantisation planned; used for the main results table and baseline comparisons |
| GPT-J-6B | `EleutherAI/gpt-j-6b` | **Formal evaluation** — small-scale / early validation, cheaper iteration than Llama-3-8B | `EasyEdit/hugging_cache/gpt-j-6b` | Used for MEND/MALMEN early validation and quick pipeline checks |
| gpt2-xl | `gpt2-xl` | **Fast verification** — ROME single-fact demo already runs | `EasyEdit/hugging_cache/gpt2-xl` | Verified via `EasyEdit/examples/w1_rome_demo.py` (6-case smoke test) |
| gpt2-small | `gpt2` | **Fast verification** — hypernetwork (MEND) pipeline check | `EasyEdit/hugging_cache/gpt2-small` | Verified via `EasyEdit/examples/w1_mend_small_train.py`; see `docs/w1_hypernetwork_verification.md` |

**Decision**: All formal numbers are produced on Llama-3-8B (main) and GPT-J-6B (early/small-scale). gpt2-xl and gpt2-small remain the fast-iteration models for pipeline and interface smoke tests and are **not** used for reported metrics.

---

## 2. Interface Decisions

### 2.1 Unified editing entry point: `CALMEditor`

- Location: `calm/calm_editor.py` (project-level module, sibling of the `EasyEdit/` code base).
- API style: compatible with the EasyEdit `BaseEditor` surface — `from_hparams`, `edit`, `batch_edit`.
- Purpose: one entry point for all CALM editing flows (single edit, batch edit, and lifelong/stream editing via the hypernetwork + side memory + conflict gate).

### 2.2 Three components and their responsibilities

| Component | Responsibility | Docking point |
|---|---|---|
| `HypernetworkEditor` | MEND/MALMEN-style hypernetwork that maps a subject–relation–object edit into a low-rank weight update | Produces the candidate parameter update; exposes `generate_update(edit)` |
| `SideMemory` | WISE-style non-parametric side memory (sharded) for edits that are routed instead of written to parameters | Exposes `lookup(edit)`, `store(edit, value)`; shards are never overwritten |
| `ConflictGate` | Scores the candidate edit against (a) current answer distribution, (b) closest previously stored edit, (c) retrieved context; decides write / route / reject | Exposes `decide(edit, candidate_update, context) -> Write | Route | Reject` |

### 2.3 Final implementation target

CALM is implemented as a **new `BaseEditor` subclass** in EasyEdit (per the README Collaboration Guidelines): the hypernetwork produces the update, the conflict gate decides the write destination, and the side memory is sharded. The `calm/calm_editor.py` draft is the agreed interface blueprint; W3 implementation lands it as the actual EasyEdit editor subclass.

**Decision**: The interface in `calm/calm_editor.py` is frozen for the alignment meeting. Any change requires a team decision and a follow-up revision of this document.

---

## 3. Gate 1 Checklist (end of W1)

Progress is measured by what has actually been produced (README Stage gates).

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Data pipelines: CounterFact / zsRE / MQuAKE data loaders all run | In progress | Loader definitions in `EasyEdit/easyeditor/dataset/`; small zsRE-format subset verified during MEND training (`EasyEdit/data/zsre_mend_train_small.json`) |
| 2 | Metric scripts: rewrite_acc, paraphrase, locality, portability run under one protocol | In progress | Protocol defined in `docs/evaluation_protocol.md`; metric implementations available in `EasyEdit/easyeditor/evaluate/` |
| 3 | Hypernetwork training pipeline validated at small scale | **Done** | `docs/w1_hypernetwork_verification.md` — MEND on gpt2-small ran 501 steps end-to-end, loss converged (edit_train 12.49 → 8.87, total_train 1.30 → 0.90), `EasyEdit/results/results.json` produced |

**Acceptance for Gate 1**: items 1–2 must be reported as "all run" at the end of W1 (each member confirms on their own branch); item 3 is already satisfied.

---

## 4. Team Onboarding Actions

Each member (Fong Chin Wai, Fong Hio Lam, Tang I Choi) should complete the following before the alignment meeting:

1. **Clone the repository**
   ```bash
   git clone https://github.com/FongChinWai-Vivi/CALM-Editor-CISC7021-Task1-Group26.git
   cd CALM-Editor-CISC7021-Task1-Group26
   ```
   (If already cloned, `git pull` the latest `main`.)

2. **Set up the environment** — create/use the project-root `.venv` (Python 3.11) and install EasyEdit requirements:
   ```bash
   # From the project root
   python -m venv .venv
   .venv\Scripts\activate            # Windows
   pip install -r EasyEdit\requirements.txt
   # PyTorch with CUDA 12.6 (required for GPU training/editing)
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```
   Note: install the CUDA 12.6 build of torch from `download.pytorch.org/whl/cu126` so that the GPU runtime matches the machine (RTX 4070, 8 GB).

3. **Confirm GPU availability** — verify CUDA is visible and VRAM is free:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Expected: `True RTX 4070` (or similar).

4. **Read the three W1 docs** (5 min each):
   - `docs/evaluation_protocol.md` — metrics and protocol
   - `docs/w1_hypernetwork_verification.md` — pipeline status
   - `docs/w1_alignment_decision.md` — this document

5. **Join the ~30-minute alignment meeting** to lock:
   - the model roster (Section 1),
   - the `CALMEditor` interface and component docking (Section 2),
   - the Gate 1 acceptance status (Section 3).

**Alignment meeting outcome**: any change to the roster or interface is recorded as a new revision of this document; the frozen version is committed to `main` after the meeting.
*（内容由AI生成，仅供参考）*
