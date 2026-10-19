# W1 demo: ROME on gpt2-xl (EasyEdit framework smoke test)
# Run from anywhere; the script locates the EasyEdit repo relative to this file.
# Verified: 6 factual edits succeed on RTX 4070 (8GB) with wikitext mom2 stats.
import sys
import os
import json
from pathlib import Path

# Repo root = parent of this file's directory (examples/ -> EasyEdit/)
EASYEDIT = str(Path(__file__).resolve().parent.parent)
os.chdir(EASYEDIT)
sys.path.insert(0, EASYEDIT)

from easyeditor import ROMEHyperParams, BaseEditor

hparams = ROMEHyperParams.from_hparams(os.path.join(EASYEDIT, "hparams/ROME/gpt2-xl.yaml"))
hparams.model_name = os.path.join(EASYEDIT, "hugging_cache", "gpt2-xl")
hparams.stats_dir = os.path.join(EASYEDIT, "data", "stats")
# Use wikitext-103 for mom2 stats to avoid multi-GB wikipedia download in smoke test.
hparams.mom2_dataset = "wikitext"

prompts = [
    "What university did Watts Humphrey attend?",
    "Which family does Ramalinaceae belong to",
    "What role does Denny Herzig play in football?",
    "Who was the designer of Lahti Town Hall?",
    "What is the original channel that It's a Business played on?",
    "What city did Marl Young live when he died?",
]
ground_truth = [
    "Illinois Institute of Technology",
    "Lecanorales",
    "defender",
    "Eliel Saarinen",
    "DuMont Television Network",
    "Los Angeles",
]
target_new = [
    "University of Michigan",
    "Lamiinae",
    "winger",
    "Alfred Lahti",
    "ITV",
    "New Orleans",
]
subject = [
    "Watts Humphrey",
    "Ramalinaceae",
    "Denny Herzig",
    "Lahti Town Hall",
    "It's a Business",
    "Marl Young",
]

print("[W1] loading editor ...", flush=True)
editor = BaseEditor.from_hparams(hparams)
print("[W1] editor loaded, starting edit ...", flush=True)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    keep_original_weight=True,
)
print("[W1] edit finished", flush=True)
print("[W1-METRICS]", json.dumps(metrics, indent=2, default=str), flush=True)
