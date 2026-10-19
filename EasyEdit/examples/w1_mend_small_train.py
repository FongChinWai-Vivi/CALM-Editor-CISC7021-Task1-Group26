# W1 small-scale hypernetwork (MEND) training pipeline verification.
# Trains a MEND editor on gpt2-small (124M) with a small zsRE-format subset.
# Goal: prove the pipeline runs end-to-end (data -> EditTrainer -> loss logs).
import sys
import os
import json
import logging
from pathlib import Path

EASYEDIT = str(Path(__file__).resolve().parent.parent)
os.chdir(EASYEDIT)
sys.path.insert(0, EASYEDIT)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)

from easyeditor import MENDTrainingHparams, ZsreDataset, EditTrainer

HPARAMS = os.path.join(EASYEDIT, "hparams/TRAINING/MEND/gpt2-small.yaml")
TRAIN_DS = os.path.join(EASYEDIT, "data/zsre_mend_train_small.json")
EVAL_DS = os.path.join(EASYEDIT, "data/zsre_mend_eval_small.json")

print("[W1-MEND] loading training hparams ...", flush=True)
training_hparams = MENDTrainingHparams.from_hparams(HPARAMS)

print("[W1-MEND] loading small zsRE datasets ...", flush=True)
train_ds = ZsreDataset(TRAIN_DS, config=training_hparams)
eval_ds = ZsreDataset(EVAL_DS, config=training_hparams)
print(f"[W1-MEND] train={len(train_ds)} eval={len(eval_ds)}", flush=True)

trainer = EditTrainer(config=training_hparams, train_set=train_ds, val_set=eval_ds)
print("[W1-MEND] starting trainer.run() ...", flush=True)
trainer.run()
print("[W1-MEND] trainer.run() finished", flush=True)
