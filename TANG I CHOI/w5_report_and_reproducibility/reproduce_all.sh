#!/usr/bin/env bash
# W5 (Tang I Choi) - one-command reproduction of the W3/W4/W5 artefacts.
#
#   bash "TANG I CHOI/w5_report_and_reproducibility/reproduce_all.sh"
#
# Every step is idempotent and writes its own log into logs/.  Steps 1-2
# (data build + classifier training) are skipped when the artefacts already
# exist in the repository, because they need the sampled CounterFact answers
# produced by the GPU stream runs; set FORCE_TRAIN=1 to rebuild anyway.
#
# Environment knobs:
#   PY           python interpreter          (default: python)
#   SUBJECTS     distinct subjects in the W4 stream (default: 40)
#   FORCE_TRAIN  rebuild data + retrain the W3 classifier (default: 0)
#   DEVICE       torch device, passed to the CALM hparams (default: mps if available)
set -euo pipefail

PY="${PY:-python}"
SUBJECTS="${SUBJECTS:-40}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W3="${ROOT}/TANG I CHOI/w3_conflict_gate_training_and_classifier"
W4="${ROOT}/TANG I CHOI/w4_gate_ablations"
W5="${ROOT}/TANG I CHOI/w5_report_and_reproducibility"
LOGS="${W5}/logs"
mkdir -p "${LOGS}"

echo "== repository: ${ROOT}"
echo "== python:     $(${PY} -c 'import sys; print(sys.version.split()[0])')"
echo "== steps log:  ${LOGS}"

step() { echo; echo "== [$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- W3 data ---
if [[ "${FORCE_TRAIN}" == "1" || ! -f "${W3}/gate_datasets/gate_train.jsonl" ]]; then
  step "W3: build the four-label gate dataset"
  ( cd "${W3}" && ${PY} build_gate_training_data.py ) \
    2>&1 | tee "${LOGS}/w3_build_data.log"
else
  echo "== W3: gate_datasets present, skipping the data build"
fi

# ------------------------------------------------------- W3 classifier ------
if [[ "${FORCE_TRAIN}" == "1" || ! -f "${W3}/gate_model/gate_classifier.joblib" ]]; then
  step "W3: train the multinomial logistic gate"
  ( cd "${W3}" && ${PY} train_gate_classifier.py ) \
    2>&1 | tee "${LOGS}/w3_train_gate.log"
else
  echo "== W3: gate_classifier.joblib present, skipping training"
fi

# --------------------------------------------- W3 offline/online gap --------
step "W3: quantify the offline -> CALM-stream feature gap"
( cd "${W3}" && ${PY} evaluate_gate_online_gap.py ) \
  2>&1 | tee "${LOGS}/w3_gate_online_gap.log"

# ------------------------------------------------------- W4 ablations -------
step "W4: run the four gate arms over the stream (subjects=${SUBJECTS})"
( cd "${W4}" && ${PY} run_w4_gate_ablation.py --all-arms --subjects "${SUBJECTS}" ) \
  2>&1 | tee "${LOGS}/w4_ablation_$(date +%Y%m%d).log"

# ------------------------------------------------- W4 statistics ------------
step "W4: paired tests, confidence intervals, per-conflict breakdown (191-edit stream)"
( cd "${W4}" && ${PY} analyze_w4_statistics.py --stream canonical ) \
  2>&1 | tee "${LOGS}/w4_statistics.log"

# The deduped rerun exists to isolate the immediate-rewrite defect; render its
# table too when the rerun JSON is present, without ever overwriting the 191-edit
# table used by the report.
if ls "${W4}"/results/w4_ablation_*_deduped.json >/dev/null 2>&1; then
  step "W4: render the deduped-stream table (defect 1, see docs/w4_immediate_rewrite_diagnosis.md)"
  ( cd "${W4}" && ${PY} analyze_w4_statistics.py --stream deduped ) \
    2>&1 | tee "${LOGS}/w4_statistics_deduped.log"
fi

# ------------------------------------------------------- W5 figures ---------
step "W5: render the results figures"
( cd "${W5}" && ${PY} make_w5_figures.py ) \
  2>&1 | tee "${LOGS}/w5_figures.log"

# ------------------------------------------------------- W5 tables ----------
step "W5: refresh the results tables copied into the report"
( cd "${W5}" && ${PY} collect_w5_tables.py ) \
  2>&1 | tee "${LOGS}/w5_tables.log"

echo
echo "== done. Artefacts:"
echo "   ${W4}/results/w4_ablation_*.json           (per-arm raw metrics)"
echo "   ${W4}/results/w4_statistics.json           (all statistical tests)"
echo "   ${W4}/results/w4_ablation_table.md         (ablation table, 191-edit stream)"
echo "   ${W4}/results/w4_ablation_table_deduped.md (ablation table, 40-prompt rerun)"
echo "   ${W3}/results/gate_online_gap.json         (offline vs stream gate)"
echo "   ${W5}/figures/*.png                        (figures 1-3)"
echo "   ${W5}/report/results_tables.md             (tables used by the report)"
