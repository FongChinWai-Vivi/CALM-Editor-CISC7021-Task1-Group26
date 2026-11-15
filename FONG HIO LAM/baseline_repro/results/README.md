# ROME gpt2-xl baseline — evaluation results (W2)

This directory holds the raw evaluation JSON of the ROME baseline produced by
`EasyEdit` on the local RTX 4070 (8 GB). The files are copied verbatim from
`EasyEdit/results/eval/` (which is git-ignored by the generic `results/` rule),
so that the numbers quoted in Table 1 of the report are reproducible from the
repository without a GPU run.

## Files

| File | Split | Scope | n | rewrite_acc | rephrase_acc | locality (NQ) | portability (p1) |
|---|---|---|---|---|---|---|---|
| `ROME_gpt2-xl_20261018_175857.json` | counterfact | official, first 3 | 3 | 1.0000 | 0.3333 | 0.9000 | – |
| `ROME_gpt2-xl_20261018_180857.json` | counterfact | official, first 50 | 50 | 1.0000 | 0.7000 | 0.8420 | – |
| `ROME_gpt2-xl_20261018_185020.json` | zsre | official, first 3 | 3 | 1.0000 | 1.0000 | 1.0000 | – |
| `ROME_gpt2-xl_20261018_185854.json` | mquake | official, first 3 | 3 | 0.8333 | – | – | 0.1349 |
| `ROME_gpt2-xl_20261018_190605.json` | zsre | official, first 50 | 50 | 0.9967 | 0.8073 | 0.9656 | – |
| `ROME_gpt2-xl_20261018_191619.json` | mquake | official, first 50 | 50 | 0.9467 | – | – | 0.0695 |

The `n=50` files are the source of the ROME rows of Table 1 in
`Group26_CALM_Final_Report`. The `n=3` files are the matching quick-check runs
used to validate the evaluation pipeline before the full sweep; they are kept
for provenance only and must not be cited as the reported numbers.

## Common run settings (identical in all six files)

- `alg = ROME`, `model = gpt2-xl`, `hparams = hparams/ROME/gpt2-xl.yaml`, `seed = 42`
- `smoke = 0` (real edits, not smoke mode)
- `gpu = RTX4070-8GB`, `fp16 = true`, `mom2_dataset = wikitext2`, `mom2_n_samples = 10000`
- `elapsed_s` / `peak_vram_mb` are recorded per file (peak VRAM 4.2–5.1 GB,
  i.e. comfortably inside the 8 GB budget of the target card)

## How to reproduce

Run from the repository root, with the virtual environment activated
(see the *Environment* section of the top-level `README.md`):

```bash
.venv/Scripts/python.exe EasyEdit/examples/w2_rome_repro.py \
    --data counterfact --n 50 --mom2-dataset wikitext2 --mom2-n-samples 10000 --fp16
.venv/Scripts/python.exe EasyEdit/examples/w2_rome_repro.py --data zsre   --n 50 --fp16
.venv/Scripts/python.exe EasyEdit/examples/w2_rome_repro.py --data mquake --n 50 --fp16
```

Output JSON is written to `EasyEdit/results/eval/` with the naming scheme
`ROME_<model>_<YYYYMMDD>_<HHMMSS>.json`; the six files here are the runs whose
timestamps appear in the file names (`2026-10-18`, 17:58–19:16). The `n=3`
variants are produced by the same commands with `--n 3`.

Note: `EasyEdit/.gitignore` ignores `EasyEdit/results/`, so a fresh run will not
show up in `git status`. Evidence JSON is therefore copied into this directory,
which is whitelisted in the top-level `.gitignore`
(`!FONG HIO LAM/baseline_repro/results/*.json`).
