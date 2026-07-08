# DWARF-v2

DWARF-v2 is a research prototype for experimenting with sparse language-model training paths, centered on HISA/DSQG-D routing and optional DSQG-W evidence recomposition. It is a compact, source-focused workspace extracted from a larger research tree so the trainer, kernels, tests, and reusable run scripts are easier to inspect and run.

This is not production-ready software. Expect rough edges, fast-changing experiment flags, hardware assumptions, and incomplete ergonomics. Treat it as a prototype for reproducing and extending research runs, not as a supported training framework.

## What is included

```text
train/                         main D512/L10 trainer entrypoint
kernels/                       sparse attention, scan, q6/HKV, and DSQG-W code
scripts/                       reusable launchers, profilers, and diagnostics
tests/                         focused regression and smoke tests
tools/passkey_eval.py          passkey guardrail evaluator
tokenizers/                    tracked tokenizer assets
datasets/                      local training data location; gitignored
checkpoints/, runs/, logs/     local run outputs; gitignored
```

Research notes, measurement dumps, raw eval JSON, checkpoints, and datasets are intentionally not tracked.

## Setup

Use a Python environment with CUDA-capable PyTorch available. One typical setup is:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default tokenizer is committed at:

```text
tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json
```

Training data is not committed. Place a compatible tokenized dataset at the default path:

```text
datasets/dwarf_base_v1_olmo1tok_2048_2b.pt
```

or pass another dataset path with `--dataset` / `DWARF_DATASET`.

## Run a small training smoke

A bounded launcher is provided for quick checks before longer experiments:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dsqg_w_full_training.py \
  --output-dir runs/dsqg_w_smoke \
  --dataset datasets/dwarf_base_v1_olmo1tok_2048_2b.pt \
  --gpu 0 \
  --max-acc-steps 25 \
  --train-seqs 256 \
  --val-seqs 128 \
  --batch-size 1 \
  --grad-accum 1 \
  --epochs 1 \
  --execute
```

For a configuration-only dry run, replace `--execute` with `--dry-run`; this writes `run_config.json` without launching the trainer.

## Direct trainer entrypoint

The underlying trainer can also be run directly through environment variables:

```bash
DWARF_DATASET=datasets/dwarf_base_v1_olmo1tok_2048_2b.pt \
DWARF_TOKENIZER=tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json \
DWARF_MAX_ACC_STEPS=25 \
DWARF_EPOCHS=1 \
DWARF_LIGER=0 \
PYTHONPATH=. .venv/bin/python train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py
```

Most architecture switches are environment variables beginning with `DWARF_`. See the trainer and launcher source for the current set of experiment flags.

## Tests

Run focused tests with:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

Some tests and training paths require CUDA and the optional packages listed in `requirements.txt`.

## License

Apache-2.0. See `LICENSE`.
