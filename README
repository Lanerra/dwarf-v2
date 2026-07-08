# DWARF-v2

Clean DWARF v2 workspace for DSQG-W experiments.

This repo intentionally starts from the smallest useful DWARF trainer slice rather than the dirty historical DWARF workspace.

## Layout

```text
train/                         D512/L10 HISA+DSQG-D trainer entrypoint
kernels/                       kernel code kept together
  dsqg_attention_v20_bf16_se.py DSQG-D reference/current kernel path
  hierarchical_sparse_attn_v15_hisa.py
  causal_ema_scan.py
  q6_g128/                     q6/HKV support used by the trainer
  dsqg_w/                      DSQG-W-only kernel/operator work
    dsqg_w_mvp.py              first diagnostic DSQG-W reference implementation
tools/passkey_eval.py          passkey guardrail evaluator used by trainer
tokenizers/                    OLMo-1/GPT-NeoX-Dolma tokenizer
datasets/                      local dataset artifacts, gitignored
tests/                         focused DSQG-W tests
```

## Local data

The current local trainer dataset has been copied into:

```text
datasets/dwarf_base_v1_olmo1tok_2048_2b.pt
```

It is intentionally gitignored because it is about 8 GB. The code and tokenizer are committed; large run artifacts stay local.

## Smoke commands

DSQG-W reference tests:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest tests/test_dsqg_w_mvp.py -q
```

Trainer import smoke:

```bash
DWARF_DISABLE_BNB=1 DWARF_LIGER=0 PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location('trainer', 'train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.TriadicJ96Dsr.__name__, mod.EMBEDDING_DIM, mod.NUM_LAYERS)
PY
```

Minimal one-step trainer smoke can be run by setting `DWARF_MAX_ACC_STEPS=1`, `DWARF_EPOCHS=1`, `DWARF_LIGER=0`, and appropriate CUDA env if desired.
