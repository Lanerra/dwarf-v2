<p align="center">
  <img src="dwarf-logo.png" alt="DWARF logo: a dwarf holding two axes" width="360">
</p>

# DWARF-v2

DWARF-v2 is a compact causal language-model architecture built from Dynamic Sparse Query-Gather (DSQG) blocks and one L3 global mixer. This repository is the minimal source needed to train the active architecture.

The public tree intentionally contains only the runtime source needed to construct and train the active architecture.  Datasets, checkpoints, launch scripts, evaluation outputs, Hugging Face staging files, diagnostics, and retired experiments remain local and are not part of this repository.

## Architecture

- Nine triadic DSQG blocks using the canonical 96-offset lattice and calibrated content-dependent MOVT initialization.
- A causal-EMA interference injection in the final pre-L3 DSQG block.
- One L3 global mixer: strict-causal V16 HISA by default, or full causal SDPA with `--global-mixer fa`.

The HISA kernel uses a 64-token local lane, 16-token selector tiles, blocked bounded-memory metadata construction, and the masked atomic backward path. Its routing and final attention are strict causal at fixed sequence geometry.

## Training recipes

The trainer applies Muon to eligible hidden matrices and AdamW to embeddings, biases, norms, and the explicitly special DSQG phase, scale-embedding, and NPCI groups. Those special groups retain their validated learning-rate multipliers. Both recipes use warmup-stable-cosine-decay (WSD) to a 0.1× floor.

| Recipe | LR | BS × GA | Updates | Input positions | WSD |
|---|---:|---:|---:|---:|---:|
| `eb210-1b` (default) | `3.0e-4` | `15 × 14` | 2,325 | 999,936,000 | 116 / 1,860 / 349 |
| `eb84-400m` | `5.1e-4` | `14 × 6` | 2,300 | 395,673,600 | 115 / 1,840 / 345 |

`eb84-400m` is retained as an explicit recipe, not the universal default. Its confirmation run was healthy and improved aggregate promotion BPB, but exceeded the predeclared `0.005` per-source parity tolerance on one promotion source and one sealed source.

## Requirements

DWARF training requires an NVIDIA GPU, a CUDA-enabled PyTorch installation, and Triton.  Create an environment and install the matching PyTorch wheel using the [official selector](https://pytorch.org/get-started/locally/), then install the remaining runtime dependency:

```bash
python -m pip install -r requirements.txt
```

## Dataset contract

The trainer accepts a local `torch.save` artifact containing an `int32` or `int64` tensor shaped `[rows, sequence_length]`. A dictionary containing that tensor under `train`, `input_ids`, `tokens`, or `data` is also accepted. Rows are shuffled deterministically without replacement from `--seed`.

Use the tokenizer tracked at:

```text
tokenizers/dwarf_bpe_v32768_tokenizer.json
```

The tokenizer has 32,768 contiguous token IDs. Dataset construction remains separate so users can choose their own corpus and packing policy.

## Train

The standard model is D=512, H=8, L=10, FFN=1536, sequence length 2048, and vocabulary size 32,768. A one-update smoke needs at least 210 packed rows because it retains the default recipe's effective batch:

```bash
python train/train_dwarf.py \
  --dataset /absolute/path/to/packed_tokens.pt \
  --output-dir runs/dwarf-smoke \
  --stop-after 1
```

Run the full default recipe by omitting `--stop-after`. Select the 400M recipe explicitly with `--recipe eb84-400m`. Resume a staged or interrupted run with:

```bash
python train/train_dwarf.py \
  --dataset /absolute/path/to/packed_tokens.pt \
  --output-dir runs/dwarf-eb210 \
  --resume runs/dwarf-eb210/dwarf_step_0000582.pt
```

EB210 saves at 25%, 50%, 75%, and final; EB84 saves the final checkpoint. `--save-every N` adds an interval, and bounded runs always save their stopping step. Checkpoints are atomically replaced and contain model, optimizer, recipe/configuration, RNG, and dataset-identity state required for resume.

## Scope and limitations

This source reproduces the active architecture and validated optimization recipes, not the private training corpus or evaluation harness. Model quality still depends materially on corpus quality, packing, source mixture, and evaluation protocol.

## License

[Apache-2.0](LICENSE)
