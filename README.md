<p align="center">
  <img src="dwarf-logo.png" alt="DWARF logo: a dwarf holding two axes" width="360">
</p>

# DWARF-v2

DWARF-v2 is a compact causal language-model architecture built from Dynamic Sparse Query-Gather (DSQG) blocks and one L3 global mixer. This repository is the minimal source needed to train the active architecture.

The public tree intentionally contains only the runtime source needed to construct and train the active architecture.  Datasets, checkpoints, launch scripts, evaluation outputs, Hugging Face staging files, diagnostics, and retired experiments remain local and are not part of this repository.

## Architecture

- Nine triadic DSQG blocks use disjoint thirds of the canonical offset lattice, centered FP32 virtual-key scale embeddings, and one online sparse-softmax traversal over each retained causal offset.
- A causal-EMA interference packet constructed at the L3 boundary and injected only into the global mixer's K/V streams.
- One strict-causal V18 HISA global mixer at L3. Per-DSQG NPCI state is not part of the public architecture.

The HISA kernel uses a 64-token local lane, 16-token selector tiles, blocked bounded-memory metadata construction, the Triton global backend, and the masked atomic backward path with `BLOCK_Q=16`. Token selection is `auto`, local attention is `flex`, and routing diagnostics are disabled. These values are explicit `DwarfConfig` fields, are passed directly to HISA rather than resolved from environment variables, and are serialized with HISA semantic/execution metadata in every resumable checkpoint.

## Training recipes

The trainer applies Muon to eligible hidden matrices and AdamW to embeddings, biases, norms, DSQG scale embeddings, positional parameters, the global HISA NPCI pair, routing parameters, and causal-EMA packet parameters. It uses warmup-stable-cosine-decay (WSD) to a 0.1× floor.

| Recipe | LR | BS × GA | Updates | Input positions | WSD |
|---|---:|---:|---:|---:|---:|
| public default | `3.0e-4` | `15 × 14` | 4,653 | 2,001,162,240 | 233 / 3,722 / 698 |

## Requirements

DWARF training requires an NVIDIA GPU, a CUDA-enabled PyTorch installation, and Triton.  Create an environment and install the matching PyTorch wheel using the [official selector](https://pytorch.org/get-started/locally/), then install the remaining runtime dependency:

```bash
python -m pip install -r requirements.txt
```

## Dataset contract

The trainer accepts a local `torch.save` artifact containing an `int32` or `int64` tensor shaped `[rows, sequence_length]`. A dictionary containing that tensor under `train`, `input_ids`, `tokens`, or `data` is also accepted. The public recipe consumes a deterministic sequential prefix and binds the dataset path, SHA-256, size, row count, tokenizer identity, and selected row count into the checkpoint contract.

Use the tokenizer tracked at:

```text
tokenizers/dwarf_bpe_v32768_tokenizer.json
```

The tokenizer has 32,768 contiguous token IDs and reserves 57 atomic control markers at IDs 0-56, including BOS/EOS/PAD/UNK/EOD, ChatML roles, reasoning sections, tool calls, FIM/repository boundaries, and optional image/video markers. Dataset construction remains separate so users can choose their own corpus and packing policy.

## Train

The standard model is D=512, H=8, L=10, FFN=2400 (SwiGLU hidden width 1600), sequence length 2048, and vocabulary size 32,768. A one-update smoke needs at least 210 packed rows because it retains the default recipe's effective batch:

```bash
python train/train_dwarf.py \
  --dataset /absolute/path/to/packed_tokens.pt \
  --tokenizer tokenizers/dwarf_bpe_v32768_tokenizer.json \
  --output-dir runs/dwarf-smoke \
  --stop-after 1
```

Run the full default recipe by omitting `--stop-after`. Resume a staged or interrupted schema-v2 run with:

```bash
python train/train_dwarf.py \
  --dataset /absolute/path/to/packed_tokens.pt \
  --tokenizer tokenizers/dwarf_bpe_v32768_tokenizer.json \
  --output-dir runs/dwarf-public \
  --resume runs/dwarf-public/dwarf_step_0001164.pt
```

The trainer saves at warmup completion, 25%, 50%, 75%, and final. `--save-every N` adds an interval, and bounded runs always save their stopping step. Schema-v2 checkpoints are atomically replaced and contain model, optimizer, recipe/configuration, resolved HISA policy, source hashes, RNG, dataset identity, and an explicit native-or-migrated lineage receipt required for fail-closed resume.

The immediately preceding public v1 schema contained 18 unreachable per-DSQG NPCI tensors. Those exact hash-pinned checkpoints can be migrated only by adding `--migrate-legacy-v1-default-hisa-policy` to the resume command. The flag explicitly asserts that the unreceipted v1 HISA execution policy was the canonical `triton/auto/flex/BLOCK_Q=16/atomic_masked` policy. Any other source manifest, model-key set, optimizer layout, policy, scheduled LR, or lineage receipt still fails closed. Migrated checkpoints retain the exact migration receipt in subsequent saves.

Run `python train/train_dwarf.py --self-test` before training to verify the canonical parameter count, seeded state fingerprint, HISA policy, optimizer partition, and public kernel contracts.

## Scope and limitations

This source reproduces the active architecture and validated optimization recipes, not the private training corpus or evaluation harness. Model quality still depends materially on corpus quality, packing, source mixture, and evaluation protocol.

## License

[Apache-2.0](LICENSE)
