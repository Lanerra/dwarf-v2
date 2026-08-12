<p align="center">
  <img src="dwarf-logo.png" alt="DWARF logo: a dwarf holding two axes" width="360">
</p>

# DWARF-v2

DWARF-v2 is an experimental causal language-model architecture built around sparse token mixing. It combines local Dynamic Sparse Query-Gather (DSQG) blocks with strict-causal Hierarchical Sparse Attention (HISA) global mixers.

This repository contains the architecture and training source required to train your own DWARF model. It does not include a trained model, model weights, checkpoints, or training data.

The included trainer provides a working reference configuration, but the source is intended to be adapted. Users can choose their own model width (D), depth (L), head count (H), feed-forward width (FFN), context length, and training recipe.

## Architecture

DWARF interleaves DSQG blocks with HISA global mixers. HISA combines a strict-causal local lane with routed global context, while a causal exponential-moving-average summary can provide additional context to a global mixer. All selection and mixing paths preserve autoregressive causality.

The bundled reference configuration uses:

- Model width: 512
- Attention heads: 8
- Blocks: 12 (10 DSQG, 2 HISA)
- Feed-forward width: 2,048 with a 1,344-unit SwiGLU hidden layer
- Context length: 2,047 input tokens from 2,048-token packed rows
- Vocabulary: 32,768 tokens
- Parameters: 58,591,773 total; 58,591,757 trainable

Its HISA mixers appear at block indices 3 and 9 and use a 64-token local lane. Model and recipe defaults are defined by `DwarfConfig` and `TrainRecipe` in `train/train_dwarf.py`. The checked-in self-test fingerprints this reference configuration, so intentional configuration changes must update the corresponding contract values.

## Installation

Training requires Linux, an NVIDIA GPU, CUDA-enabled PyTorch, and Triton. Create a virtual environment, install the appropriate PyTorch build using the [official selector](https://pytorch.org/get-started/locally/), and install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the bundled architecture and kernel contracts before starting a run:

```bash
python train/train_dwarf.py --self-test
```

## Dataset format

The trainer reads a local `torch.save` artifact containing an `int32` or `int64` tensor shaped `[rows, sequence_length]`. The tensor may also be stored in a dictionary under `train`, `input_ids`, `tokens`, or `data`. The bundled configuration uses a sequence length of 2,048.

Pack the data with the included tokenizer:

```text
tokenizers/dwarf_bpe_v32768_tokenizer.json
```

With the included tokenizer, token IDs must be in `[0, 32768)`. The trainer validates the tensor shape, dtype, token range, tokenizer identity, and dataset hash before training. Rows are consumed in deterministic sequential order. A different vocabulary requires a matching tokenizer and configuration.

## Training

For the bundled configuration, run a one-update smoke test with at least 210 packed rows:

```bash
python train/train_dwarf.py \
  --dataset /absolute/path/to/packed_tokens.pt \
  --output-dir runs/dwarf-smoke \
  --stop-after 1
```

Omit `--stop-after` to use the bundled reference training recipe:

| Learning rate | Batch × accumulation | Updates | Packed rows | Input tokens |
|---:|---:|---:|---:|---:|
| `3.0e-4` | `15 × 14` | 4,653 | 977,130 | 2,001,162,240 |

The trainer uses Muon and AdamW with a warmup-stable-decay schedule. It writes resumable checkpoints after warmup, at 25%, 50%, 75%, and at the end of the run. Use `--save-every N` for additional intervals.

Resume from a checkpoint with the same dataset and output directory:

```bash
python train/train_dwarf.py \
  --dataset /absolute/path/to/packed_tokens.pt \
  --output-dir runs/dwarf-train \
  --resume runs/dwarf-train/dwarf_step_0001164.pt
```

Checkpoints include the model, optimizer, training configuration, random-number-generator state, source hashes, tokenizer identity, and dataset identity. Resume validation rejects incompatible state rather than partially loading it.

> [!NOTE]
> DWARF-v2 is research software. Model quality depends heavily on the training corpus, packing strategy, source mixture, and evaluation protocol.

## License

[Apache-2.0](LICENSE)
