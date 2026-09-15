<p align="center">
  <img src="dwarf-logo.png" alt="DWARF logo: a dwarf holding two axes" width="360">
</p>

# DWARF-v2

DWARF-v2 is an experimental causal language-model architecture that combines sparse local token mixing with routed global attention. Its goal is to give a model access to distant context without using dense attention in every block.

This repository contains the model definition, CUDA/Triton kernels, tokenizer, and a self-contained trainer. It does not include pretrained weights or training data.

The bundled profile is a starting point. Model dimensions, context length, and the training recipe are configurable in `train/train_dwarf.py`.

## How it works

DWARF interleaves two kinds of blocks:

- **DSQG** gathers information from a bounded set of causal offsets instead of attending to every earlier token.
- **HISA** combines a causal local window with a small set of routed chunks from earlier in the sequence.
- **Causal EMA packets** give selected HISA layers a compressed view of prior state. In the final mixer, base keys decide where to look while packet-adjusted keys and values determine how the selected content is read.

All routing and mixing paths are autoregressive. The reference profile is D512/H8/L24 with 21 DSQG blocks, 3 HISA mixers executing K=3 routed pages from M=6 exact-reranked candidates, a 1,344-wide SwiGLU FFN (legacy `ffn_dim=2048`), a 2,047-token training context, a 32,768-token vocabulary, and about 100 million parameters.

## Quick start

Training requires Linux, an NVIDIA GPU, CUDA-enabled PyTorch, and Triton. Install PyTorch using the [official selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

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

## Data

The trainer reads a local `torch.save` artifact containing an `int32` or `int64` tensor shaped `[rows, sequence_length]`. The tensor may also be stored in a dictionary under `train`, `input_ids`, `tokens`, or `data`. The reference profile expects 2,048-token rows, which provide 2,047 next-token targets each.

Use the included tokenizer when preparing rows:

```text
tokenizers/dwarf_bpe_v32768_tokenizer.json
```

Rows should be shuffled before they are saved; the trainer consumes them sequentially. Before training, it checks the tensor shape, dtype, token range, tokenizer identity, and dataset hash.

## Training

For the bundled configuration, run a one-update smoke test with at least 128 packed rows:

```bash
python train/train_dwarf.py \
  --dataset /absolute/path/to/packed_tokens.pt \
  --output-dir runs/dwarf-smoke \
  --stop-after 1
```

The canonical recipe adopts the AutoDWARF `09102026/t00042` optimization bundle while retaining **B16/GA8 (128 rows/update)**, not the research run's B4/GA1. It uses peak LR **0.00135**, weight decay **0.2** on decay-enabled groups, Muon momentum **0.9** and clipping **0.25**, and AdamW betas **(0.95, 0.97625)** with clipping **1.0**. Compiled BF16 training and 2,048-token fused-CE chunks remain enabled. The canonical seed remains 42; the winning research replicate used 137. The short-budget result is not a long-run quality qualification of this batch geometry.

The WSD schedule uses **3.125% warmup, 25% terminal cosine cooldown, and a 10% LR floor**, rounded to whole optimizer updates. For fixed rows, the default horizon remains 76,331 updates (approximately 20 billion targets); use `--schedule-steps N` to size a different full run. `--stop-after` is only a temporary stopping cap and does **not** resize that schedule. `--save-every N` adds checkpoint intervals.

The mixed-length route uses `--mixed-length-manifest` and `--total-target-budget` instead of `--dataset`. It resolves the schedule from the complete manifest-backed target-budget plan, independently of `--mixed-length-max-steps`. The resolved recipe is recorded in checkpoints and checked on resume; changing the horizon is not a strict resume.

Both routes always save a **full-state pre-decay checkpoint** after the last full-LR update, before the first decayed update, regardless of periodic save settings. Decay cannot proceed if that atomic save fails. Its step is reported as `pre_decay_step`; the file is `dwarf_step_<step:07d>.pt` or `mixed_length_step_<step:07d>.pt`. The endpoint is saved separately, preserving the model, optimizer, RNG, and data cursor at the branch point for a future explicitly configured continuation.

Checkpoints contain the model, optimizer, RNG state, source hashes, tokenizer identity, and dataset identity. Resume with the same dataset and output directory:

```bash
python train/train_dwarf.py \
  --dataset /absolute/path/to/packed_tokens.pt \
  --output-dir runs/dwarf-train \
  --resume runs/dwarf-train/dwarf_step_0001909.pt
```

Run `python train/train_dwarf.py --help` for the full command-line interface.

> [!NOTE]
> DWARF-v2 is research software. Model quality depends heavily on the training corpus, packing strategy, source mixture, and evaluation protocol.

## License

[Apache-2.0](LICENSE)
