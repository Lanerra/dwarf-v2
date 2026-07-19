# DWARF-v2

DWARF-v2 is a compact research implementation of a causal language-model architecture built from Dynamic Sparse Query-Gather (DSQG) blocks and one L3 global mixer.  It is for reproducing and extending DWARF training runs, not production deployment.

The public tree intentionally contains only the runtime source needed to construct and train the active architecture.  Datasets, checkpoints, launch scripts, evaluation outputs, Hugging Face staging files, diagnostics, and retired experiments remain local and are not part of this repository.

## Included topology

- Nine triadic DSQG sparse-attention blocks using the canonical 96-offset lattice.
- A causal-EMA interference injection in the final pre-L3 DSQG block.
- One L3 global mixer, selectable as either:
  - **strict-causal V16 HISA** (`--global-mixer hisa`, the default); or
  - **full causal SDPA** (`--global-mixer fa`), the FA@L3 topology used by DWARF-55M-Base.

## Repository layout

```text
train/train_dwarf.py                         reference training entrypoint
kernels/dsqg_attention_v20_bf16_se.py       triadic DSQG CUDA/Triton kernel
kernels/causal_ema_scan.py                   causal EMA scan used by L2 interference
kernels/hierarchical_sparse_attn_v16_hisa_causal.py  strict-causal HISA L3 mixer
tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json  DWARF-55M tokenizer asset
```

## Requirements

DWARF training requires an NVIDIA GPU, a CUDA-enabled PyTorch installation, and Triton.  Create an environment and install the matching PyTorch wheel using the [official selector](https://pytorch.org/get-started/locally/), then install the remaining runtime dependency:

```bash
python -m pip install -r requirements.txt
```

## Dataset contract

The trainer accepts a local `torch.save` artifact containing an integer tensor with shape `[rows, sequence_length]`.  A dictionary containing that tensor under `input_ids`, `tokens`, or `data` is also accepted.  Dataset construction is intentionally out of scope for this source-only repository.

Use the tokenizer tracked at:

```text
tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json
```

## Run a DWARF training smoke

The standard model is D=512, H=8, L=10, FFN=1536, sequence length 2048, and vocabulary size 50,282.  Start with a small row count and a disposable output directory:

```bash
python train/train_dwarf.py \
  --dataset /absolute/path/to/packed_tokens.pt \
  --output-dir runs/dwarf-smoke \
  --device cuda \
  --batch-size 1 \
  --max-steps 25 \
  --save-every 25
```

For the FA@L3 topology used by DWARF-55M-Base, add:

```bash
--global-mixer fa
```

The trainer writes model-only checkpoints containing `model_state_dict`, the resolved architecture configuration, and the completed step.  It does not package datasets, optimizer state, evaluation labels, or external-run artifacts.

## Scope and limitations

This is an architecture/reference-training implementation.  It does not claim a turnkey reproduction of any particular data mixture, compute budget, optimizer schedule, or benchmark result.  Those choices materially affect model quality and must be specified by a downstream experiment.

## License

[Apache-2.0](LICENSE)