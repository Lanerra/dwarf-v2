# DWARF-v2

DWARF-v2 is a compact research implementation of a causal language-model architecture built from Dynamic Sparse Query-Gather (DSQG) blocks and one L3 global mixer.  It is for reproducing and extending DWARF training runs, not production deployment.

The public tree intentionally contains only the runtime source needed to construct and train the active architecture.  Datasets, checkpoints, launch scripts, evaluation outputs, Hugging Face staging files, diagnostics, and retired experiments remain local and are not part of this repository.

## Included topology

- Nine triadic DSQG sparse-attention blocks using the canonical 96-offset lattice.
- A causal-EMA interference injection in the final pre-L3 DSQG block.
- One L3 global mixer, selectable as either:
  - **strict-causal V16 HISA** (`--global-mixer hisa`); or
  - **full causal SDPA** (`--global-mixer fa`, the default), the FA@L3 topology used by DWARF-55M-Base.

## Trainer options

```text
     Required I/O
       --dataset PATH — Required local PyTorch .pt packed-token dataset. It must be an int32/int64 tensor shaped [rows, seq_len], or a dict containing one under input_ids, tokens, or data.
       --output-dir PATH — Required directory for model-only checkpoints, named dwarf_step_<step>.pt.
     
     Runtime/topology
       --device DEVICE — PyTorch device string; default cuda. The trainer rejects a CUDA device if CUDA is unavailable; its comment reserves CPU for small FA smoke tests.                              
       --global-mixer {hisa,fa} — Default hisa. Chooses the one L3 global block: strict-causal V16 HISA (hisa) or dense full-causal SDPA (fa, the published DWARF-55M-Base topology).                   
       --vocab-size INT — Default 50282; embedding/output vocabulary size.
       --embedding-dim INT — Default 512; hidden width. Must divide evenly by --num-heads.
       --num-heads INT — Default 8; attention heads.
       --ffn-dim INT — Default 1536; hidden width of each two-layer GELU FFN.
       --seq-len INT — Default 2048; required packed-row length. Training shifts each row by one, so the model predicts seq_len - 1 targets per row.
       --num-chunks INT — Default 32; HISA’s sequence chunk count. Relevant only when --global-mixer hisa.
       --top-k-chunks INT — Default 4; number of causal HISA chunks selected per query. Relevant only with HISA.
       --hisa-top-m-tokens INT — Default 64; token candidates retained within HISA’s selected chunks. Relevant only with HISA.
       --dropout FLOAT — Default 0.1; dropout in embeddings, attention-output path, and FFNs.

     Training/optimization
       --batch-size INT — Default 1; randomly sampled packed rows per update. The dataset must contain at least this many rows.
       --max-steps INT — Default 1000; number of optimizer updates.
       --save-every INT — Default 100; checkpoint interval; the final step is always saved even if it is not on the interval.
       --learning-rate FLOAT — Default 3e-4; AdamW learning rate.
       --weight-decay FLOAT — Default 0.1; AdamW weight decay.
       --grad-clip-norm FLOAT — Default 1.0; maximum global gradient norm passed to clip_grad_norm_.
       --seed INT — Default 42; seed supplied to torch.manual_seed, governing initialization and random row sampling.
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
