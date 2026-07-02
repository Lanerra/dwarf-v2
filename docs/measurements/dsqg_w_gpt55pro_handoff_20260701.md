# GPT-5.5 Pro handoff: DSQG-W throughput is still dominated by W integration overhead

You are a senior CUDA/Triton/PyTorch performance engineer. The task is to explain and propose a concrete optimization plan for DWARF DSQG-W throughput. Do **not** assume V20 fixed-offset overlap-slab geometry applies wholesale; prove any geometry assumption from DSQG-W candidate metadata before using it.

## Repo / branch context

Worktree:

```text
/home/dlewis3/.config/superpowers/worktrees/DWARF-v2/dsqg-w-triton-true-bwd-coder
```

Branch:

```text
perf/dsqg-w-triton-true-backward
```

Relevant recent commit baseline:

```text
025b5c8 perf: add DSQG-W Triton compact read backward
```

Current uncommitted changes add V20-style scheduling and opt-in split backward organization while keeping fused true-backward default.

## Files to inspect

Must inspect:

```text
kernels/dsqg_w/dsqg_w_mvp.py
scripts/bench_dsqg_w_triton_sourcewise.py
scripts/run_dsqg_w_full_training.py
train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py
tests/test_dsqg_w_candidate_provider_optimization.py
tests/test_dsqg_w_trainer_insertion.py
tests/test_dsqg_w_parity_harness.py
kernels/dsqg_attention_v20_bf16_se.py
```

Useful measurement notes:

```text
docs/measurements/dsqg_w_triton_true_backward_20260701.md
docs/measurements/dsqg_w_v20_backward_organization_20260701.md
```

## What is already true

DSQG-W compact-read Triton path already has:

- compact read-slot forward;
- saved fp32 LSE `[B,N,H]`;
- backward recomputes scores/probabilities;
- no materialized backward `[B,N,J,H]` probability tensor;
- no materialized `[B,N,J,D]` candidate states in the fast path;
- direct atomics for source K/V and small embedding/bias gradients.

The latest code also has:

- centralized `_dsqg_w_triton_schedule(head_dim, device)`;
- opt-in split backward via `DWARF_DSQG_W_TRITON_BACKWARD_ORGANIZATION=v20_split`;
- default remains fused monolithic true-backward because split is slower.

## Current evidence

### Microbench

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/bench_dsqg_w_triton_sourcewise.py
```

RTX 4090, `B=1,N=256,D=512,H=8,HD=64,J=16`:

| path | fwd+bwd ms | peak MB |
|---|---:|---:|
| dense | ~5.5 | ~171.9 |
| eager sourcewise | ~19.3 | ~163.6 |
| Triton compact-read true backward default | ~4.0 | ~170.0 |
| PyTorch VJP fallback | ~22.9 | ~170.0 |

Opt-in split backward:

```bash
DWARF_DSQG_W_TRITON_BACKWARD_ORGANIZATION=v20_split \
  PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/bench_dsqg_w_triton_sourcewise.py
```

Result: ~4.27 ms fwd+bwd, correct but slower than fused default because it recomputes score/probability twice.

### Trainer 20-step bench-only comparison

Common shape/config from logs:

```text
DSQG-W recomposer sites=layer_2,layer_6,final
J<=16 bottleneck=64
gate_init=-2.5 fuse_init_std=0.02
sourcewise/triton_sourcewise varied
width_cell=False typed_mixer=False query_type_bias=False
dsr_candidates=True candidates=DSR_SELECTED_QUESTION_L3_SKIP_NULL
```

Measured rows:

| run | trailing_avg_ms | steady tok/s | peak VRAM |
|---|---:|---:|---:|
| no DSQG-W baseline (`--disable-dsqg-w`) | 97.8 | 20923 | 2816 MB |
| dense DSQG-W | 347.4-363.1 | 5638-5893 | 3801 MB |
| eager sourcewise DSQG-W | 1749-1863 | 1099-1170 | 3507 MB |
| Triton sourcewise true-backward warmed | 372-387 | 5293-5499 | 3122 MB |

Key observation: Triton sourcewise is close to dense DSQG-W and much faster than eager sourcewise, but DSQG-W itself still makes the trainer ~3.7x slower than the no-W baseline (`~20.9K tok/s -> ~5.3-5.9K tok/s`).

## Main question

Where is the remaining DSQG-W overhead actually coming from?

Do not stop at the compact read microkernel. The compact read path is no longer the obvious bottleneck: microbench beats dense and trainer is near dense-W speed. The target is to recover much more of the no-W baseline throughput, ideally toward 20K+ tok/s before a 200K-sequence comparator.

## Hypotheses to test / rank

1. DSQG-W overhead is dominated by per-site projection/norm/fuse/read-mix/outside-kernel PyTorch work, not the score/read kernel itself.
2. DSQG-W is inserted at three sites (`layer_2,layer_6,final`), so even modest per-site overhead multiplies.
3. Candidate metadata build/packing or HISA/DSR selected-token packing may dominate at trainer scale even if the compact read microbench is good.
4. Autograd graph fragmentation/kernel-launch count around DSQG-W MLP/norm/projection/read_mix may dominate steady trainer timing.
5. V20 overlap-slab helps only if candidate token indices have fixed-offset/contiguous-slab geometry. Current DSR_SELECTED_QUESTION_L3_SKIP_NULL candidates may not; verify before proposing a slab kernel.
6. Dense W path is already much slower than no-W, so overlap slab cannot be the whole answer unless score/read dominates dense W overhead.

## What I want from you

Produce:

1. A ranked bottleneck diagnosis with specific predicted signatures.
2. A minimal profiler plan using `torch.profiler`/NVTX/nsys or existing trainer hooks to decompose one step into:
   - candidate metadata build/packing;
   - per-site norm/projections;
   - DSQG-W score/read kernel;
   - read_mix/fuse/output MLP;
   - backward source atomics;
   - optimizer/activation-checkpoint effects.
3. A concrete optimization plan with staged PR-sized changes.
4. A verdict on whether V20 overlap-slab should be ported:
   - only yes if you can identify a candidate subset with fixed offset/slab reuse;
   - otherwise propose a different fusion boundary.
5. For every proposed speedup, state the correctness tests needed: forward parity, grad parity, causality/prefix consistency, no forbidden materialization, and trainer smoke.

Hard constraints:

- Preserve DSQG-W semantics; no detach, skipped gradients, dtype narrowing, or changed candidate ordering.
- Preserve D-fed W behavior over HISA/DSR-selected evidence.
- Preserve sites such as `layer_2,layer_6,final` / `6,final`.
- Keep no materialized `[B,N,J,D]` and no backward `[B,N,J,H]` probabilities in the fast path.
- Treat early passkey as non-diagnostic for W; throughput and semantic/composer quality gates are separate.

