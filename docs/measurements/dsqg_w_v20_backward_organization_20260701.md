# DSQG-W V20-style backward organization and scheduling

Date: 2026-07-01
Worktree: `/home/dlewis3/.config/superpowers/worktrees/DWARF-v2/dsqg-w-triton-true-bwd-coder`

## Change

Ported V20's scheduling discipline and backward-organization hooks to the DSQG-W compact-read Triton path without assuming V20 fixed-offset overlap-slab topology.

Implemented:

- Central `_dsqg_w_triton_schedule(head_dim, device)` helper for DSQG-W Triton row/head kernels.
- Forward and backward launches now use centralized `BLOCK_HD`, `num_warps`, and `num_stages` selection.
- The compact-read backward kernel accepts compile-time `COMPUTE_QUERY` and `COMPUTE_SOURCE` flags.
- Default remains the fused monolithic true-backward launch because it is faster in trainer windows.
- V20-style split query/source launches are available for profiling with:

```bash
DWARF_DSQG_W_TRITON_BACKWARD_ORGANIZATION=v20_split
```

This preserves the current DSQG-W invariant:

- saved fp32 LSE `[B,N,H]`;
- score/probability recomputation in backward;
- no materialized backward `[B,N,J,H]` probability tensor;
- no V20 overlap-slab assumption over arbitrary DSQG-W candidate metadata.

## Microbench evidence

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/bench_dsqg_w_triton_sourcewise.py
```

RTX 4090, `B=1,N=256,D=512,H=8,HD=64,J=16`:

| path | forward ms | fwd+bwd ms | peak MB fwd+bwd |
|---|---:|---:|---:|
| dense | 1.849 | 5.509 | 171.9 |
| eager sourcewise | 10.460 | 19.301 | 163.6 |
| Triton compact-read true backward, default monolithic | 1.215 | 3.996 | 170.0 |
| Triton compact-read PyTorch VJP fallback | — | 22.922 | 170.0 |

Materialization telemetry:

```text
triton_true_backward: 1.0
triton_backward_probs_materialized: 0.0
triton_backward_lse_saved: 1.0
triton_backward_reduction_buffer_bytes: 0.0
triton_backward_v20_split_kernels: 0.0
triton_schedule_block_hd: 64.0
triton_schedule_num_warps: 1.0
triton_schedule_num_stages: 2.0
```

Opt-in split-backward microbench:

```bash
DWARF_DSQG_W_TRITON_BACKWARD_ORGANIZATION=v20_split \
  PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/bench_dsqg_w_triton_sourcewise.py
```

Result: `triton_sourcewise_fwd_bwd = 4.268 ms`, with query/source split telemetry enabled. The split is correct but slower than the default fused launch for this shape because it recomputes the score/probability pass twice.

## Trainer smoke evidence

Default monolithic true-backward, warmed 20-step bench-only run:

```bash
DWARF_BENCH_ONLY=1 PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python \
  scripts/run_dsqg_w_full_training.py \
  --output-dir results/dsqg_w_triton_v20_org_20260701_stage2_triton20_warm \
  --run-name v20_org_stage2_triton20_warm \
  --max-acc-steps 20 --train-seqs 32 --val-seqs 4 \
  --batch-size 1 --grad-accum 1 --passkey-trials 0 --execute --triton-sourcewise
```

Result:

```text
first_step_ms=1096.1
trailing_avg_ms=386.8
steady_tok_s=5293
peak_vram=3122MB
```

For context from the same measurement batch:

| path | trailing_avg_ms | steady tok/s | peak VRAM |
|---|---:|---:|---:|
| dense DSQG-W | 363.1 | 5638 | 3801 MB |
| eager sourcewise | 1862.9 | 1099 | 3507 MB |
| Triton default monolithic true-backward | 386.8 | 5293 | 3122 MB |

The split query/source organization is kept as an explicit profiling mode, not the default, because it loses to the fused monolithic launch in trainer timing.

## Verification

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py \
  tests/test_dsqg_w_trainer_insertion.py \
  tests/test_dsqg_w_parity_harness.py \
  tests/test_dsqg_w_full_run_launcher.py -q
```

Result:

```text
36 passed in 5.55s
```

Additional checks:

```text
py_compile kernels/dsqg_w/dsqg_w_mvp.py scripts/bench_dsqg_w_triton_sourcewise.py tests/test_dsqg_w_candidate_provider_optimization.py: pass
git diff --check: pass
```
