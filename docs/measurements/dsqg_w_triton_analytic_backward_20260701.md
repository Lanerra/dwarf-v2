# DSQG-W Triton compact read-slots + partial analytic backward — 2026-07-01

Branch: `perf/dsqg-w-triton-analytic-backward`

Base commit: `bc930d0 docs: record DSQG-W Triton trainer comparison`

## Scope

This is a safe partial landing, not the final fully analytic Triton backward.

Implemented:

1. Replaced the optimized no-routing sourcewise forward kernel's repeated read projection with a compact read-slot split:
   - one Triton program per `(B,N,H)` computes scores/softmax once for that head row;
   - it stores compact read slots `[B,N,S,D]`, where `S = 1 + len(block.read_type_ids)`;
   - Python applies only the active `read_mix` weight slices from those compact slots.
2. Added a custom autograd node for the compact read-slot kernel:
   - forward uses Triton;
   - backward is analytic at the read-slot boundary: it computes softmax VJP, `dQ/dK/dV`, role/source/type/source-bias grads, and optional query-type-bias grads explicitly in PyTorch tensor ops;
   - PyTorch autograd then carries those gradients through `norm_x`, `norm_c`, `q_proj`, `k_proj`, `v_proj`, `read_mix`, `norm_z`, `fuse`, and `gate`.
3. Removed the default trainer path's call into the previous whole-block PyTorch recompute helper `_dsqg_w_sourcewise_functional_recompute`.

Not implemented:

- No Triton backward kernel for `dQ/dK/dV/read slots` yet.
- The compact read-slot backward still materializes `[B,N,J,H]` scores/probs internally during backward VJP.
- It is correct and trainer-safe, but not faster than the previous whole-block recompute backward in the microbench/trainer runs below.

## RED tests before implementation

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_autograd_uses_compact_read_backward_not_full_recompute_on_cuda \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_no_routing_and_fused_read_mix_do_not_materialize_forbidden_outputs \
  tests/test_dsqg_w_trainer_insertion.py::test_dsqg_w_triton_sourcewise_env_is_training_smoke_accepted_on_cuda -q
```

Expected RED output observed:

```text
FFF                                                                      [100%]
AssertionError: assert 1.0 == 0.0
KeyError: 'dsqg_w_triton_compact_read_slots_materialized'
AssertionError: assert 1.0 == 0.0
3 failed in 2.59s
```

The new tests required:

- no-routing backward telemetry to report `dsqg_w_triton_sourcewise_recompute_backward=0`;
- a monkeypatch of `_dsqg_w_sourcewise_functional_recompute` to not be called;
- gradient parity for `x`, `l3_states`, `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `read_mix.weight`, `norm_z.weight`, `fuse[0].weight`, `fuse[2].weight`, and `gate`;
- materialization telemetry for compact read slots and `dsqg_w_triton_score_recompute_blocks=1`.

## GREEN focused test output

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_autograd_uses_compact_read_backward_not_full_recompute_on_cuda \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_no_routing_and_fused_read_mix_do_not_materialize_forbidden_outputs \
  tests/test_dsqg_w_trainer_insertion.py::test_dsqg_w_triton_sourcewise_env_is_training_smoke_accepted_on_cuda -q
```

Output:

```text
...                                                                      [100%]
3 passed in 2.36s
```

Broader focused suites:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest tests/test_dsqg_w_candidate_provider_optimization.py -q
```

Output:

```text
..........                                                               [100%]
10 passed, 2 warnings in 11.45s
```

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest tests/test_dsqg_w_trainer_insertion.py -q
```

Output:

```text
.............                                                            [100%]
13 passed in 3.45s
```

## CUDA microbench

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/bench_dsqg_w_triton_sourcewise.py
```

Output summary on RTX 4090, shape `B=1,N=256,D=512,H=8,HD=64,J=16`:

| path | forward ms | peak MB |
|---|---:|---:|
| dense forward no-routing | 1.754 | 132.5 |
| eager sourcewise forward no-routing | 11.626 | 87.3 |
| Triton compact read-slots forward no-routing | 1.305 | 83.9 |
| Triton compact read-slots forward with routing | 1.688 | 84.0 |

| path | fwd+bwd ms | peak MB |
|---|---:|---:|
| dense fwd+bwd | 6.150 | 158.3 |
| eager sourcewise fwd+bwd | 20.695 | 150.0 |
| Triton compact read-slots fwd+bwd | 24.889 | 157.3 |

Parity/materialization from the same run:

```text
eager_vs_dense_out_max_abs_diff = 2.384185791015625e-07
triton_vs_eager_out_max_abs_diff = 2.384185791015625e-07
triton_no_route_vs_route_out_max_abs_diff = 0.0
triton_vs_eager_probs_max_abs_diff = 1.7881393432617188e-07
metadata_candidate_state_numel = 0
triton_no_route_probs_materialized = 0.0
triton_read_accum_materialized = 0.0
triton_read_mix_fused = 0.0
triton_compact_read_slots_materialized = 1.0
triton_compact_read_slots = 5.0
triton_score_recompute_blocks = 1.0
```

Interpretation:

- Forward score/softmax repetition was eliminated across output blocks: telemetry reports `score_recompute_blocks=1` instead of the old `ceil(D/16)` output-block repeats.
- Forward is faster than the prior fused-read kernel measurement (`~1.30 ms` here vs `~1.76 ms` in the recompute-backward doc) and still lower VRAM than dense.
- Fwd+bwd is slower than eager sourcewise and slower than the previous whole-block PyTorch recompute backward. The explicit Python/tensor analytic VJP is correct but not yet performance-good.

## 20-step bench-only trainer comparison

Commands used the requested shape:

```bash
DWARF_BENCH_ONLY=1 PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python \
  scripts/run_dsqg_w_full_training.py \
  --max-acc-steps 20 --train-seqs 32 --val-seqs 4 \
  --batch-size 1 --grad-accum 1 --passkey-trials 0 --execute \
  [dense default | --sourcewise | --triton-sourcewise]
```

Dense/eager logs are under:

```text
results/dsqg_w_triton_compact_read_20260701_trainer20/
```

The first Triton attempt in that directory hit a BF16 dtype mismatch in the manual backward scatter and was fixed. The successful Triton retry is under:

```text
results/dsqg_w_triton_compact_read_20260701_trainer20_retry/
```

All successful dense/eager/Triton runs exited 0 with empty wrapper stderr.

| path | first_step_ms | trailing_avg_ms | steady_tok_s | peak_vram |
|---|---:|---:|---:|---:|
| dense DSQG-W | 1089.5 | 373.1 | 5487 | 3801 MB |
| eager sourcewise DSQG-W | 2496.9 | 1890.9 | 1083 | 3507 MB |
| Triton compact read-slots | 2018.2 | 1303.3 | 1571 | 3116 MB |

Interpretation:

- The compact read-slot Triton path trains and is faster than eager sourcewise in the real trainer (`1571` vs `1083` tok/s).
- It is much slower than dense and much slower than the previous Triton recompute trainer result (`4895` tok/s in `dsqg_w_triton_backward_readmix_20260701.md`).
- Peak VRAM remains lower than dense/eager trainer runs (`3116 MB` vs `3801/3507 MB`) but higher than the previous recompute-forward path (`2930 MB`) because compact read slots `[B,N,S,D]` are now stored.

## Materialization status

Avoided in optimized no-routing forward/trainer path:

1. `[B,N,J,D]` candidate states: metadata candidate state tensor stays empty (`metadata_candidate_state_numel=0`).
2. `[B,N,J,H,HD]` candidate projected K/V surfaces: source projections remain rank-4 `[B,N,H,HD]`; candidates are gathered/streamed by metadata.
3. `[B,N,J,H]` probs when `return_routing=False`: telemetry `dsqg_w_triton_probs_materialized=0`; `dsqg_w_probs` absent.
4. `[B,N,(n_types+1)*D]` read_accum: telemetry `dsqg_w_triton_read_accum_materialized=0`.

New bounded intermediate:

- `[B,N,S,D]` compact read slots, where `S=1+len(block.read_type_ids)`. In the microbench `S=5`, so this is `1*256*5*512*4 = 2,621,440 bytes` in fp32. It is not the forbidden full `[B,N,(n_types+1)*D]` surface (`n_types+1=12` here), but it is a real bounded materialization and accounts for part of the VRAM increase.

Still present in non-hot or backward paths:

- `[B,N,J,H]` probs are still stored for `return_routing=True` telemetry.
- The compact read-slot backward computes `[B,N,J,H]` scores/probs internally for the softmax VJP. This is the next target for a true Triton analytic backward kernel.

## HISA status

HISA retrieval/selector code was not touched. The compact token metadata and Stage-2 rep path remain unchanged; this patch only changes DSQG-W sourcewise code, DSQG-W tests, and the DSQG-W microbench script.

## Required final verification

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py \
  tests/test_dsqg_w_trainer_insertion.py \
  tests/test_dsqg_w_parity_harness.py \
  tests/test_dsqg_w_full_run_launcher.py -q
```

Output:

```text
..................................                                       [100%]
34 passed in 5.48s
```

```bash
/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m py_compile \
  kernels/dsqg_w/dsqg_w_mvp.py \
  train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py \
  scripts/run_dsqg_w_full_training.py \
  scripts/bench_dsqg_w_triton_sourcewise.py
```

Output: exit code 0, no stdout/stderr.

```bash
git diff --check
```

Output: exit code 0, no stdout/stderr.

## Next bottleneck / next step

This branch proves the clean split and gradient surface, but the backward is not the desired speed path. The next step should be a real Triton backward kernel for the compact read-slot operation:

1. forward should optionally save compact LSE `[B,N,H]`;
2. backward should recompute scores once per `(B,N,H)` tile, use LSE to form probabilities without storing `[B,N,J,H]`, and accumulate `dQ/dK/dV` plus read-slot grads in Triton;
3. parameter reductions for role/source embeddings and biases can either use atomics directly or a two-stage reduction buffer; document any bounded buffer shape explicitly;
4. keep the compact read-slot forward if the forward speed win survives longer trainer windows.
