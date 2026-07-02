# DSQG-W Triton compact read-slots true backward — 2026-07-01

Branch: `perf/dsqg-w-triton-true-backward`

Base commit: `e21c007 perf: add DSQG-W compact read backward prototype`

## Scope

Implemented the default no-routing DSQG-W Triton sourcewise path with a real Triton VJP for the compact read-slot operation.

What changed:

1. Forward compact read-slot kernel now saves LSE `[B,N,H]` when autograd is active.
2. Default compact read-slot backward launches a Triton kernel per `(B,N,H)` row/head.
3. Backward recomputes scores in-kernel, uses saved LSE to form probabilities, and never stores `[B,N,J,H]` backward probabilities.
4. Backward accumulates:
   - `dQ`
   - source-projected `dK`/`dV` for final/L3/summary source tensors
   - role/source key embedding grads
   - type/source bias grads
   - query-type-bias projection grads when enabled
   - compact read-slot VJP from `grad_read_slots`
5. The previous Python compact-read VJP remains available only when `DWARF_DSQG_W_TRITON_COMPACT_READ_BACKWARD=pytorch` is set during backward.
6. Dense DSQG-W and eager sourcewise behavior are preserved. Triton sourcewise remains opt-in via `DWARF_DSQG_W_TRITON_SOURCEWISE=1` / `--triton-sourcewise`.

No HISA retrieval/selector semantics or HISA code were touched.

## RED test before implementation

New focused CUDA test added:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_default_backward_uses_true_kernel_and_no_backward_probs_on_cuda -q
```

Observed RED output before implementation:

```text
F                                                                        [100%]
KeyError: 'dsqg_w_triton_true_backward'
1 failed, 1 warning in 2.44s
```

The test requires:

- default backward telemetry `dsqg_w_triton_true_backward=1.0`;
- `dsqg_w_triton_backward_probs_materialized=0.0`;
- `dsqg_w_triton_backward_lse_saved=1.0`;
- no bounded reduction buffer (`dsqg_w_triton_backward_reduction_buffer_bytes=0.0`);
- no whole-block PyTorch recompute helper call;
- no explicit Python compact-read VJP fallback call;
- non-power-of-two head dim coverage (`D=60,H=4,HD=15`, padded `BLOCK_HD=16`);
- CUDA gradient parity for `x`, `l3_states`, `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `read_mix.weight`, `role_key.weight`, `source_key.weight`, `type_bias`, `source_bias`, `query_type_bias.weight`, `norm_z.weight`, `fuse[0].weight`, `fuse[2].weight`, and `gate`.

## Focused GREEN verification

```bash
/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m py_compile kernels/dsqg_w/dsqg_w_mvp.py && \
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_default_backward_uses_true_kernel_and_no_backward_probs_on_cuda -q
```

Output:

```text
.                                                                        [100%]
1 passed, 2 warnings in 7.12s
```

Focused regression/smoke set:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_autograd_uses_compact_read_backward_not_full_recompute_on_cuda \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_no_routing_and_fused_read_mix_do_not_materialize_forbidden_outputs \
  tests/test_dsqg_w_trainer_insertion.py::test_dsqg_w_triton_sourcewise_env_is_training_smoke_accepted_on_cuda -q
```

Output:

```text
...                                                                      [100%]
3 passed, 5 warnings in 39.15s
```

Full candidate/provider optimization file:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest tests/test_dsqg_w_candidate_provider_optimization.py -q
```

Output:

```text
...........                                                              [100%]
11 passed, 2 warnings in 11.05s
```

## CUDA microbench

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/bench_dsqg_w_triton_sourcewise.py
```

Device/shape:

```text
NVIDIA GeForce RTX 4090
B=1, N=256, D=512, H=8, HD=64, J=16
```

Forward timings:

| path | ms | peak MB |
|---|---:|---:|
| dense forward no-routing | 1.690 | 146.2 |
| eager sourcewise forward no-routing | 9.922 | 100.9 |
| Triton compact read-slots forward no-routing | 1.125 | 97.5 |
| Triton compact read-slots forward with routing | 1.645 | 97.6 |

Fwd+bwd timings:

| path | ms | peak MB |
|---|---:|---:|
| dense fwd+bwd | 5.372 | 171.9 |
| eager sourcewise fwd+bwd | 18.832 | 163.6 |
| Triton compact read-slots + true Triton backward | 3.528 | 170.0 |
| Triton compact read-slots + PyTorch VJP fallback | 21.895 | 170.0 |

Parity/materialization from the same run:

```text
eager_vs_dense_out_max_abs_diff = 2.384185791015625e-07
triton_vs_eager_out_max_abs_diff = 2.384185791015625e-07
triton_no_route_vs_route_out_max_abs_diff = 0.0
triton_vs_eager_probs_max_abs_diff = 1.7881393432617188e-07
metadata_candidate_state_numel = 0
triton_no_route_probs_materialized = 0.0
triton_read_accum_materialized = 0.0
triton_compact_read_slots_materialized = 1.0
triton_compact_read_slots = 5.0
triton_score_recompute_blocks = 1.0
triton_true_backward = 1.0
triton_backward_probs_materialized = 0.0
triton_backward_lse_saved = 1.0
triton_backward_reduction_buffer_bytes = 0.0
```

Interpretation:

- The true Triton backward is faster than dense in the standalone DSQG-W microbench (`3.53 ms` vs `5.37 ms`) and much faster than eager/sourcewise fallback (`18.83`/`21.90 ms`).
- LSE `[B,N,H]` is saved as fp32: for the bench shape, `1*256*8*4 = 8192 bytes`.
- No reduction buffers are used; parameter grads are accumulated with direct atomics.

## 20-step bench-only trainer comparison

Commands used distinct output dirs under `results/dsqg_w_triton_true_backward_20260701_trainer20_*`:

```bash
DWARF_BENCH_ONLY=1 PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python \
  scripts/run_dsqg_w_full_training.py \
  --output-dir results/dsqg_w_triton_true_backward_20260701_trainer20_[dense|sourcewise|triton] \
  --max-acc-steps 20 --train-seqs 32 --val-seqs 4 \
  --batch-size 1 --grad-accum 1 --passkey-trials 0 --execute \
  [default | --sourcewise | --triton-sourcewise]
```

All three wrappers exited 0; trainer stderr logs were empty.

| path | first_step_ms | trailing_avg_ms | steady_tok_s | peak_vram |
|---|---:|---:|---:|---:|
| dense DSQG-W | 1030.6 | 347.4 | 5893 | 3801 MB |
| eager sourcewise DSQG-W | 2330.7 | 1749.2 | 1170 | 3507 MB |
| Triton compact read-slots + true backward | 1069.1 | 372.2 | 5499 | 3122 MB |

Interpretation:

- The real trainer path is now trainer-safe and nearly dense-speed: `5499 tok/s` vs dense `5893 tok/s` on this run.
- It is ~4.7x faster than eager sourcewise (`5499` vs `1170 tok/s`).
- It recovers the previous recompute-branch trainer-speed class while keeping the compact read-slot architecture and lower VRAM than dense (`3122 MB` vs `3801 MB`).

## Materialization status

Avoided in the optimized no-routing forward/backward/trainer path:

1. `[B,N,J,D]` candidate states: metadata candidate state tensor remains empty (`metadata_candidate_state_numel=0`).
2. `[B,N,J,H,HD]` candidate projected K/V tensors: source projections remain `[B,N,H,HD]`; candidates are gathered/streamed by metadata inside Triton.
3. `[B,N,J,H]` forward probs when `return_routing=False`: telemetry `dsqg_w_triton_probs_materialized=0.0`; `dsqg_w_probs` absent.
4. `[B,N,J,H]` backward probs: telemetry `dsqg_w_triton_backward_probs_materialized=0.0`; probabilities are scalar-per-candidate inside the Triton row/head program and are not stored.
5. `[B,N,(n_types+1)*D]` read_accum: telemetry `dsqg_w_triton_read_accum_materialized=0.0`.

Allowed bounded intermediates:

1. Compact read slots `[B,N,S,D]`, `S=1+len(read_type_ids)`. In the bench shape, `S=5`, so fp32 bytes are `1*256*5*512*4 = 2,621,440`.
2. LSE `[B,N,H]` fp32, bench bytes `1*256*8*4 = 8,192`.

No bounded reduction buffers are used (`dsqg_w_triton_backward_reduction_buffer_bytes=0.0`). Role/source embeddings and bias/query-type-bias grads use direct atomics.

Still present outside the no-routing hot path:

- `return_routing=True` still materializes `[B,N,J,H]` probabilities for telemetry/routing output.
- The explicit debug fallback `DWARF_DSQG_W_TRITON_COMPACT_READ_BACKWARD=pytorch` still materializes backward scores/probs internally and should not be used for default training.

## Unsupported / guarded cases

- Triton sourcewise remains CUDA-only and opt-in.
- Head dim `>128` is still rejected by the existing sourcewise Triton guard.
- Width-cell and typed-mixer sourcewise paths remain unsupported, as before.
- Score-bias/candidate-score gradients are not returned; candidate scores are metadata inputs, not trainable parameters in the default trainer path.
- `return_routing=True` remains a diagnostic/telemetry path that stores probs.

## HISA untouched note

HISA retrieval/selector code and semantics were not edited. The change is isolated to DSQG-W sourcewise Triton kernels/autograd, DSQG-W tests, benchmark script, and this measurement document.

## Next bottleneck / next step

The next bottleneck is not backward probability materialization anymore; it is direct atomic accumulation pressure in the Triton backward (especially role/source embedding and bias atomics) plus compact read-slot storage bandwidth. If this path needs another speed step, compare:

1. two-stage bounded reduction buffers for role/source/type/source/query-type-bias grads vs current direct atomics;
2. fusing read-mix gradient with compact-slot backward to reduce read-slot traffic;
3. keeping LSE fp32 vs reduced precision after parity testing.
