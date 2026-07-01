# DSQG-W source-wise accumulation prototype — 2026-07-01

Base commit: `8543acd perf: trim DSQG-W materialization overhead`.

Branch: `perf/dsqg-w-sourcewise-accumulation`.

## Change

Implemented an explicit opt-in eager PyTorch source-wise DSQG-W score/read path:

- switch: `DWARF_DSQG_W_SOURCEWISE=1`
- launcher arg: `scripts/run_dsqg_w_full_training.py --sourcewise`
- dense DSQG-W remains the default (`DWARF_DSQG_W_SOURCEWISE=0`).

The sourcewise path adds:

1. `CandidateProvider.build_metadata(...)`, which emits candidate indices/types/sources/masks/scores but returns an empty `cand_states` tensor instead of constructing `[B,N,J,D]` candidate states.
2. `DSQGWBlock.forward_sourcewise(...)`, which projects full source tensors once (`x`, L3/HISA source, null source), then loops over candidate slots/sources to form `[B,N,J,H]` scores and accumulate reads without constructing candidate-state `[B,N,J,D]` or candidate projected-K/V `[B,N,J,H,HD]` tensors.

Unsupported combinations: `width_cell` and `typed_mixer` still require candidate states; `forward_sourcewise` raises `NotImplementedError` if either is enabled.

## Correctness / materialization tests

Focused RED first:

```text
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_sourcewise_accumulation_matches_dense_forward_and_backward \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_sourcewise_path_does_not_build_candidate_state_or_projected_kv_surfaces -q

FF                                                                       [100%]
AttributeError: 'CandidateProvider' object has no attribute 'build_metadata'
```

After implementation:

```text
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_sourcewise_accumulation_matches_dense_forward_and_backward \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_sourcewise_path_does_not_build_candidate_state_or_projected_kv_surfaces -q

..                                                                       [100%]
2 passed in 0.73s
```

The focused parity test checks deterministic dense-vs-sourcewise:

- output parity: `atol=1e-5`, `rtol=1e-5`
- routing `dsqg_w_probs` parity: `atol=1e-6`, `rtol=1e-6`
- scalar read/source telemetry parity for exposed keys
- backward gradient parity for `x` and non-final `l3_states`

The materialization guard checks:

- `build_metadata` does not call `_gather_states`
- `metadata.cand_states.numel() == 0`
- sourcewise K/V projection inputs are rank-3 `[B,N,D]`, not rank-4 `[B,N,J,D]`

## Required verification

```text
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py \
  tests/test_dsqg_w_trainer_insertion.py \
  tests/test_dsqg_w_parity_harness.py \
  tests/test_dsqg_w_full_run_launcher.py -q

............................                                             [100%]
28 passed in 4.83s
```

```text
/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m py_compile \
  kernels/dsqg_w/dsqg_w_mvp.py \
  train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py \
  scripts/run_dsqg_w_full_training.py

exit 0
```

## Microbench

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python \
  results/dsqg_w_sourcewise_microbench_20260701.py
```

Shape: `B=2`, `N=512`, `D=512`, `H=8`, `J=16`, CUDA, DSR-selected-like candidate mix (`QUESTION/HISA/L3_SKIP/NULL`, no local/long offsets).

Parity/materialization metrics from the run:

```json
{
  "out_max_abs_diff": 2.384185791015625e-07,
  "probs_max_abs_diff": 4.172325134277344e-07,
  "metadata_candidate_state_numel": 0,
  "dense_candidate_state_bytes": 33554432,
  "forbidden_projected_kv_surface_bytes_each": 33554432
}
```

Timing/memory:

| region | dense | sourcewise | sourcewise/dense | dense peak | sourcewise peak |
|---|---:|---:|---:|---:|---:|
| provider+block forward | 3.819 ms | 11.176 ms | 0.342x | 352.8 MB | 128.7 MB |
| provider+block fwd/bwd | 10.122 ms | 21.369 ms | 0.474x | 371.9 MB | 216.1 MB |

Interpretation: eager sourcewise materially lowers peak memory, but is much slower than the dense vectorized path because it pays Python-loop and many small gather/projection-combine overheads.

## Real trainer bench-only smoke

Both runs used:

- RTX CUDA device 0
- `DWARF_DSQG_W=1`, `DWARF_DSQG_W_SITES=2,6,final`
- `DWARF_DSQG_W_LOCAL_OFFSETS=none`, `DWARF_DSQG_W_LONG_OFFSETS=none`
- DSR selected candidates + question + L3 skip
- `DWARF_HISA_STAGE2_REP_R=4`
- `DWARF_BENCH_ONLY=1`
- `DWARF_MAX_ACC_STEPS=4`, `DWARF_MAX_TRAIN_SEQS=32`, `DWARF_MAX_VAL_SEQS=16`
- `DWARF_BS=1`, `DWARF_GA=1`, `DWARF_PASSKEY_TRIALS=0`

Dense command variant used `DWARF_DSQG_W_SOURCEWISE=0`; sourcewise command variant used `DWARF_DSQG_W_SOURCEWISE=1`.

Trainer bench outputs:

```text
# dense
[BENCH] first_step_ms=1068.3 trailing_avg_ms=499.4 steady_tok_s=4099 approx_compile_overhead_ms=569.0
[BENCH] peak_vram=3988MB compile=False mode=eager window=4 steps=4

# sourcewise
[BENCH] first_step_ms=2365.8 trailing_avg_ms=1698.2 steady_tok_s=1205 approx_compile_overhead_ms=667.6
[BENCH] peak_vram=3697MB compile=False mode=eager window=4 steps=4
```

Trainer delta:

| path | steady tok/s | relative | peak VRAM |
|---|---:|---:|---:|
| dense default | 4,099 | 1.000x | 3,988 MB |
| sourcewise eager | 1,205 | 0.294x | 3,697 MB |

Interpretation: sourcewise eager saves about `291 MB` in this tiny trainer smoke but loses about `71%` throughput. This is not the real speed target.

## Conclusion

The opt-in path proves the source-wise accumulation interface and memory shape, and it avoids both forbidden materializations in the tested path:

1. no full `[B,N,J,D]` candidate-state surface (`cand_states.numel() == 0`),
2. no full `[B,N,J,H,HD]` candidate projected-K/V surface (K/V projection sees only `[B,N,D]` source tensors; per-slot gathers are transient `[B,N,H,HD]`).

It is not faster than dense. Dense remains default and should remain the production path.

## Next bottleneck / next step

Eager PyTorch sourcewise is dominated by Python loops over `J` and sources plus many small gathers. To hit the real DSQG-W speed target, this needs a fused CUDA/Triton kernel that, per `(B,N,H)` row, streams candidate metadata, gathers from source-projected tensors, computes online softmax over `J`, and accumulates all/typed reads in one kernel. The current prototype is useful as the correctness oracle for that kernel boundary, not as a trainer-speed win.
