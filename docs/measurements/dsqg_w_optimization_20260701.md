# DSQG-W materialization tightening pass — 2026-07-01

Commit base: `cd5cb91`.

Raw local artifact directory: `results/dsqg_w_optimization_20260701_111719/` (ignored by git).

## Change

First parity-preserving DSQG-W systems slice:

1. `CandidateProvider._build_vectorized` now skips source gathers that cannot be used by the configured candidate set. For the quality-gate shape (`DSR_SELECTED_QUESTION_L3_SKIP_NULL`, no chunk reps), this removes the unused summary gather and reuses the final-state gather when L3/HISA falls back to final states.
2. `DSQGWBlock` now derives `read_type_ids` from `DSQGWConfig` and avoids materializing/linear-mixing zero typed reads for candidate types that the config cannot emit. It accumulates the equivalent `read_mix` contribution per possible type instead of concatenating every enum slot into a full `(n_types+1)*D` read vector.

## Microprofile shape

Synthetic CUDA microprofile, RTX 4090:

- `B=4`, `N=2048`, `J=16`, `D=512`, `H=8`, bottleneck `128`
- source mix: final `0.539`, HISA `0.307`, L3 `0.154`, summary `0.0`
- valid candidate count mean `12.956`

| region | before | after | speedup | peak before | peak after |
|---|---:|---:|---:|---:|---:|
| provider only | 3.964 ms | 3.215 ms | 1.23x | 956 MB | 825 MB |
| block forward | 11.983 ms | 8.797 ms | 1.36x | 1888 MB | 1725 MB |
| provider + block forward | 15.706 ms | 11.892 ms | 1.32x | 2156 MB | 1992 MB |
| provider + block fwd/bwd | 75.536 ms | 65.678 ms | 1.15x | 1481 MB | 1368 MB |

Post-patch profiler still shows the main remaining hotspot as `index_select_backward/index_add_` from candidate-state gathers, so the next optimization should target fused/packed source gather + projection rather than more enum-read cleanup.

## Trainer smoke

Bench-only trainer smoke matching the real gate shape (`sites=2,6,final`, no local/long offsets, DSR selected + question + L3 skip, rep4):

- `BS=4`, `GA=2`, `MAX_ACC_STEPS=120`, `MAX_TRAIN_SEQS=8192`, `MAX_VAL_SEQS=128`, `PASSKEY_TRIALS=0`
- steady tok/s: `15,330`
- peak VRAM: `8,368 MB`

Reference from the preceding 1000-step gate, same W rep4 family:

- average logged tok/s: `13,705`
- peak VRAM: `8,891 MB`

This is a meaningful first slice (~12% trainer tok/s improvement, ~0.5 GB lower peak), but it does not close the 60K vs 16K gap. The remaining gap needs a top-down fusion pass around candidate gather/materialization and source-state projection.

## Verification

```text
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py \
  tests/test_dsqg_w_trainer_insertion.py \
  tests/test_dsqg_w_parity_harness.py \
  tests/test_dsqg_w_full_run_launcher.py -q

24 passed in 5.04s
```

```text
/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m py_compile \
  kernels/dsqg_w/dsqg_w_mvp.py \
  tests/test_dsqg_w_candidate_provider_optimization.py
```

passed with exit `0`.
