# DSQG-W Triton sourcewise score/read prototype — 2026-07-01

Base commit: `1b4231c perf: prototype DSQG-W sourcewise accumulation`.

Branch: `perf/dsqg-w-triton-sourcewise`.

## Change

Implemented a separate explicit opt-in forward-only Triton sourcewise DSQG-W score/read path:

- switch: `DWARF_DSQG_W_TRITON_SOURCEWISE=1`
- launcher arg: `scripts/run_dsqg_w_full_training.py --triton-sourcewise` (also forces `DWARF_DSQG_W_SOURCEWISE=1` in the launcher config)
- dense DSQG-W remains default
- eager sourcewise remains the training-capable sourcewise path

The Triton path is selected inside `DSQGWBlock.forward_sourcewise(...)` only when the env switch is set. It projects source tensors once with existing PyTorch linears, then launches a Triton kernel over `(B, N, H)` rows that:

1. streams compact candidate metadata (`cand_token_indices`, `cand_types`, `cand_sources`, `cand_mask`, optional centered `cand_scores`),
2. gathers source-projected K/V rows by candidate source/token,
3. computes scores over bounded `J`, including role/source/type/query-type biases,
4. computes softmax over `J`,
5. accumulates all-read and per-type reads into the existing `read_mix` input layout.

It does not construct `[B,N,J,D]` candidate states and does not construct `[B,N,J,H,HD]` candidate K/V projection tensors.

## Supported prototype shape/features

Tested:

- common DSR-selected-like DSQG-W gate shape: `B=2`, `N=512`, `D=512`, `H=8`, `HD=64`, `J=16`
- CUDA / RTX 4090 / Triton 3.7.1
- candidate sources: FINAL / HISA(L3-backed) / L3 / NULL-compatible; SUMMARY pointer is wired but not exercised in the main microbench
- candidate types include NULL, QUESTION, HISA_EVIDENCE, L3_SKIP, and LOCAL in the small padded-HD guard
- optional candidate score bias
- query type bias
- non-power-of-two head dim guard: `D=20`, `H=4`, `HD=5` with padded Triton lanes

Unsupported / guarded:

- `width_cell` and `typed_mixer` remain unsupported in sourcewise and still raise via the existing sourcewise guard.
- Backward is not implemented. With `DWARF_DSQG_W_TRITON_SOURCEWISE=1`, `forward_sourcewise` raises `NotImplementedError` when autograd is enabled. This prevents silent use as a trainer speed path.
- This is a forward microbench/prototype, not a training win.

## RED tests before implementation

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_matches_eager_sourcewise_on_cuda \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_avoids_forbidden_candidate_surfaces_with_padded_hd -q
```

Observed failure before production code:

```text
FF                                                                       [100%]
KeyError: 'dsqg_w_triton_sourcewise'
AssertionError: Triton sourcewise must not use eager per-candidate gather rows
2 failed in 1.57s
```

## Focused correctness / materialization tests after implementation

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py -q
```

Output:

```text
.........                                                                [100%]
9 passed in 1.84s
```

The CUDA tests check:

- Triton output parity vs eager sourcewise at `atol=2e-5`, `rtol=2e-5`
- routing probability parity vs eager sourcewise at `atol=2e-5`, `rtol=2e-5`
- read/source telemetry parity for exposed read/source mass keys
- no candidate states in metadata (`metadata.cand_states.numel() == 0`)
- Triton path does not call the eager `_gather_source_rows` per-candidate gather
- K/V projections see rank-3 source tensors only, not `[B,N,J,D]`
- padded lane stores/loads work for `HD=5`
- backward/training is explicitly rejected as forward-only

## Microbench

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python \
  scripts/bench_dsqg_w_triton_sourcewise.py
```

Output:

```json
{
  "device": "NVIDIA GeForce RTX 4090",
  "features": {
    "k_hisa_evidence": 4,
    "k_l3_skip": 2,
    "k_question": 4,
    "local_offsets": [],
    "long_offsets": [],
    "query_type_bias": true,
    "sourcewise": true,
    "triton_forward_only": true
  },
  "materialization": {
    "dense_candidate_state_bytes": 33554432,
    "forbidden_projected_kv_surface_bytes_each": 33554432,
    "metadata_candidate_state_numel": 0
  },
  "parity": {
    "dense_read_norm": 1.4916660785675049,
    "eager_read_norm": 1.4916660785675049,
    "eager_vs_dense_out_max_abs_diff": 2.384185791015625e-07,
    "triton_read_norm": 1.4916660785675049,
    "triton_vs_dense_out_max_abs_diff": 2.384185791015625e-07,
    "triton_vs_eager_out_max_abs_diff": 2.384185791015625e-07,
    "triton_vs_eager_probs_max_abs_diff": 2.682209014892578e-07
  },
  "shape": {
    "B": 2,
    "D": 512,
    "H": 8,
    "HD": 64,
    "J": 16,
    "N": 512
  },
  "timings": [
    {
      "iters": 50,
      "ms": 2.411029815673828,
      "name": "dense_forward",
      "peak_mb": 343.74365234375,
      "warmup": 10
    },
    {
      "iters": 50,
      "ms": 10.052396240234375,
      "name": "eager_sourcewise_forward",
      "peak_mb": 162.85498046875,
      "warmup": 10
    },
    {
      "iters": 50,
      "ms": 1.764638671875,
      "name": "triton_sourcewise_forward",
      "peak_mb": 163.72265625,
      "warmup": 10
    }
  ]
}
```

Forward-only interpretation:

| path | forward ms | relative to dense | relative to eager sourcewise | peak MB |
|---|---:|---:|---:|---:|
| dense | 2.411 | 1.00x | 4.17x faster than eager | 343.7 |
| eager sourcewise | 10.052 | 0.24x dense | 1.00x | 162.9 |
| Triton sourcewise | 1.765 | 1.37x faster than dense | 5.70x faster than eager | 163.7 |

This is a real forward speed win for the tested microbench shape while preserving the sourcewise memory profile. It is not a training speed win because backward is not implemented.

## Materialization status

Avoided in the tested Triton path:

1. `[B,N,J,D]` candidate-state surface: metadata uses `cand_states.numel() == 0`.
2. `[B,N,J,H,HD]` candidate projected K/V surfaces: source tensors are projected as rank-3 `[B,N,D]`, reshaped to source-projected `[B,N,H,HD]`, then the Triton kernel streams candidate rows directly from compact metadata.

The Triton kernel does allocate/read-write:

- `probs`: `[B,N,J,H]` for routing telemetry/parity
- `read_accum`: `[B,N,(n_types+1)*D]`, matching the existing read-mix input shape and not candidate-projected K/V

## Caveats / next kernel steps

1. Add a custom autograd Function or recompute backward. Until then, the path is intentionally forward-only and guarded against trainer use.
2. Consider a no-routing mode that does not store `[B,N,J,H]` probabilities when telemetry is not requested.
3. Move `read_mix` into Triton or fuse the read accumulation with the downstream projection to avoid materializing `[B,N,(n_types+1)*D]`.
4. Exercise SUMMARY / chunk-rep source candidates if they become active in a real DSQG-W config.
5. Benchmark larger `N` and other `J` values after backward exists; forward-only numbers should not drive training decisions.
