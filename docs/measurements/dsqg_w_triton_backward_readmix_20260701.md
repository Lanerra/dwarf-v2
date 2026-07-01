# DSQG-W Triton sourcewise recompute backward + no-routing/read_mix fusion — 2026-07-01

Branch: `perf/dsqg-w-triton-backward-readmix`

Base commit: `9568153 perf: add Triton DSQG-W sourcewise prototype`

## Change

Implemented the next opt-in Triton sourcewise step behind the existing explicit switch:

- `DWARF_DSQG_W_TRITON_SOURCEWISE=1`
- dense DSQG-W default remains unchanged
- eager sourcewise remains available as the correctness oracle via `DWARF_DSQG_W_SOURCEWISE=1`
- launcher `--triton-sourcewise` now enables the training-capable Triton sourcewise path

The Triton forward path now:

1. streams compact candidate metadata; no `[B,N,J,D]` candidate state materialization,
2. projects source tensors as rank-3 `[B,N,D]` sources; no `[B,N,J,H,HD]` candidate K/V surfaces,
3. supports no-routing mode and does not allocate/store `[B,N,J,H]` probs when `return_routing=False`,
4. fuses `read_mix` into the Triton read kernel, accumulating directly into `[B,N,D]` `read`; no `[B,N,(n_types+1)*D]` `read_accum` in the optimized no-routing path,
5. uses a custom `torch.autograd.Function` for training: Triton forward, PyTorch sourcewise recompute in backward.

## Backward implementation

`_DSQGWSourcewiseTritonRecompute` is a custom autograd function. Forward calls the Triton sourcewise kernel under autograd's no-grad custom-forward context. Backward recomputes the sourcewise graph with PyTorch ops and calls `torch.autograd.grad` for:

- `x`
- optional `l3_states`
- optional `chunk_rep_states`
- `norm_x`, `norm_c`, `q_proj`, `k_proj`, `v_proj`
- role/source embeddings and type/source/query-type biases
- `read_mix`
- `norm_z`, `fuse`, and gate parameters

This is correctness-first and bounded by sourcewise metadata. It avoids the old hard training guard, but it is not yet an analytic Triton backward. Cost is documented in the benchmarks below.

## RED tests before implementation

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_autograd_matches_eager_sourcewise_backward_on_cuda \
  tests/test_dsqg_w_candidate_provider_optimization.py::test_triton_sourcewise_no_routing_and_fused_read_mix_do_not_materialize_forbidden_outputs \
  tests/test_dsqg_w_trainer_insertion.py::test_dsqg_w_triton_sourcewise_env_is_training_smoke_accepted_on_cuda -q
```

Observed expected RED output:

```text
FFF                                                                      [100%]
NotImplementedError: DWARF_DSQG_W_TRITON_SOURCEWISE=1 is forward-only; run under torch.no_grad() or use eager sourcewise for training
KeyError: 'dsqg_w_triton_probs_materialized'
NotImplementedError: DWARF_DSQG_W_TRITON_SOURCEWISE=1 is forward-only; run under torch.no_grad() or use eager sourcewise for training
3 failed, 1 warning in 4.13s
```

## Focused verification after implementation

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest tests/test_dsqg_w_candidate_provider_optimization.py -q
```

Output:

```text
..........                                                               [100%]
10 passed in 1.95s
```

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest tests/test_dsqg_w_trainer_insertion.py -q
```

Output:

```text
.............                                                            [100%]
13 passed in 3.50s
```

```bash
/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m py_compile \
  kernels/dsqg_w/dsqg_w_mvp.py \
  train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py \
  scripts/run_dsqg_w_full_training.py \
  scripts/bench_dsqg_w_triton_sourcewise.py
```

Output: exit code 0, no stdout/stderr.

## CUDA microbench

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/bench_dsqg_w_triton_sourcewise.py
```

Output summary on RTX 4090, shape `B=1,N=256,D=512,H=8,HD=64,J=16`:

| path | forward ms | peak MB |
|---|---:|---:|
| dense forward no-routing | 1.723 | 132.5 |
| eager sourcewise forward no-routing | 10.183 | 87.3 |
| Triton sourcewise forward no-routing | 1.765 | 81.4 |
| Triton sourcewise forward with routing telemetry | 18.706 | 84.3 |

| path | fwd+bwd ms | peak MB |
|---|---:|---:|
| dense fwd+bwd | 5.300 | 158.3 |
| eager sourcewise fwd+bwd | 17.822 | 150.0 |
| Triton sourcewise fwd+bwd | 17.664 | 168.7 |

Parity/materialization from the same run:

```text
eager_vs_dense_out_max_abs_diff = 2.384185791015625e-07
triton_vs_eager_out_max_abs_diff = 2.384185791015625e-07
triton_no_route_vs_route_out_max_abs_diff = 2.384185791015625e-07
triton_vs_eager_probs_max_abs_diff = 1.7881393432617188e-07
metadata_candidate_state_numel = 0
triton_no_route_probs_materialized = 0.0
triton_read_accum_materialized = 0.0
triton_read_mix_fused = 1.0
```

Interpretation:

- No-routing Triton forward keeps the previous dense-like speed and sourcewise memory profile while avoiding `probs` and `read_accum`.
- Routing telemetry is much slower because the diagnostic path still stores probs and recomputes typed read norms for exact telemetry.
- Fwd+bwd is approximately tied with eager sourcewise because backward is PyTorch recompute, not an analytic Triton backward.
- Triton fwd+bwd currently uses more peak memory than eager in this microbench because custom backward saves parameters/metadata and recomputes through PyTorch autograd.

Full JSON output was printed by the command and is reproducible from the script.

## Tiny real-trainer bench-only comparisons

Commands:

```bash
DWARF_BENCH_ONLY=1 PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/run_dsqg_w_full_training.py \
  --output-dir results/dsqg_w_triton_backward_readmix/dense \
  --run-name dense_bench --max-acc-steps 3 --train-seqs 8 --val-seqs 4 \
  --batch-size 1 --grad-accum 1 --passkey-trials 0 --execute

DWARF_BENCH_ONLY=1 PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/run_dsqg_w_full_training.py \
  --output-dir results/dsqg_w_triton_backward_readmix/eager \
  --run-name eager_sourcewise_bench --max-acc-steps 3 --train-seqs 8 --val-seqs 4 \
  --batch-size 1 --grad-accum 1 --passkey-trials 0 --sourcewise --execute

DWARF_BENCH_ONLY=1 PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/run_dsqg_w_full_training.py \
  --output-dir results/dsqg_w_triton_backward_readmix/triton \
  --run-name triton_sourcewise_bench --max-acc-steps 3 --train-seqs 8 --val-seqs 4 \
  --batch-size 1 --grad-accum 1 --passkey-trials 0 --triton-sourcewise --execute
```

All three exited 0 with empty stderr logs.

Trainer stdout bench lines:

| path | first_step_ms | trailing_avg_ms | steady_tok_s | peak_vram |
|---|---:|---:|---:|---:|
| dense DSQG-W | 1020.2 | 547.3 | 3740 | 3801 MB |
| eager sourcewise DSQG-W | 2404.9 | 1965.1 | 1042 | 3507 MB |
| Triton sourcewise recompute | 8839.8 | 3187.4 | 642 | 2930 MB |

Interpretation:

- The Triton trainer smoke now works and backpropagates, but this recompute-backward version is slower than eager sourcewise on the tiny real trainer.
- Peak VRAM is lower than dense and eager trainer runs in this tiny bench (`2930 MB` vs `3801 MB` dense, `3507 MB` eager), consistent with avoiding dense candidate state/read surfaces in forward.
- First Triton step includes substantial compile overhead (`approx_compile_overhead_ms=5652.3`) on a 3-step window; still, the trailing average is slower than eager because backward is PyTorch recompute and the fused read projection repeats score work across output blocks.

## Fresh 20-step trainer comparison after verification

After the coder run completed, a fresh independent 20-step bench-only comparison was run from the same worktree. Raw local logs are under:

```text
results/dsqg_w_triton_backward_fresh_trainer20_20260701_140640/
```

Command shape for each variant:

```bash
DWARF_BENCH_ONLY=1 PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python \
  scripts/run_dsqg_w_full_training.py \
  --max-acc-steps 20 --train-seqs 32 --val-seqs 4 \
  --batch-size 1 --grad-accum 1 --passkey-trials 0 --execute \
  [dense default | --sourcewise | --triton-sourcewise]
```

All three runs exited 0 with empty stderr logs.

| path | first_step_ms | trailing_avg_ms | steady_tok_s | peak_vram |
|---|---:|---:|---:|---:|
| dense DSQG-W | 1090.9 | 370.1 | 5530 | 3801 MB |
| eager sourcewise DSQG-W | 2475.6 | 1877.6 | 1090 | 3507 MB |
| Triton sourcewise recompute | 1137.3 | 418.2 | 4895 | 2930 MB |

Fresh 20-step interpretation:

- Triton recompute is now a large trainer-speed win over eager sourcewise in this post-compile short window: `4895` vs `1090` tok/s, about `4.49x` faster.
- Triton remains slower than dense default in step time (`418.2` vs `370.1` ms; `0.885x` dense tok/s), but much closer than eager sourcewise.
- Triton peak VRAM is lower than both dense and eager (`2930 MB` vs `3801 MB` dense and `3507 MB` eager).
- This 20-step result is more representative than the earlier 3-step cold smoke; the 3-step window overweighted first-use/compile effects and Python recompute setup.

## Materialization status

Avoided in the optimized no-routing Triton path:

1. `[B,N,J,D]` candidate states (`metadata.cand_states.numel() == 0`).
2. `[B,N,J,H,HD]` candidate K/V projection surfaces (source tensors are projected as `[B,N,H,HD]`, candidates are streamed by metadata in Triton).
3. `[B,N,J,H]` probabilities when `return_routing=False` (`dsqg_w_triton_probs_materialized=0`).
4. `[B,N,(n_types+1)*D]` `read_accum` (`dsqg_w_triton_read_accum_materialized=0`).

Still present or intentionally allowed:

- `[B,N,J,H]` probs are stored only when `return_routing=True` for parity/telemetry.
- Backward recompute uses PyTorch sourcewise ops and materializes intermediate autograd tensors during backward. This is bounded and correctness-verified but not the final performance kernel.

## Caveats / next kernel steps

1. Replace PyTorch recompute backward with analytic Triton backward for score/read/fused read_mix.
2. Improve fused read projection: current kernel repeats score/softmax work for each output block to avoid `read_accum`. A better split would compute reusable softmax/read pieces without restoring the large forbidden surface, or use a more efficient output tiling/atomic strategy.
3. Keep exact routing/typed-read diagnostics out of trainer no-routing mode; telemetry path is now intentionally slower.
4. Benchmark a longer post-compile trainer window after analytic backward exists. This smoke proves training works, not a speed win.
