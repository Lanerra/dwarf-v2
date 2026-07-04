# DSQG-W systems/objective reset throughput sprint — 2026-07-04

## Decision recorded

- Aligned-L3 is closed as the main product path; it remains a control only.
- The current width aux objective is invalid/cheating as a promotion signal. Aux is forced to `0.0` for this throughput gate.
- No 100K+ architecture/quality claims until full-candidate W reaches the usable tok/s floor.

## Acceptance ladder

Target trainer shape: RTX 4090 via `CUDA_VISIBLE_DEVICES=0`, N=2048, BS=16, GA=1, 40 bench-only steps, final-site DSQG-W first.

Gate order:

1. no-W DSR backbone control
2. full candidate W, final site only, sourcewise Triton, no typed mixer, no width cell, aux=0
3. + typed mixer
4. + width cell / typed+width only after the earlier gates are sane

Promotion floor for step 2: `>=50K tok/s`.

## Implementation changes

1. Added `scripts/run_dsqg_w_reset_throughput_matrix.py` as a reusable bench-only acceptance runner for:
   - `no_w`
   - `w_base_final`
   - `w_typed_final`
   - `w_width_final`
   - `w_typed_width_final`

2. Optimized `_pack_hisa_selected_tokens_for_dsqg_w` in `kernels/hierarchical_sparse_attn_v15_hisa.py`.

   Previous behavior rebuilt a `[B, seq_len]` scatter table and ran `topk` once per token row in each query chunk. At N=2048 that produced thousands of `topk`/scatter kernels per trainer step.

   New behavior dedupes candidates once per query chunk, applies a causal prefix mask across all rows in that chunk, and runs chunk-batched top-k over `[B, chunk_rows, prefix_len]`.

3. Fixed telemetry-only OOM patterns in DSQG-W typed/width cells by replacing `delta.masked_select(cand_mask[..., None])` with mask-weighted per-candidate norm reductions. This avoids multi-GB boolean-expanded telemetry materialization.

## Evidence

Focused tests after changes:

```text
76 passed in 6.25s
```

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_hisa_stage2_query_representatives.py \
  tests/test_dsqg_w_mvp.py \
  tests/test_dsqg_w_candidate_provider_optimization.py \
  tests/test_dsqg_w_trainer_insertion.py \
  tests/test_dsqg_w_parity_harness.py \
  tests/test_dsqg_w_full_run_launcher.py -q
```

BS16 acceptance run:

```text
results/dsqg_w_reset_throughput_20260704_093040_bs16_packfix/summary.md
```

| variant | rc | steady tok/s | trailing ms | peak MB | last CE | gate |
|---|---:|---:|---:|---:|---:|---|
| no_w | 0 | 71,034 | 461.1 | 13,122 | 8.2174 | control |
| w_base_final | 0 | 52,581 | 622.9 | 14,227 | 8.3257 | **PASS** |
| w_typed_final | 2 | 0 | 0.0 | 0 | 10.9510 at step 1 | blocked/OOM |

Before the HISA pack optimization, the same BS16 final-site base W gate was:

```text
results/dsqg_w_reset_throughput_20260704_091116_bs16/summary.md
w_base_final: 39,262 tok/s, 834.2 ms, 14,227 MB
```

So the pack optimization moved final-site full-candidate W from below floor to above floor:

```text
39,262 -> 52,581 tok/s  (~1.34x)
834.2 -> 622.9 ms
```

## Current state / next gate

The base full-candidate W final-site acceptance gate is now met.

Typed mixer is not yet promotable at BS16:

- Without checkpointing it OOMs after step 1 around the dense fallback path (`k_eff = k + role + source`) because typed mixer currently forces sourcewise W to materialize candidate states and then use the dense DSQG-W forward.
- With `DWARF_CKPT=all`, typed mixer fits but only reaches about `38.7K tok/s`:

```text
results/dsqg_w_typed_ckpt_probe_20260704_093317/trainer.stdout.log
[ep1 step 40/40] ... 38,721 tok/s ... w_mat=1.000
[BENCH] peak_vram=11064MB
```

Therefore the next systems target is the typed-mixer sourcewise path: avoid falling back to dense `[B,T,J,D] -> k/v -> k_eff` materialization, or otherwise fuse/recompute it so typed-mixer BS16 can be measured against the same `>=50K tok/s` floor.

Width cell remains behind typed/base gates. The telemetry-only OOM was fixed, but the semantic width cell still materializes candidate states and JxJ tensors; do not promote it until the typed/sourcewise materialization problem is solved.
