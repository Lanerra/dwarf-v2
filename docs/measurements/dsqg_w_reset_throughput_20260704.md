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

The base full-candidate W final-site acceptance gate is met.

A follow-up typed-mixer pass removed the worst dense fallback overheads:

- typed mixer and width-cell dense scoring/read paths avoid broadcast multiply/sum materialization in favor of einsum contractions;
- source materialization groups identical source surfaces so `FINAL`/`QUESTION_CACHE` and `L3`/`HISA` are gathered once per surface instead of once per source id;
- `DWARF_DSQG_W_FAST_TELEMETRY=1` disables routing/diagnostic mass reductions during throughput/quality training, because routing telemetry is explicitly not a promotion signal;
- `DWARF_DSQG_W_TYPED_MIXER_PAIR_BIAS=0` disables the tiny typed-pair bias term in the typed mixer. That term is not the target semantic mechanism and costs enough at BS16 to put the gate on the edge.

Latest acceptance evidence:

```text
results/dsqg_w_reset_throughput_20260704_104749_typed_gate_nopair/summary.md
DWARF_DSQG_W_FAST_TELEMETRY=1 DWARF_DSQG_W_TYPED_MIXER_PAIR_BIAS=0
```

| variant | rc | steady tok/s | trailing ms | peak MB | last CE | gate |
|---|---:|---:|---:|---:|---:|---|
| no_w | 0 | 71,344 | 459.1 | 13,122 | 8.2176 | control |
| w_base_final | 0 | 54,282 | 603.4 | 14,227 | 8.3259 | **PASS** |
| w_typed_final | 0 | 51,852 | 631.6 | 19,230 | 8.3755 | **PASS** |

Width cell remains blocked as the suspected killer:

```text
results/dsqg_w_width_gate_20260704_104405/summary.md
DWARF_DSQG_W_FAST_TELEMETRY=1 DWARF_DSQG_W_TYPED_MIXER_PAIR_BIAS=0
```

| variant | rc | steady tok/s | result |
|---|---:|---:|---|
| w_width_final | 2 | 0 | OOM during backward after step 1 |
| w_typed_width_final | 2 | 0 | OOM during forward materialized read |

Conclusion: the reset's first two throughput gates are satisfied for final-site full-candidate W and +typed_mixer. Width-cell alone collapses at BS16 and should stay out of the default path. The next track is the 20K/50K CE + semantic-transfer sanity ladder with aux off by default; do not use aux/routing telemetry to claim W is alive.
