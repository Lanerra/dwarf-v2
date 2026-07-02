# DSQG-W DSR-selected compact-J + read_mix prototype — 2026-07-01

## Context

After the profiler/pro-plan pass, the dominant remaining gap was the DSR-selected candidate path:

- no-W: ~19.9K tok/s
- no-DSR candidates: ~10.0K tok/s
- DSR-selected Triton W: ~5-6K tok/s

This pass targeted DSR-selected candidate specialization and compact-read/read_mix traffic.

## Changes

1. Added a DSR-selected metadata fast path behind:

```text
DWARF_DSQG_W_SPECIALIZED_METADATA=1   # default enabled
```

Eligibility is deliberately narrow:

- no local offsets
- no long offsets
- no chunk candidates
- typed HISA reps disabled
- question + HISA evidence + HISA scores + L3 skip tensors present

The fast path preserves generic priority/score/dedup semantics, but it avoids the generic all-candidate same-key machinery for the known DSR-selected shape.

2. Compact candidate-slot count for the DSR fast path.

The previous metadata path padded DSR-selected metadata to `max_candidates=16`. For the current DSR selected shape, only 11 raw slots are emitted:

```text
question4 + hisa4 + l3skip2 + null1 = J=11
```

The specialized path now returns compact `J=11` instead of padded `J=16`, reducing Triton loop work in the sourcewise forward/backward kernels.

3. Fixed generic padded-order semantics.

When `max_candidates > raw_candidate_count`, generic vectorized candidate construction padded `order_idx` with zeros and could preserve candidate-0 metadata in padded columns. The mask is now explicitly padded false before gather, so padded columns become invalid/null consistently.

4. Added compact read-slot read_mix helper and env-gated batched prototype:

```text
DWARF_DSQG_W_BATCHED_READ_MIX=1
```

This is not full Triton read_mix fusion. It is a narrower traffic/prototype path that batches the per-slot `F.linear` operations. It matches gradients exactly in focused tests but is only a small/neutral runtime win.

5. Added telemetry/logging:

```text
dsqg_w_candidate_specialized_metadata
dsqg_w_candidate_slot_count
w_j=<candidate slot count>
dsqg_w_batched_read_mix
```

## Focused verification

```bash
/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m py_compile \
  kernels/dsqg_w/dsqg_w_mvp.py \
  train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py \
  tests/test_dsqg_w_candidate_provider_optimization.py

PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py \
  tests/test_dsqg_w_trainer_insertion.py \
  tests/test_dsqg_w_parity_harness.py \
  tests/test_dsqg_w_full_run_launcher.py -q
```

Result:

```text
38 passed, 2 warnings in 36.64s
```

Warnings are existing Triton/Python 3.15 deprecation warnings from `triton.compiler.code_generator`.

## Bench evidence

All runs used:

```text
--max-acc-steps 40 --train-seqs 160 --val-seqs 4
--batch-size 1 --grad-accum 1 --log-interval 40 --passkey-trials 0
--sites 2,6,final --sourcewise --triton-sourcewise
```

Warmed comparison before compact-J telemetry relaunch:

| variant | trailing_avg_ms | tok/s | peak MB |
|---|---:|---:|---:|
| specialized metadata on, padded J=16 | 338.0 | 6057 | 3119 |
| specialized metadata off, generic J=16 | 332.2 | 6162 | 3119 |
| specialized on + batched read_mix, padded J=16 | 334.7 | 6116 | 3125 |

After compact-J specialization:

| variant | trailing_avg_ms | tok/s | peak MB |
|---|---:|---:|---:|
| specialized compact J=11 | 334.6 | 6118 | 3119 |
| specialized compact J=11 + batched read_mix | 331.8 | 6170 | 3124 |
| specialized off, generic J=16 | 335.9 | 6094 | 3119 |

A later 20-step run confirmed log telemetry:

```text
w_j=11.000
```

but that run had a cold compile after the kernel-shape change and is not used as speed evidence.

## Interpretation

This is a small but real hygiene/performance step, not the breakthrough:

- Compacting DSR selected metadata from J=16 to J=11 removes avoidable kernel loop work.
- The measured gain is modest (~0.4% over generic in one warmed comparison; ~1.2% with batched read_mix).
- The read_mix Python-side batched prototype is neutral/slightly positive but does not attack the dominant compact-read backward cost.

The next serious optimization needs to target the Triton compact-read backward kernel itself: fewer score recompute passes, DSR-source-specialized load paths, or a fused autograd path that avoids materializing/read-slot gradient traffic without adding a worse recompute.
