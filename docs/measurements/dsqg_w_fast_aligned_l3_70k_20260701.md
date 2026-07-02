# DSQG-W fast aligned-L3 path reaches 70K tok/s — 2026-07-01

## Goal

Dennis set the target at roughly DSQG-D throughput: **~70K tok/s** for DSQG-W while preserving low memory consumption.

This pass used Context7 Triton guidance and MCP CodeGraph/source navigation to separate kernel-level overhead from algorithmic memory traffic. The key result is that the full score/read DSQG-W candidate path is not close enough; the target is reachable only by avoiding per-token candidate gathers/backward and using a much cheaper semantic W perturbation.

## Fresh BS16 baseline

All measurements use CUDA device mapping verified live (`CUDA_VISIBLE_DEVICES=0` exposes the RTX 4090 in PyTorch), N=2048 trainer shape, BS=16, GA=1, ~40 steps unless noted.

| variant | throughput | peak VRAM | notes |
|---|---:|---:|---|
| no-W DSR backbone | ~73.3K tok/s | ~13.1GB | target ceiling for this trainer shape |
| full DSQG-W, 3 sites, DSR selected, Triton sourcewise | ~28.6K tok/s | ~16.4GB | `w_j=11`; trainable W backward dominates |
| full DSQG-W, 3 sites, no DSR candidates | ~35.8K tok/s | ~16.4GB | removing DSR-selected candidates helps but not enough |
| detached full W, 3 sites | ~47.9K tok/s | ~13.2GB | proves W backward is a large memory/speed cost |
| fast evidence mean with 10 gathered evidence slots | ~48.4K tok/s | ~13.2GB | removes score/read kernels but still does per-token gather fanout |
| fast evidence mean with 1 gathered evidence slot | ~49.9K tok/s | ~13.2GB | width reduction barely helps; gather path overhead remains |
| fast no-op W (`J=0`) | ~71.6K tok/s | ~13.2GB | target reachable if W does not gather candidates |
| **fast aligned-L3 W, 3 sites (`J=1`)** | **71.4K tok/s** | **13.17GB** | target reached with nonzero semantic perturbation |

## Implemented mode

New explicit env knobs:

```text
DWARF_DSQG_W_DETACH_RECOMPOSER=1
DWARF_DSQG_W_FAST_EVIDENCE_MEAN=1
```

When `DWARF_DSQG_W_FAST_EVIDENCE_MEAN=1`:

- If candidate indices are provided, it computes a simple evidence mean over those candidates.
- If no candidate indices are provided but L3/DSR state exists, it uses **aligned L3 state** as the semantic read vector (`J=1`).
- If `DWARF_DSQG_W_DETACH_RECOMPOSER=1`, the W perturbation is forward-only:
  - output still changes: `x + detached_delta`
  - W internals do not run backward
  - trunk gradient remains identity through the recomposer boundary

This is deliberately an experimental fast semantic-W path, not a claim that it is equivalent to trainable full DSQG-W.

## Launcher support

`scripts/run_dsqg_w_full_training.py` now exposes:

```text
--detach-recomposer
--fast-evidence-mean
--k-question N
--k-hisa-evidence N
--k-l3-skip N
```

Reproducer for the target-reaching run:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/run_dsqg_w_full_training.py \
  --gpu 0 \
  --run-name dsqg_w_launcher_fastmean_aligned_l3_2_6_final_bs16_20260701 \
  --output-dir results/dsqg_w_launcher_fastmean_aligned_l3_2_6_final_bs16_20260701 \
  --max-acc-steps 40 --train-seqs 640 --val-seqs 4 \
  --batch-size 16 --grad-accum 1 --log-interval 40 --passkey-trials 0 \
  --sites 2,6,final --sourcewise --triton-sourcewise \
  --detach-recomposer --fast-evidence-mean \
  --k-question 0 --k-hisa-evidence 0 --k-l3-skip 0 \
  --execute
```

Observed launcher output:

```text
[ep1 step 40/40] ... 71486 tok/s ... w_dx=0.711 w_cache=0.000 w_j=1.000 w_det=1.000 w_fast=1.000
peak_vram=13172MB  elapsed=29s
```

## Interpretation

The 70K target is reached, but only by changing the W training semantics:

- Full trainable DSQG-W remains much slower (`~28.6K tok/s`).
- Detaching full W gets only to `~48K`, so W forward/gather traffic is also too expensive.
- Candidate gathers are catastrophic: even `J=1` gathered evidence stays around `~50K`.
- Aligned L3 avoids candidate gather fanout and recovers DSQG-D-like throughput with a nonzero semantic perturbation (`w_dx≈0.71`).

Next scientific question: whether aligned-L3 fast W preserves/improves CE, passkey, and external evals versus no-W and full-W controls. Performance target is met; quality is not yet established.

## Verification

Focused tests:

```text
42 passed in 5.87s
```

Command:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py \
  tests/test_dsqg_w_trainer_insertion.py \
  tests/test_dsqg_w_parity_harness.py \
  tests/test_dsqg_w_full_run_launcher.py -q
```

Additional checks:

```text
py_compile passed
git diff --check passed
```
