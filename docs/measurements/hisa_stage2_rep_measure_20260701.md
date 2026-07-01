# HISA Stage-2 query-representative measurement — 2026-07-01

Commit under test: `fe343b9 feat: default HISA Stage-2 to query reps`.

Raw local artifact directory: `results/hisa_stage2_rep_measure_20260701_080940/` (ignored by git; contains logs, runner scripts, parser, and `results.json`). Large transient checkpoints were deleted after parsing.

Device: RTX 4090 via `CUDA_VISIBLE_DEVICES=0` / PyTorch-visible `NVIDIA GeForce RTX 4090`.
Dataset: `datasets/dwarf_base_v1_olmo1tok_2048_2b.pt`; tokenizer: OLMo-1 GPT-NeoX/Dolma tokenizer.
Common knobs: `DWARF_TORCH_COMPILE=0`, `DWARF_DISABLE_BNB=1`, `DWARF_Q6_G128=0`, `DWARF_CKPT=none`, `HISA_TELEMETRY=1`.

## Selector coverage audit

Forward-only audit on two validation sequences at N=2048, C=32, top_k=4, M=64. Rowmax is the oracle token set; representative variants are compared against rowmax-selected token IDs and rowmax score mass.

| variant | mean token recall vs rowmax | score-mass recall | rowmax top1 hit | exact set rate | stage2 selected frac | hidden L2 delta vs rowmax |
|---|---:|---:|---:|---:|---:|---:|
| rep4 | 93.4% | 93.9% | 96.8% | 75.6% | 0.887 | 0.0313 |
| rep8 | 96.0% | 96.4% | 98.3% | 77.6% | 0.911 | 0.0241 |

Rowmax selected fraction: `0.949`.

## Trainer throughput bench

Scratch trainer, DSQG-W off, 12 optimizer steps, `BS=4`, `GA=2`, `MAX_TRAIN_SEQS=128`, bench-only.

| rep_r | steady tok/s | relative to rowmax | trailing avg ms | peak VRAM | final logged CE | final stage2 frac | routing entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48,620 | 1.000x | 336.8 | 3,798 MB | 10.2567 | 0.949 | 2.527 |
| 4 | 48,519 | 0.998x | 337.5 | 3,798 MB | 10.2568 | 0.891 | 2.527 |
| 8 | 48,690 | 1.001x | 336.3 | 3,798 MB | 10.2569 | 0.915 | 2.527 |

## Tiny scratch quality screen: DSQG-W off

One tiny epoch: 12 optimizer steps, `BS=1`, `GA=1`, `MAX_TRAIN_SEQS=64`, `MAX_VAL_SEQS=32`, passkey trials=1. This is a regression/health screen, not a quality claim.

| rep_r | val PPL | passkey mean | peak VRAM | elapsed | final train CE | final stage2 frac |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 33992.43 | 0.0% | 1,804 MB | 16s | 10.4572 | 0.949 |
| 4 | 33997.20 | 0.0% | 1,804 MB | 14s | 10.4573 | 0.890 |
| 8 | 33992.56 | 0.0% | 1,804 MB | 14s | 10.4574 | 0.911 |

## Tiny scratch quality screen: DSQG-W on

DSQG-W sites `2,6,final`, DSR-selected/question/L3-skip candidates, 8 optimizer steps, `MAX_TRAIN_SEQS=48`, `MAX_VAL_SEQS=16`. This checks whether the DSR-selected candidate surface changes grossly under rep selectors.

| rep_r | val PPL | passkey mean | peak VRAM | elapsed | final train CE | stage2 frac | final w_hisa | final w_score_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 37177.12 | 0.0% | 2,713 MB | 27s | 10.7549 | 0.949 | 0.546 | 0.279 |
| 4 | 37186.82 | 0.0% | 2,713 MB | 27s | 10.7538 | 0.893 | 0.535 | 0.263 |
| 8 | 37184.36 | 0.0% | 2,713 MB | 29s | 10.7543 | 0.917 | 0.538 | 0.267 |

## 200-step D-only rep_r sweep

Raw local artifact directory: `results/hisa_stage2_rep_r_sweep_20260701_163442/` (ignored by git; contains `summary.md`, `sweep_results.json`, per-rep logs/configs, and transient checkpoints).

Real trainer, DSQG-W off to isolate HISA Stage-2, `MAX_ACC_STEPS=200`, `MAX_TRAIN_SEQS=256`, `MAX_VAL_SEQS=128`, `BS=1`, `GA=1`, `LOG_INTERVAL=10`, passkey trials=1, RTX 4090 via `CUDA_VISIBLE_DEVICES=0`. All variants passed mechanical health: expected GPU observed, final step `200/200`, finite CE/PPL, matching `rep_r`, no fatal log patterns.

| rep_r | avg tok/s | relative to rowmax | final CE | ΔCE | val PPL | ΔPPL | passkey | stage2 frac | routing ent | peak VRAM | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 19,430 | 1.000x | 7.3927 | 0.0000 | 2014.50 | 0.00 | 0.0% | 0.949 | 2.530 | 1,804 MB | 35s |
| 1 | 19,053 | 0.981x | 7.3934 | +0.0007 | 2015.56 | +1.06 | 0.0% | 0.827 | 2.530 | 1,804 MB | 36s |
| 2 | 18,977 | 0.977x | 7.3932 | +0.0005 | 2014.79 | +0.29 | 0.0% | 0.863 | 2.530 | 1,804 MB | 36s |
| 4 | 18,866 | 0.971x | 7.3929 | +0.0002 | 2014.87 | +0.37 | 0.0% | 0.897 | 2.530 | 1,804 MB | 36s |
| 8 | 18,996 | 0.978x | 7.3928 | +0.0001 | 2014.58 | +0.08 | 0.0% | 0.917 | 2.529 | 1,804 MB | 36s |

## Readout

- At N=2048 trainer scale, query representatives do not improve end-to-end trainer throughput; the 200-step D-only sweep shows `rep1..rep8` at ~0.97-0.98x rowmax. The full trainer is not Stage-2-selector dominated at this context length, and the rep path has enough PyTorch/gather overhead to erase the smaller dot-product count.
- The representative selector materially narrows the Stage-2 candidate surface: rowmax selected fraction `0.949`; rep1 `0.827`; rep2 `0.863`; rep4 `0.897`; rep8 `0.917`.
- Short-run quality is effectively unchanged in this scratch screen: 200-step final CE deltas are <= +0.0007 and val PPL deltas are <= +1.06 versus rowmax. This is a health/throughput screen, not a retrieval-quality claim; passkey stays 0% for all variants from this random-init horizon.
- Coverage is decent but not identical: the forward audit showed rep4 recovers 93.4% of rowmax tokens / 93.9% of rowmax score mass and hits rowmax top-1 96.8%; rep8 improves to 96.0% / 96.4% / 98.3%.
- DSQG-W candidate telemetry tracks the expected surface change in the tiny W-on screen: `w_hisa` final is slightly lower for rep4/rep8 than rowmax, consistent with narrower DSR-selected evidence, while PPL differences are noise at that scale.
