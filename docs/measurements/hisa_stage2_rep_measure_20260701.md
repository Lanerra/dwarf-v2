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

## Readout

- At N=2048 trainer scale, `rep4` and `rep8` are effectively throughput-neutral in the full trainer: `rep4` is 0.998x rowmax and `rep8` is 1.001x rowmax in this 12-step bench. The full model is not Stage-2-selector dominated at this context length.
- The representative selector materially narrows the Stage-2 candidate surface: rowmax selected fraction `0.949`; rep4 `~0.89`; rep8 `~0.91`.
- Coverage is decent but not identical: rep4 recovers 93.4% of rowmax tokens / 93.9% of rowmax score mass and hits rowmax top-1 96.8%; rep8 improves to 96.0% / 96.4% / 98.3%.
- Tiny scratch PPL/passkey screens show no gross regression, but they are too short and from random init; passkey stays 0% for all variants, so this does not validate retrieval quality.
- DSQG-W candidate telemetry tracks the expected surface change: `w_hisa` final is slightly lower for rep4/rep8 than rowmax, consistent with narrower DSR-selected evidence, while PPL differences are noise at this scale.
