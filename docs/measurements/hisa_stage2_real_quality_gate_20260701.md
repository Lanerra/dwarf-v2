# HISA Stage-2 real quality gate — 2026-07-01

Commit under test: `280b2fc`.

Raw local artifact directory: `results/hisa_stage2_real_quality_gate_20260701_083218/` (ignored by git; contains `run_quality_gate.py`, `run_tmux.sh`, per-variant stdout/stderr, `run_result.json`, `results.json`, and `summary.md`). Transient checkpoints were deleted after parsing by the runner.

## Protocol

Matched from-scratch 1000-step screen on RTX 4090:

- `N=2048`
- `BS=4`, `GA=2`
- `MAX_TRAIN_SEQS=8192`
- `MAX_VAL_SEQS=512`
- `PASSKEY_TRIALS=5`
- `DWARF_LIGER=1`
- `DWARF_DISABLE_BNB=1`
- `DWARF_Q6_G128=0`
- `DWARF_TORCH_COMPILE=0`
- `DWARF_CKPT=none`
- `HISA_TELEMETRY=1`

All variants completed with `returncode=0` and parser health pass.

## Results

| variant | rep_r | DSQG-W | final CE | val PPL | passkey | avg logged tok/s | peak VRAM | elapsed | routing_ent | w_hisa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P_pure_dsqg_v1 | — | 0 | 5.8015 | 312.89 | 0.0% | 64,874 | 3,815 MB | 270s | — | — |
| A_dsr_rowmax | 0 | 0 | 5.8626 | 311.95 | 0.0% | 62,334 | 3,798 MB | 285s | 2.430 | — |
| B_dsr_rep4 | 4 | 0 | 5.8709 | 313.97 | 0.0% | 61,448 | 3,798 MB | 288s | 2.445 | — |
| E_dsr_rep8 | 8 | 0 | 5.8667 | 313.23 | 0.0% | 61,396 | 3,798 MB | 289s | 2.435 | — |
| C_w_rowmax | 0 | 1 | 5.9243 | 311.33 | 0.0% | 13,795 | 8,891 MB | 1303s | 2.386 | 0.872 |
| D_w_rep4 | 4 | 1 | 5.9345 | 314.02 | 0.0% | 13,705 | 8,891 MB | 1313s | 2.403 | 0.837 |
| F_w_rep8 | 8 | 1 | 5.9326 | 313.20 | 0.0% | 13,781 | 8,891 MB | 1311s | 2.394 | 0.851 |

## Ratios and interpretation

- D-only rep4 is `0.986x` rowmax throughput; D-only rep8 is `0.985x` rowmax throughput. At N=2048, representative Stage-2 is not a speed win in the full trainer, but the overhead is small and health is clean.
- DSQG-W is the dominant slowdown: W rowmax is `0.221x` D-rowmax throughput; W rep4 is `0.223x` D-rep4 throughput; W rep8 is `0.224x` D-rep8 throughput.
- DSQG-W raises peak VRAM from `~3.8 GB` to `8.9 GB` (`2.34x` vs D-only rep4).
- rep4 is not the W performance villain: W rep4 is `0.993x` W rowmax throughput; W rep8 is `0.999x` W rowmax throughput.
- Best PPL in this screen is W rowmax (`311.33`), but the margin over D-rowmax (`311.95`) and pure DSQG-D (`312.89`) is small relative to the cost. W rep4 (`314.02`) does not justify the 4.5x throughput hit here.
- Passkey remains `0.0%` across all variants, so this gate is mostly CE/PPL/health/systems evidence, not retrieval-success evidence.

## Decision signal

Keep the committed HISA Stage-2 rep4 default as mechanically healthy, but do not promote DSQG-W as-is for routine iteration. The next useful step is a DSQG-W optimization/profiling sprint, not another long quality gate:

1. profile candidate-provider build, source gathers, full `B×N×J×D` candidate materialization, composer forward/backward, and per-site cost;
2. ablate sites (`final`, `6,final`, `2,6,final`), candidates (`4/8/16`), bottleneck (`32/64/128`), and candidate source sets;
3. reduce or fuse materialization while preserving candidate indices/types/masks and output/grad parity;
4. rerun a smaller speed+parity gate before any further quality escalation.
