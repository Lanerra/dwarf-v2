# DSQG-W GPT-5.5 Pro plan fixes + profiler execution — 2026-07-01

## Implemented

Based on `docs/55pro-response.txt`, this pass implemented the low-risk front of the ranked plan:

1. **PR0 instrumentation/profiler hooks**
   - Added `DWARF_PROFILE_DSQG_W=1` `torch.profiler.record_function` ranges around:
     - `dsqg_w/site=<site>`
     - `dsqg_w/candidate_metadata_build`
     - `dsqg_w/q_projection`
     - `dsqg_w/source_projection_cache`
     - `dsqg_w/read_mix`
     - `dsqg_w/fuse_norm_mlp_gate`
   - Added trainer-level torch profiler support via:
     - `DWARF_PROFILE_DSQG_W=1`
     - `DWARF_PROFILE_DSQG_W_TRACE_DIR=...`
     - `DWARF_PROFILE_DSQG_W_TABLE=...`
     - `DWARF_PROFILE_DSQG_W_WAIT/WARMUP/ACTIVE`
   - Added `scripts/profile_dsqg_w_bottlenecks.py` to run the minimal matrix and save summaries/traces.

2. **Candidate geometry audit**
   - Added opt-in `DWARF_DSQG_W_GEOMETRY_AUDIT=1` telemetry:
     - `dsqg_w_geometry_fixed_slots`
     - `dsqg_w_geometry_fixed_slot_fraction`
     - `dsqg_w_geometry_mode_delta_fraction`
     - `dsqg_w_geometry_slab_candidate_slots`
   - Geometry audit also enables automatically under `DWARF_PROFILE_DSQG_W=1`.
   - This is diagnostic only and intentionally off in normal runs because it can synchronize.

3. **PR1 metadata cache across legal sourcewise sites**
   - Sourcewise trainer path now caches `CandidateProvider.build_metadata(...)` output per forward call using a key over shape/device/dtype and the external/candidate index and score tensors.
   - Cache is cleared at the start of each `forward`/`forward_hidden` call.
   - Cache does not include hidden-value tensors because sourcewise metadata does not materialize candidate states and depends on positions, candidate indices/scores, config, and masks.
   - Telemetry: `dsqg_w_metadata_cache_hit`.

4. **PR2 static source set into source projection path**
   - `CandidateBatch` carries `active_source_ids` from metadata construction.
   - `DSQGWBlock.forward_sourcewise(..., needed_source_ids=...)` can now avoid live tensor `.any()` scans for source-set discovery.
   - Telemetry: `dsqg_w_static_source_count`, `dsqg_w_static_source_set_used`.

## Profiler matrix executed

Script:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python \
  scripts/profile_dsqg_w_bottlenecks.py \
  --output-dir results/dsqg_w_profile_20260701_pro_plan \
  --wait 1 --warmup 1 --active 3
```

Artifacts:

```text
results/dsqg_w_profile_20260701_pro_plan/summary.md
results/dsqg_w_profile_20260701_pro_plan/summary.json
results/dsqg_w_profile_20260701_pro_plan/*/trace/key_averages.txt
results/dsqg_w_profile_20260701_pro_plan/*/trace/*.pt.trace.json
```

Note: these traces are large (~2.4 GiB total) and are profiler artifacts, not source artifacts.

## Profiled matrix results

Profiler timings are heavily perturbed by `torch.profiler` and the geometry audit. Use them for attribution, not throughput claims.

| variant | profiled trailing ms | profiled tok/s | peak MB |
|---|---:|---:|---:|
| no_w | 323.0 | 6338 | 2816 |
| triton_final | 5677.3 | 361 | 2920 |
| triton_6_final | 1502.3 | 1363 | 3019 |
| triton_2_6_final | 1566.9 | 1306 | 3119 |
| triton_no_dsr | 453.6 | 4513 | 3118 |
| triton_split | 5582.6 | 367 | 3119 |

Key profiler rows from the full 3-site DSR-selected path:

```text
_DSQGWSourcewiseTritonCompactReadBackward: 226.624ms CUDA total, 37.84%
_dsqg_w_sourcewise_read_slots_backward_kernel: 226.624ms CUDA total, 37.84%
dsqg_w/candidate_metadata_build: 55.081ms CUDA total, 9.20%
aten::where: 20.299ms CUDA total, 3.39%
aten::addmm: 21.609ms CUDA total, 3.61%
aten::mm: 26.127ms CUDA total, 4.36%
```

Split backward remains worse:

```text
triton_split compact-read backward: 333.641ms CUDA total, 47.20%, 18 launches
monolithic full path: 226.624ms CUDA total, 37.84%, 9 launches
```

## Geometry audit result

The logged final-site telemetry during the profiler matrix consistently reported:

```text
w_fixed=1.000
w_slab=2.000
w_srcs=4.000
```

With `J<=16`, this is not a strong case for a wholesale V20 overlap-slab port. It suggests at most a small fixed/mostly-fixed subset should be considered later, after higher-impact DSR/compact-read/read-mix work.

## Unprofiled sanity runs

Because profiler overhead distorted throughput, I also ran unprofiled 20-step bench-only controls:

| variant | trailing ms | tok/s | peak MB |
|---|---:|---:|---:|
| no_w | 102.9 | 19895 | 2816 |
| triton_2_6_final, DSR-selected | 406.6 | 5035 | 3119 |
| triton_2_6_final, `--no-dsr-candidates` | 205.6 | 9955 | 3118 |

Commands used the same launcher shape as prior 20-step smokes: `--max-acc-steps 20 --train-seqs 80 --val-seqs 4 --batch-size 1 --grad-accum 1 --passkey-trials 0`.

Important interpretation: **DSR-selected candidates are now the clearest throughput split**. Disabling DSR candidates nearly doubles throughput (`5035 -> 9955 tok/s`) while keeping the same three W sites and Triton sourcewise path. This is larger than the site-count signal in the profiled run and bigger than a likely overlap-slab win.

## Verification

```bash
/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m py_compile \
  kernels/dsqg_w/dsqg_w_mvp.py \
  train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py \
  scripts/profile_dsqg_w_bottlenecks.py
```

passed.

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest \
  tests/test_dsqg_w_candidate_provider_optimization.py \
  tests/test_dsqg_w_trainer_insertion.py \
  tests/test_dsqg_w_parity_harness.py \
  tests/test_dsqg_w_full_run_launcher.py -q
```

Result:

```text
36 passed, 2 warnings in 36.41s
```

`git diff --check` passed.

## Current conclusion

GPT-5.5 Pro's ranked diagnosis was mostly right, but the fresh evidence sharpens it:

1. The current local kernel bottleneck is still compact-read backward, especially monolithic source gradient/atomic work.
2. The full-system bottleneck now points hard at the DSR-selected candidate path: metadata/score-biased candidate construction plus DSR-specific compact-read behavior cuts throughput roughly in half versus fallback/no-DSR candidates.
3. Metadata caching and static source-set routing are implemented and working (`w_cache=1` for reused post-DSR sites), but they are not enough by themselves.
4. Wholesale V20 overlap-slab remains unsupported by geometry (`~1 fixed slot, ~2 slab-candidate slots out of J<=16` in this audit).

Recommended next implementation target: specialize/fuse the DSR-selected candidate path before overlap slab:

- reduce candidate metadata sort/where work for DSR-selected fixed `J` shapes;
- investigate score-bias handling cost and whether centering/normalization can be fused or precomputed;
- fuse compact read with `read_mix` to remove `[B,N,S,D]` read-slot traffic;
- then revisit bounded reductions for small parameter atomics.
