# DSQG-W Width-Native Optimization Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn DSQG-W from a diagnostic bounded candidate reader into a width-native semantic transfer mechanism whose candidate dimension performs useful computation, not just selection.

**Architecture:** Current DSQG-W is a safe heterogeneous attention adapter: it gathers causal candidates, scores each candidate independently against the query, makes typed reads, and applies a gated residual. The width-native version should add bounded candidate-to-candidate computation inside each token's candidate set so semantic cues, evidence tokens, relation/type roles, and source layers can interact before the final read. The first concrete mechanism is a gated candidate lateral interaction cell over the `J <= 16` candidate axis.

**Tech Stack:** PyTorch reference path in `kernels/dsqg_w/dsqg_w_mvp.py`, trainer integration in `train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py`, pytest, existing lexical-gap/checkpoint eval scripts.

---

## Diagnosis

The current DSQG-W block is intentionally conservative:

- Candidate construction is bounded, causal, heterogeneous, and now vectorized.
- Candidate score is basically `query · (candidate_key + role_key + source_key)` plus type/source bias.
- Read path computes global read plus per-type typed reads.
- Fusion is `[x, read, x*read, read-x] -> bottleneck -> gated residual`.
- Gate starts near identity for trainer safety.

That was correct for proving integration, parity, checkpointing, and throughput. It is not enough for width.

DSQG-D has deep mechanism: offset topology, scale embeddings, positional bias, MOVT phase rotation, interference/EMA injection, DSR/HISA retrieval, q6 direct-consume paths, and percolation dynamics. DSQG-W currently has only candidate typing and a shallow MLP fusion. It can choose among candidates, but it does not let candidates compose with each other.

Width-native DSQG-W should make the candidate set itself a small semantic computation field.

---

## Proposed mechanism: Candidate Lateral Interaction Cell

Add an opt-in `DSQGWWidthCell` inside `DSQGWBlock` before query scoring.

For each token position, candidates form a tiny causal set:

```text
C[t] = {local, long, question, HISA evidence, L3 skip, null}, J <= 16
```

The width cell performs candidate-to-candidate transfer over this set:

```text
candidate_i' = candidate_i + lateral_gate * sum_j softmax(score(i, j)) * value_j
```

where `score(i, j)` should include:

1. content compatibility: projected candidate_i · projected candidate_j,
2. relation features: `(candidate_i - candidate_j)` and `candidate_i * candidate_j` via low-rank projections,
3. type-pair bias: e.g. QUESTION -> HISA_EVIDENCE can learn a distinct prior from LOCAL -> LOCAL,
4. source-pair bias: FINAL/L3/SUMMARY source interactions can specialize,
5. optional anti-self bias so the cell cannot collapse to identity-only lateral reads.

This is bounded `O(B*T*J^2*D)` with `J=16`, so it is cheap relative to the trunk. It is also conceptually width-native: the candidate axis is no longer a passive menu; it becomes a semantic transfer workspace.

Initial safety:

- gated residual with `width_gate_init=-5.0`,
- final lateral projection initialized near zero,
- env flag disabled by default until tests pass,
- telemetry for lateral entropy, self-mass, question→evidence mass, evidence→question mass, delta norm.

---

## Success criteria

A successful first version does not need to win full pretraining immediately. It must prove the mechanism is alive and useful on targeted probes.

Required:

1. Unit behavior:
   - shape/no-NaN,
   - all-invalid rejection remains,
   - near-identity init preserved,
   - disabled width cell exactly matches current DSQG-W output,
   - enabled width cell changes output when lateral gate is opened,
   - pair-type/source bias affects lateral routing in a controlled toy case.

2. Throughput:
   - D512/T2048/J16 provider + block benchmark remains within an acceptable factor of current optimized DSQG-W.
   - Target: no worse than 1.5x current DSQG-W block time for first PyTorch reference.

3. Targeted lexical-gap:
   - On frozen/tiny lexical-gap microtrain, width cell should reduce answer CE faster than current DSQG-W at equal steps.
   - On full 2k checkpoints plus short auxiliary fine-tune, DSQG-W+width should improve answer CE/MRR/top100 faster than DSQG-D control or current DSQG-W.

4. Trainer stability:
   - 500-step real trainer smoke: no NaNs, no OOM, checkpoint/eval path clean.

---

## Task 1: Add tests for width-cell identity and activation

**Objective:** Lock the safety contract before modifying DSQG-W production code.

**Files:**
- Modify: `tests/test_dsqg_w_mvp.py`
- Modify: `kernels/dsqg_w/dsqg_w_mvp.py`

**Step 1: Write failing tests**

Add tests that construct a small `DSQGWBlock` with `use_width_cell=False` and `use_width_cell=True`.

Required assertions:

- disabled width cell output equals legacy block output when weights are copied,
- enabled width cell with closed gate stays near identity,
- manually opening width gate and setting pair bias changes routing/output,
- no NaNs with mixed candidate types/sources.

**Step 2: Run RED**

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest tests/test_dsqg_w_mvp.py -q
```

Expected: failure because `DSQGWWidthCell` / config flags do not exist yet.

---

## Task 2: Extend `DSQGWConfig` with width-cell knobs

**Objective:** Add opt-in config without changing current behavior.

**Files:**
- Modify: `kernels/dsqg_w/dsqg_w_mvp.py`
- Modify: trainer env config in `train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py`

Add fields:

```python
use_width_cell: bool = False
width_bottleneck: int = 64
width_gate_init: float = -5.0
width_self_bias_init: float = 0.0
```

Trainer env:

```text
DWARF_DSQG_W_WIDTH_CELL=0/1
DWARF_DSQG_W_WIDTH_BOTTLENECK=64
DWARF_DSQG_W_WIDTH_GATE_INIT=-5.0
```

Current launcher should keep the width cell disabled unless explicitly requested.

---

## Task 3: Implement `DSQGWWidthCell`

**Objective:** Add bounded candidate-to-candidate semantic transfer.

**Files:**
- Modify: `kernels/dsqg_w/dsqg_w_mvp.py`

Implementation sketch:

```python
class DSQGWWidthCell(nn.Module):
    def __init__(self, d, n_heads, n_types, n_sources, bottleneck, gate_init=-5.0):
        ...

    def forward(self, cand_states, cand_types, cand_sources, cand_mask):
        # [B,T,J,D] -> [B,T,J,D], telemetry
```

Core operations:

- normalize candidate states,
- project pair queries/keys/values,
- add type-pair and source-pair bias tables,
- mask invalid candidates,
- softmax over source candidate `j`,
- lateral read per candidate,
- fuse `[c_i, lateral_i, c_i*lateral_i, lateral_i-c_i]`,
- gated residual into candidate state.

Keep the first implementation simple and readable. J is tiny. Do not optimize before proving signal.

---

## Task 4: Insert width cell before DSQG-W query read

**Objective:** Let candidate states interact before the existing query-to-candidate scorer.

**Files:**
- Modify: `kernels/dsqg_w/dsqg_w_mvp.py`

In `DSQGWBlock.forward()`:

```python
if self.width_cell is not None:
    cand_states, width_telemetry = self.width_cell(cand_states, cand_types, cand_sources, cand_mask)
```

Merge telemetry into the existing DSQG-W telemetry.

Preserve current behavior exactly when disabled.

---

## Task 5: Add trainer and launcher plumbing

**Objective:** Make width cell experiment-runnable without editing code.

**Files:**
- Modify: `train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py`
- Modify: `scripts/run_dsqg_w_full_training.py`
- Modify tests for launcher config if needed.

Add CLI/env flags so this can run:

```bash
scripts/run_dsqg_w_full_training.py \
  --execute \
  --width-cell \
  --output-dir runs/dsqg_w_width_cell_4090_step500
```

The default remains current DSQG-W.

---

## Task 6: Benchmark and targeted eval

**Objective:** Prove whether the width cell gives actual semantic-transfer signal.

Run focused tests:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest tests/test_dsqg_w_mvp.py tests/test_dsqg_w_parity_harness.py -q
```

Run full suite:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python -m pytest tests -q
```

Run microbench:

```bash
/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python - <<'PY'
# Same D512/T2048/J16 provider+block timing probe used for vectorization work.
PY
```

Run trainer smoke:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/run_dsqg_w_full_training.py \
  --execute \
  --output-dir runs/dsqg_w_width_cell_4090_step500 \
  --run-name width_cell_step500_2_6_final_4090 \
  --max-acc-steps 500 \
  --train-seqs 8192 \
  --val-seqs 1024 \
  --batch-size 1 \
  --grad-accum 1 \
  --passkey-trials 3 \
  --width-cell
```

Then run lexical-gap eval against matched checkpoints:

```bash
PYTHONPATH=. /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python scripts/evaluate_lexical_gap_checkpoints.py \
  --dsqg-d-checkpoint runs/dsqg_d_baseline_4090_step2000_matched/checkpoints/d512_l10_dsqg_d_step2000_matched_4090_ep1.pt \
  --dsqg-w-checkpoint runs/dsqg_w_width_cell_4090_step500/checkpoints/<checkpoint>.pt \
  --output runs/lexical_gap_width_cell_eval.json \
  --val-size 144 \
  --device cuda:0
```

If checkpoint step counts differ, report that explicitly; do not pretend it is a matched comparison.

---

## Design notes / non-goals

- Do not copy DSQG-D offset machinery into DSQG-W. Width should not become another depth/retrieval kernel.
- Do not add dense `T*T` attention.
- Do not make the candidate provider more complex until the candidate interaction cell is tested.
- Do not chase passkey first. The first target is lexical-gap answer-rank movement.
- Do not make the first width cell Triton. PyTorch reference first, then optimize only if signal appears.

---

## Expected interpretation

If width cell improves lexical-gap answer CE/MRR/top100 without breaking PPL too badly, DSQG-W has its first real width-native mechanism.

If it does not move targeted metrics, the next mechanism should be more explicitly relational:

- learned relation slots by fact kind,
- question/evidence cross-type contrastive loss,
- answer-head-side auxiliary projection,
- width-cell supervision from gold evidence/cue pairs.

But the immediate missing piece is candidate-to-candidate semantic transfer. Current DSQG-W cannot do that. This plan adds it.
