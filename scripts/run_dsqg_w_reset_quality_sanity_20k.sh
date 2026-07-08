#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/dlewis3/Desktop/AI/DWARF-v2}"
PY="${PY:-/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python}"
GPU="${GPU:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/dsqg_w_reset_quality_${STAMP}_20k}"
TRAIN_SEQS="${TRAIN_SEQS:-20000}"
VAL_SEQS="${VAL_SEQS:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_ACC_STEPS="${MAX_ACC_STEPS:-1250}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"

mkdir -p "$RUN_ROOT/pretrain"
cd "$ROOT"

COMMON=(
  --gpu "$GPU"
  --max-acc-steps "$MAX_ACC_STEPS"
  --train-seqs "$TRAIN_SEQS"
  --val-seqs "$VAL_SEQS"
  --batch-size "$BATCH_SIZE"
  --grad-accum 1
  --log-interval "$LOG_INTERVAL"
  --passkey-trials 0
)

run_variant() {
  local variant="$1"; shift
  local out_dir="$RUN_ROOT/pretrain/$variant"
  mkdir -p "$out_dir"
  echo "=== $(date -Is) variant=$variant out=$out_dir ==="
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH=. \
  PYTHONUNBUFFERED=1 \
  DWARF_DSQG_W_FAST_TELEMETRY=1 \
  DWARF_DSQG_W_TYPED_MIXER_PAIR_BIAS=0 \
    "$PY" scripts/run_dsqg_w_full_training.py \
      --run-name "$variant" \
      --output-dir "$out_dir" \
      "${COMMON[@]}" \
      "$@" \
      --execute
  echo "=== $(date -Is) done variant=$variant ==="
}

run_variant no_w --disable-dsqg-w
run_variant w_typed_aux0 \
  --sites final \
  --sourcewise \
  --triton-sourcewise \
  --typed-mixer \
  --width-aux-weight 0.0

# Semantic transfer is the promotion gate, not routing telemetry. Run it after
# both checkpoints exist. Keep this on the same visible GPU by default; override
# GPU/CUDA_VISIBLE_DEVICES at launch if a separate eval GPU is desired.
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. PYTHONUNBUFFERED=1 \
  "$PY" scripts/run_dsqg_w_run_dir_semantic_eval.py \
    --run-root "$RUN_ROOT" \
    --variant-ids no_w w_typed_aux0 \
    --semantic-suite builtin_v3_deconfounded

echo "RUN_ROOT=$RUN_ROOT"
