"""
DWARF D=512 L=10 Triadic J=96 — HISA/DSQG hybrid v2 L3, Bible-Muon OLMo-1 tokenizer base_v1 variant.

q6_g128 smoke clone of train_d512_l10_muon_olmo1_base_v1.py.

The original trainer path is preserved by default. Set DWARF_Q6_G128=1 to route
selected DSQG blocks through a q6_g128 Triton direct-gather K/V read path for
real trainer-loop smoke testing.

Scratch target: train the Bible-Muon D512 architecture from random init on the
OLMo-1 / GPT-NeoX-Dolma-tokenized base_v1 Stage-A cache. This is the
external-tokenizer baseline for the 35/20/15/15/5/5/5 base_v1 data regime.

Architecture: D=512, H=8 (hd=64), L=10, FFN=1536, tied lm_head
  Triadic partitioning: 96 offsets split into 3 pure groups of 32
  HISA at L3 (replacing the first post-triad relay slot)
  HISA block: HierarchicalSparseAttentionV15HISA(C=32, top_k=4, HISA_m=32 default)
  All DSQG blocks use the V20-compatible R_PLANES=4 Triton kernel with scale_embed + sequential MOVT.

Layout:
  L00: DSQGBlock(GROUP_A)
  L01: DSQGBlock(GROUP_B)
  L02: DSQGBlock(GROUP_C) + preIF
  L03: DSRBlock / HISA
  L04: DSQGBlock(GROUP_A)
  L05: DSQGBlock(GROUP_B)
  L06: DSQGBlock(GROUP_C)
  L07: DSQGBlock(GROUP_A)
  L08: DSQGBlock(GROUP_B)
  L09: DSQGBlock(GROUP_C)
"""

import contextlib, hashlib, json, math, os, random, subprocess, sys, time, types
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Mapping
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
import torch.nn.functional as F

try:
    if os.getenv('DWARF_DISABLE_BNB', '0') == '1':
        raise ImportError('bitsandbytes disabled by DWARF_DISABLE_BNB=1')
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except Exception as _bnb_exc:
    bnb = None
    _BNB_AVAILABLE = False
    print(f"WARNING: bitsandbytes unavailable/disabled ({_bnb_exc}); using standard AdamW")

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Triton 3.5+ compatibility for module-scope constants referenced by JIT kernels.
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError as _interop_exc:
    # Import-smoke tests may load this trainer more than once in one process.
    print(f"WARNING: set_num_interop_threads skipped ({_interop_exc})")

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False

USE_LIGER_CE = _LIGER_AVAILABLE and os.getenv("DWARF_LIGER", "1") != "0"

try:
    from liger_kernel.transformers import LigerLayerNorm
    _LayerNorm = LigerLayerNorm
    _LIGER_LN = True
except ImportError:
    _LayerNorm = torch.nn.LayerNorm
    _LIGER_LN = False

# Selective Activation Checkpointing (SAC) — requires PyTorch 2.4+
try:
    from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
    _SAC_AVAILABLE = True
    _sac_intensive_ops = frozenset([
        torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten.addmm,
    ])
    def _sac_policy_fn(ctx, op, *args, **kwargs):
        return (CheckpointPolicy.MUST_SAVE if op in _sac_intensive_ops
                else CheckpointPolicy.PREFER_RECOMPUTE)
except ImportError:
    _SAC_AVAILABLE = False

import pathlib as _pl
_script_dir = str(_pl.Path(__file__).resolve().parent)
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_canonical_kernel_dir = _pl.Path(_project_root) / 'kernels'
_kernel_dir_override = os.getenv('DWARF_DSQG_KERNEL_DIR', '').strip()
if _kernel_dir_override:
    _kernel_dir_path = _pl.Path(_kernel_dir_override).expanduser().resolve()
    _kernel_module_path = _kernel_dir_path / 'dsqg_attention_v20_bf16_se.py'
    if not _kernel_module_path.is_file():
        raise FileNotFoundError(
            f'DWARF_DSQG_KERNEL_DIR must contain dsqg_attention_v20_bf16_se.py, got {_kernel_dir_path}'
        )
else:
    _kernel_dir_path = _canonical_kernel_dir
_kernel_dir = str(_kernel_dir_path)
_tools_dir = os.path.join(_project_root, 'tools')
for _d in [_script_dir, str(_canonical_kernel_dir), _kernel_dir, _tools_dir, _project_root]:
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)
# Hermes also has a top-level `tools` package. Prefer DWARF/tools when it
# exists, but allow a local passkey_eval.py when running this uploaded bundle.
if os.path.isdir(_tools_dir):
    _tools_pkg = types.ModuleType('tools')
    _tools_pkg.__path__ = [_tools_dir]
    _tools_pkg.__file__ = os.path.join(_tools_dir, '__init__.py')
    sys.modules['tools'] = _tools_pkg

try:
    from tools.passkey_eval import PasskeyConfig, format_passkey_results, passkey_prefix_consistency_audit
except Exception:
    from passkey_eval import PasskeyConfig, format_passkey_results, passkey_prefix_consistency_audit


from dsqg_attention_v20_bf16_se import (
    DSQGAttentionV19,
    dsqg_attention_v18_grouped,
    npci_rotate, R_PLANES, _next_pow2, _rms_normalize_last,
    NPCI_THETA_MAX, NPCI_THETA_INIT,
    ALL_OFFSETS,
)
DSQG_KERNEL_MODULE_PATH = str(_pl.Path(sys.modules['dsqg_attention_v20_bf16_se'].__file__).resolve())
assert R_PLANES == 4, f"Expected R_PLANES=4, got {R_PLANES}"
_DSQG_TYPES = (DSQGAttentionV19,)
print(
    '  Kernel: V20-compatible DSQG (R=4 sequential Givens, grouped sparse, SE gates); '
    f'module={DSQG_KERNEL_MODULE_PATH}; optional q6_g128 smoke path'
)
from causal_ema_scan import causal_ema_scan as _causal_ema_scan

try:
    import triton
    import triton.language as tl
    _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE = True
except Exception as _q6_stage_d_triton_exc:
    triton = None
    tl = None
    _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE = False
    print(f"WARNING: Stage-D Triton scatter unavailable ({_q6_stage_d_triton_exc}); using torch scatter fallback")

try:
    from q6_g128 import layout as _q6_layout_mod
    _Q6_LAYOUT_AVAILABLE = True
except Exception as _q6_exc:
    _q6_layout_mod = None
    _Q6_LAYOUT_AVAILABLE = False
    print(f"WARNING: q6_g128 layout helpers unavailable ({_q6_exc}); DWARF_Q6_G128=1 will fail")

try:
    from q6_g128 import decode as _q6_triton_mod
    _Q6_TRITON_AVAILABLE = True
except Exception as _q6_triton_exc:
    _q6_triton_mod = None
    _Q6_TRITON_AVAILABLE = False
    print(f"WARNING: q6_g128 Triton direct-gather helpers unavailable ({_q6_triton_exc}); DWARF_Q6_G128=1 will fail")

try:
    from q6_g128 import fused_consume as _q6_fused_mod
    _Q6_FUSED_CONSUME_AVAILABLE = True
except Exception as _q6_fused_exc:
    _q6_fused_mod = None
    _Q6_FUSED_CONSUME_AVAILABLE = False
    print(f"WARNING: q6_g128 fused direct-consume helpers unavailable ({_q6_fused_exc}); DWARF_Q6_G128_FUSED_CONSUME=1 will fail")

# =============================================================================
# DSR IMPORT
# =============================================================================

HISA_IMPL = os.getenv('DWARF_HISA_IMPL', 'v15').strip().lower()
if HISA_IMPL == 'v16':
    try:
        from hierarchical_sparse_attn_v16_hisa_causal import HierarchicalSparseAttentionV16HISACausal as HISA_IMPL_CLS
    except Exception:
        from kernels.hierarchical_sparse_attn_v16_hisa_causal import HierarchicalSparseAttentionV16HISACausal as HISA_IMPL_CLS
    _pack_hisa_selected_tokens_for_dsqg_w = None
    HISA_IMPL_LABEL = "V16 strict-causal local+completed-chunk Triton forward/backward"
elif HISA_IMPL == 'v15':
    try:
        from hierarchical_sparse_attn_v15_hisa import HierarchicalSparseAttentionV15HISA as HISA_IMPL_CLS, _pack_hisa_selected_tokens_for_dsqg_w
    except Exception:
        from hierarchical_sparse_attn_v15_hisa_triton import HierarchicalSparseAttentionV15HISA as HISA_IMPL_CLS
        _pack_hisa_selected_tokens_for_dsqg_w = None
    HISA_IMPL_LABEL = 'V15 legacy chunk-shared HISA'
else:
    raise ValueError(f"DWARF_HISA_IMPL must be 'v15' or 'v16', got {HISA_IMPL!r}")

try:
    from kernels.dsqg_w.dsqg_w_mvp import DSQGWBlock, DSQGWConfig, CandidateProvider
except Exception:
    from dsqg_w.dsqg_w_mvp import DSQGWBlock, DSQGWConfig, CandidateProvider

# =============================================================================
# OFFSET GROUPS
# =============================================================================

def _canonicalize_all_offsets(offsets):
    vals = [int(d) for d in offsets]
    if len(vals) != 96:
        raise ValueError(f'Expected 96 offsets, got {len(vals)}')
    if len(set(vals)) != len(vals):
        raise ValueError('Duplicate offsets are not supported')
    middle = [d for d in vals if not (d <= 28 or d >= 48)]
    if middle:
        raise ValueError(f'Unsupported offsets in gap 29..47: {middle}')
    ordered = sorted(vals)
    if ordered != vals:
        print('  [offsets] ALL_OFFSETS was not sorted; using sorted canonical order')
    return ordered


def _canonicalize_offset_group(offsets):
    vals = [int(d) for d in offsets]
    small = sorted(d for d in vals if d <= 28)
    large = sorted(d for d in vals if d >= 48)
    if len(small) + len(large) != len(vals):
        raise ValueError(f'Offset group contains unsupported middle offsets: {vals}')
    return small + large


def _count_small_large(offsets):
    j_small = sum(1 for d in offsets if d <= 28)
    j_large = sum(1 for d in offsets if d >= 48)
    assert j_small + j_large == len(offsets), (
        f"J_SMALL({j_small}) + J_LARGE({j_large}) != J({len(offsets)})")
    if offsets[:j_small] != sorted(offsets[:j_small]) or any(d > 28 for d in offsets[:j_small]):
        raise ValueError(f'Offset group is not small-first sorted: {offsets}')
    if offsets[j_small:] != sorted(offsets[j_small:]) or any(d < 48 for d in offsets[j_small:]):
        raise ValueError(f'Offset group is not large-second sorted: {offsets}')
    return j_small, j_large

_ALL_96_ORDERED = _canonicalize_all_offsets(ALL_OFFSETS)
GROUP_A = _canonicalize_offset_group(_ALL_96_ORDERED[0:32])
GROUP_B = _canonicalize_offset_group(_ALL_96_ORDERED[32:64])
GROUP_C = _canonicalize_offset_group(_ALL_96_ORDERED[64:96])

J_SMALL_A, J_LARGE_A = _count_small_large(GROUP_A)  # 17, 15
J_SMALL_B, J_LARGE_B = _count_small_large(GROUP_B)  # 0, 32
J_SMALL_C, J_LARGE_C = _count_small_large(GROUP_C)  # 0, 32

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

EMBEDDING_DIM    = 512
NUM_HEADS        = 8
FFN_DIM          = int(os.environ.get('DWARF_FFN_DIM', '1536'))
NUM_LAYERS       = 10
DSR_LAYER        = 3
VOCAB_SIZE       = int(os.environ.get('DWARF_VOCAB_SIZE', '50280'))

NUM_CHUNKS       = 32
TOP_K_CHUNKS     = int(os.environ.get('DWARF_HISA_TOP_K', '4'))
HISA_TOP_M_TOKENS = int(os.environ.get('DWARF_HISA_TOP_M', '64'))

SCALE_EMBED_INIT_VAL = 0.15
# Conservative audited default; override with DWARF_SCALE_EMBED_LR_MULT.
SCALE_EMBED_LR_MULT  = float(os.environ.get('DWARF_SCALE_EMBED_LR_MULT', '8.0'))
EMA_INIT  = 0.020833
EMA_FLOOR = 0.00001
PRE_HISA_EMA_ENABLED = os.getenv('DWARF_PRE_HISA_EMA', '1') == '1'
LR        = float(os.environ.get('DWARF_LR', '3e-4'))
WEIGHT_DECAY = float(os.environ.get('DWARF_WEIGHT_DECAY', '0.1'))
PHASE_LR_MULT = float(os.environ.get('DWARF_PHASE_LR_MULT', '10.0'))
NPCI_THETA_LR_MULT = float(os.environ.get('DWARF_NPCI_THETA_LR_MULT', '8.0'))
GRAD_CLIP_NORM = float(os.environ.get('DWARF_GRAD_CLIP_NORM', '1.0'))
SKIP_NONFINITE_STEP = os.getenv('DWARF_SKIP_NONFINITE_STEP', '1') == '1'
SE_MAX_ABORT = float(os.environ.get('DWARF_SE_MAX_ABORT', '0.0'))
LR_WARMUP_STEPS = int(os.environ.get('DWARF_LR_WARMUP_STEPS', '0'))
MIN_LR_RATIO = float(os.environ.get('DWARF_MIN_LR_RATIO', '0.1'))
SCALE_EMBED_CONSTANT_LR = os.getenv('DWARF_SCALE_EMBED_CONSTANT_LR', '0') == '1'
DROPOUT   = 0.1


@dataclass(frozen=True)
class LRScheduleConfig:
    kind: str
    total_steps: int
    warmup_steps: int
    stable_steps: int
    decay_steps: int
    step_offset: int


def build_lr_schedule_config(*, run_steps: int, environ: Mapping[str, str] | None = None) -> LRScheduleConfig:
    """Resolve a fail-closed local or continuation-aware LR schedule contract."""
    if run_steps <= 0:
        raise ValueError(f'run_steps must be positive, got {run_steps}')
    env = os.environ if environ is None else environ
    kind = env.get('DWARF_LR_SCHEDULE', 'cosine').strip().lower()
    if kind not in {'cosine', 'wsd'}:
        raise ValueError(f"DWARF_LR_SCHEDULE must be 'cosine' or 'wsd', got {kind!r}")
    total_steps = int(env.get('DWARF_SCHEDULE_TOTAL_STEPS', str(run_steps)))
    step_offset = int(env.get('DWARF_SCHEDULE_STEP_OFFSET', '0'))
    if total_steps <= 0 or step_offset < 0:
        raise ValueError(f'invalid schedule total_steps={total_steps} step_offset={step_offset}')
    if step_offset + run_steps > total_steps:
        raise ValueError(
            f'run_steps={run_steps} with step_offset={step_offset} exceeds schedule horizon total_steps={total_steps}'
        )

    if kind == 'cosine':
        warmup_steps = min(max(int(env.get('DWARF_LR_WARMUP_STEPS', '0')), 0), max(total_steps - 1, 0))
        return LRScheduleConfig(
            kind=kind,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            stable_steps=0,
            decay_steps=max(total_steps - warmup_steps, 1),
            step_offset=step_offset,
        )

    warmup_steps = int(env.get('DWARF_WSD_WARMUP_STEPS', str(math.ceil(total_steps * 0.05))))
    decay_steps = int(env.get('DWARF_WSD_DECAY_STEPS', str(math.ceil(total_steps * 0.15))))
    stable_steps = int(env.get('DWARF_WSD_STABLE_STEPS', str(total_steps - warmup_steps - decay_steps)))
    if min(warmup_steps, stable_steps, decay_steps) < 0 or warmup_steps + stable_steps + decay_steps != total_steps:
        raise ValueError(
            'WSD phases must be non-negative and sum to DWARF_SCHEDULE_TOTAL_STEPS: '
            f'warmup={warmup_steps} stable={stable_steps} decay={decay_steps} total={total_steps}'
        )
    if decay_steps == 0:
        raise ValueError('WSD requires a positive decay phase')
    return LRScheduleConfig(
        kind=kind,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        step_offset=step_offset,
    )


def lr_schedule_multiplier(*, step: int, config: LRScheduleConfig, min_lr_ratio: float) -> float:
    """Return the multiplier at a local scheduler step under a global schedule contract."""
    logical_step = min(max(int(step) + config.step_offset, 0), max(config.total_steps - 1, 0))
    if config.warmup_steps > 0 and logical_step < config.warmup_steps:
        return max((logical_step + 1) / config.warmup_steps, 1e-8)
    if config.kind == 'wsd' and logical_step < config.warmup_steps + config.stable_steps:
        return 1.0
    decay_start = config.warmup_steps + config.stable_steps
    decay_progress = min(max((logical_step - decay_start + 1) / max(config.decay_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def validate_schedule_resume(*, config: LRScheduleConfig, resume_path: str, skip_scheduler_state: bool) -> None:
    """Prevent a continuation offset and a restored local scheduler from double-counting progress."""
    if resume_path and config.step_offset > 0 and not skip_scheduler_state:
        raise ValueError(
            'continued schedules with DWARF_SCHEDULE_STEP_OFFSET > 0 require DWARF_SKIP_SCHED=1 '
            'so the explicit global offset is the sole source of scheduler progress'
        )


def select_train_tranche(
    *,
    train_data: torch.Tensor,
    train_loss_mask: torch.Tensor,
    max_train_seqs: int,
    offset_text: str | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]:
    """Select an explicit contiguous continuation tranche or preserve legacy randomized capping."""
    if offset_text is not None and offset_text.strip() != '':
        offset = int(offset_text)
        if offset < 0:
            raise ValueError(f'DWARF_TRAIN_SEQ_OFFSET must be non-negative, got {offset}')
        count = int(max_train_seqs)
        end = offset + count
        if end > len(train_data):
            raise ValueError(
                f'DWARF_TRAIN_SEQ_OFFSET={offset} with DWARF_MAX_TRAIN_SEQS={count} '
                f'exceeds dataset rows={len(train_data)}'
            )
        return train_data[offset:end], train_loss_mask[offset:end], {
            'mode': 'contiguous', 'offset': offset, 'count': count, 'end': end,
        }
    if len(train_data) > max_train_seqs:
        subset_idx = torch.randperm(len(train_data))[:max_train_seqs]
        return train_data[subset_idx], train_loss_mask[subset_idx], {
            'mode': 'random_cap', 'offset': 0, 'count': int(max_train_seqs), 'end': int(max_train_seqs),
        }
    return train_data, train_loss_mask, {
        'mode': 'full', 'offset': 0, 'count': len(train_data), 'end': len(train_data),
    }


BATCH_SIZE     = int(os.environ.get('DWARF_BS', '20'))
GRAD_ACCUM     = int(os.environ.get('DWARF_GA', '20'))
MAX_TRAIN_SEQS = int(os.environ.get('DWARF_MAX_TRAIN_SEQS', '976562'))
MAX_SEQ_LEN    = int(os.environ.get('DWARF_SEQ_LEN', '2048'))
MAX_VAL_SEQS   = int(os.environ.get('DWARF_MAX_VAL_SEQS', '8192'))
CE_CHUNK       = int(os.environ.get('DWARF_CE_ROWS', '2048'))  # rows per streamed final-projection CE chunk
PIN_DATASET    = os.getenv('DWARF_PIN_DATASET', '0') == '1'
SEED           = int(os.environ.get('DWARF_SEED', '42'))
REQUIRE_PREFIX_CLEAN = os.getenv('DWARF_REQUIRE_PREFIX_CLEAN', '0') == '1'
SCREEN_EPOCHS  = int(os.environ.get('DWARF_EPOCHS', '2'))

TRAIN_LOG_INTERVAL = int(os.environ.get('DWARF_LOG_INTERVAL', '100'))
MAX_ACC_STEPS = int(os.environ.get('DWARF_MAX_ACC_STEPS', '0'))
BENCH_ONLY = os.getenv('DWARF_BENCH_ONLY', '0') == '1'
PROFILE_DSQG_W = os.getenv('DWARF_PROFILE_DSQG_W', '0') == '1'
PROFILE_DSQG_W_TRACE_DIR = os.getenv('DWARF_PROFILE_DSQG_W_TRACE_DIR', 'traces/dsqg_w')
PROFILE_DSQG_W_TABLE = os.getenv('DWARF_PROFILE_DSQG_W_TABLE', '')
PROFILE_DSQG_W_WAIT = int(os.environ.get('DWARF_PROFILE_DSQG_W_WAIT', '1'))
PROFILE_DSQG_W_WARMUP = int(os.environ.get('DWARF_PROFILE_DSQG_W_WARMUP', '1'))
PROFILE_DSQG_W_ACTIVE = int(os.environ.get('DWARF_PROFILE_DSQG_W_ACTIVE', '3'))
_compile_env = os.getenv('DWARF_TORCH_COMPILE')
if _compile_env is None:
    # Custom Triton/autograd + checkpointing paths are benchmark axes, not a safe default.
    TORCH_COMPILE_ENABLED = os.getenv('DWARF_COMPILE', '0') != '0'
else:
    TORCH_COMPILE_ENABLED = _compile_env != '0'
TORCH_COMPILE_MODE = os.getenv('DWARF_TORCH_COMPILE_MODE', 'default')
COMPILE_CAPTURE_SCALARS = os.getenv('DWARF_COMPILE_CAPTURE_SCALARS', '1') == '1'
COMPILE_CAPTURE_DYNAMIC = os.getenv('DWARF_COMPILE_CAPTURE_DYNAMIC', '1') == '1'
TORCH_COMPILE_DYNAMIC = os.getenv('DWARF_TORCH_COMPILE_DYNAMIC', '1') == '1'
TORCH_COMPILE_FULLGRAPH = os.getenv('DWARF_TORCH_COMPILE_FULLGRAPH', '0') == '1'
COMPILE_SUPPRESS_ERRORS = os.getenv('DWARF_COMPILE_SUPPRESS_ERRORS', '0') == '1'
_compile_budget_env = os.getenv('DWARF_COMPILE_ACTIVATION_BUDGET', '0.3').strip().lower()
COMPILE_ACTIVATION_BUDGET = None if _compile_budget_env in ('', 'none') else float(_compile_budget_env)

TOKENIZER_CANDIDATES = [os.environ.get('DWARF_TOKENIZER', 'tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json')]
PASSKEY_DISTANCES    = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS       = int(os.environ.get('DWARF_PASSKEY_TRIALS', '20'))
PASSKEY_BATCH_SIZE   = 8
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = os.environ.get('DWARF_CHECKPOINT_DIR', 'autoresearch/checkpoints')
_CKPT_BASE_DEFAULT = 'd512_l10_muon_olmo1_base_v1_staggered_movt' if os.getenv('DWARF_STAGGER_MOVT_PLANES', '1') == '1' else 'd512_l10_muon_olmo1_base_v1'
CKPT_BASE_NAME    = os.environ.get('DWARF_CKPT_BASE_NAME', _CKPT_BASE_DEFAULT)

# A/B switch for optimizer experiments. AdamW keeps the original PagedAdamW8bit baseline path.
# Muon uses torch.optim.Muon on 2D hidden matrices only and AdamW on topology/embedding/norm/etc.
OPTIMIZER_KIND = os.getenv('DWARF_OPT', 'muon').strip().lower()
if OPTIMIZER_KIND not in {'adamw', 'muon'}:
    raise ValueError(f"Unsupported DWARF_OPT={OPTIMIZER_KIND!r}; expected 'adamw' or 'muon'")
MUON_ADJUST_LR_FN = os.getenv('DWARF_MUON_ADJUST_LR_FN', 'match_rms_adamw')
MUON_MOMENTUM = float(os.getenv('DWARF_MUON_MOMENTUM', '0.95'))
MUON_NS_STEPS = int(os.getenv('DWARF_MUON_NS_STEPS', '5'))

CHECKPOINT_STRATEGY = os.getenv('DWARF_CKPT', 'none').lower()
DATASET_PATH = os.environ.get('DWARF_DATASET', 'datasets/dwarf_base_v1_olmo1tok_2048_2b.pt')
STAGGER_MOVT_PLANES = os.getenv('DWARF_STAGGER_MOVT_PLANES', '1') == '1'
PURE_DSQG_BASELINE = os.getenv('DWARF_PURE_DSQG', '0') == '1'
Q6_G128_ENABLED = os.getenv('DWARF_Q6_G128', '0') == '1'
Q6_G128_FUSED_CONSUME = os.getenv('DWARF_Q6_G128_FUSED_CONSUME', '0') == '1'
Q6_G128_FUSED_BLOCK_N = int(os.environ.get('DWARF_Q6_G128_FUSED_BLOCK_N', '32'))
Q6_G128_STAGE_C_TILE = int(os.environ.get('DWARF_Q6_G128_STAGE_C_TILE', '512'))
Q6_G128_STAGE_D_BACKWARD = os.getenv('DWARF_Q6_G128_STAGE_D_BACKWARD', '0') == '1'
Q6_G128_STAGE_E_BACKWARD = os.getenv('DWARF_Q6_G128_STAGE_E_BACKWARD', '0') == '1'
Q6_G128_STAGE_E_SCORES_ONLY = os.getenv('DWARF_Q6_G128_STAGE_E_SCORES_ONLY', '0') == '1'
Q6_G128_STAGE_E_GROUP_SPECIALIZE = os.getenv('DWARF_Q6_G128_STAGE_E_GROUP_SPECIALIZE', '0') == '1'
Q6_G128_STAGE_F2_PAIR_REUSE = os.getenv('DWARF_Q6_G128_STAGE_F2_PAIR_REUSE', '0') == '1'
Q6_G128_STAGE_F3_PAIR_BACKWARD = os.getenv('DWARF_Q6_G128_STAGE_F3_PAIR_BACKWARD', '0') == '1'
Q6_G128_STAGE_F3_SPARSE_MOVT = os.getenv('DWARF_Q6_G128_STAGE_F3_SPARSE_MOVT', '0') == '1'
Q6_G128_STAGE_F3_SPLIT_MOVT_CORR = os.getenv('DWARF_Q6_G128_STAGE_F3_SPLIT_MOVT_CORR', '0') == '1'
Q6_G128_STAGE_F3_SPLIT_VPHASE = os.getenv('DWARF_Q6_G128_STAGE_F3_SPLIT_VPHASE', '0') == '1'
Q6_G128_STAGE_F3_SPLIT_MOVT_PROB_SCRATCH = os.getenv('DWARF_Q6_G128_STAGE_F3_SPLIT_MOVT_PROB_SCRATCH', '0') == '1'
Q6_G128_STAGE_F3_SIDECAR_PAIR_DIRECT = os.getenv('DWARF_Q6_G128_STAGE_F3_SIDECAR_PAIR_DIRECT', '0') == '1'
Q6_G128_STAGE_E_NUM_WARPS = max(1, int(os.environ.get('DWARF_Q6_G128_STAGE_E_NUM_WARPS', '4')))
Q6_G128_NUM_KV_HEADS = int(os.environ.get('DWARF_Q6_G128_NUM_KV_HEADS', str(NUM_HEADS)))
if NUM_HEADS % Q6_G128_NUM_KV_HEADS != 0:
    raise ValueError(f'DWARF_Q6_G128_NUM_KV_HEADS={Q6_G128_NUM_KV_HEADS} must divide NUM_HEADS={NUM_HEADS}')
Q6_G128_LAYER_SPEC = os.getenv('DWARF_Q6_G128_LAYERS', 'all').strip().lower()
Q6_G128_SEED = int(os.environ.get('DWARF_Q6_G128_SEED', str(SEED + 610128)))


def _parse_int_tuple_env(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return tuple(int(v) for v in default)
    if raw.strip().lower() in {'none', 'off', 'disable', 'disabled', '[]'}:
        return tuple()
    out = []
    for part in raw.split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError(f'{name} did not contain any integer entries')
    return tuple(out)


DSQG_W_ENABLED = os.getenv('DWARF_DSQG_W', '0') == '1'
DSQG_W_SOURCEWISE = os.getenv('DWARF_DSQG_W_SOURCEWISE', '0') == '1'
DSQG_W_TRITON_SOURCEWISE = os.getenv('DWARF_DSQG_W_TRITON_SOURCEWISE', '0') == '1'
DSQG_W_DETACH_RECOMPOSER = os.getenv('DWARF_DSQG_W_DETACH_RECOMPOSER', '0') == '1'
DSQG_W_ACTIVE_SITE_MODE = os.getenv('DWARF_DSQG_W_ACTIVE_SITE_MODE', 'all').strip().lower()
DSQG_W_FAST_EVIDENCE_MEAN = os.getenv('DWARF_DSQG_W_FAST_EVIDENCE_MEAN', '0') == '1'
DSQG_W_MAX_CANDIDATES = int(os.environ.get('DWARF_DSQG_W_MAX_CANDIDATES', '32'))
DSQG_W_BOTTLENECK = int(os.environ.get('DWARF_DSQG_W_BOTTLENECK', '256'))
DSQG_W_GATE_INIT = float(os.environ.get('DWARF_DSQG_W_GATE_INIT', '-5.0'))
DSQG_W_GATE_LR_MULT = float(os.environ.get('DWARF_DSQG_W_GATE_LR_MULT', '1.25'))
DSQG_W_FUSE_INIT_STD = float(os.environ.get('DWARF_DSQG_W_FUSE_INIT_STD', '0.0001'))
DSQG_W_WIDTH_CELL = os.getenv('DWARF_DSQG_W_WIDTH_CELL', '0') == '1'
DSQG_W_WIDTH_BOTTLENECK = int(os.environ.get('DWARF_DSQG_W_WIDTH_BOTTLENECK', '64'))
DSQG_W_WIDTH_GATE_INIT = float(os.environ.get('DWARF_DSQG_W_WIDTH_GATE_INIT', '-5.0'))
DSQG_W_WIDTH_AUX_WEIGHT = float(os.environ.get('DWARF_DSQG_W_WIDTH_AUX_WEIGHT', '0.0'))
DSQG_W_WIDTH_ENTROPY_FLOOR = float(os.environ.get('DWARF_DSQG_W_WIDTH_ENTROPY_FLOOR', '1.5'))
DSQG_W_WIDTH_ENTROPY_WEIGHT = float(os.environ.get('DWARF_DSQG_W_WIDTH_ENTROPY_WEIGHT', '0.25'))
DSQG_W_TYPED_MIXER = os.getenv('DWARF_DSQG_W_TYPED_MIXER', '0') == '1'
DSQG_W_TYPED_MIXER_BOTTLENECK = int(os.environ.get('DWARF_DSQG_W_TYPED_MIXER_BOTTLENECK', '64'))
DSQG_W_TYPED_MIXER_GATE_INIT = float(os.environ.get('DWARF_DSQG_W_TYPED_MIXER_GATE_INIT', '-5.0'))
DSQG_W_QUERY_TYPE_BIAS = os.getenv('DWARF_DSQG_W_QUERY_TYPE_BIAS', '0') == '1'
DSQG_W_TYPED_HISA_REPS = os.getenv('DWARF_DSQG_W_TYPED_HISA_REPS', '0') == '1'
DSQG_W_EVIDENCE_BINDING_HUB = os.getenv('DWARF_DSQG_W_EVIDENCE_BINDING_HUB', '0') == '1'
DSQG_W_EBH_BOTTLENECK = int(os.environ.get('DWARF_DSQG_W_EBH_BOTTLENECK', '256'))
DSQG_W_EBH_GATE_INIT = float(os.environ.get('DWARF_DSQG_W_EBH_GATE_INIT', '-5.0'))
DSQG_W_EBH_PHASE_BANDS = int(os.environ.get('DWARF_DSQG_W_EBH_PHASE_BANDS', '4'))
DSQG_W_EBH_SCORE_FEATURES = os.getenv('DWARF_DSQG_W_EBH_SCORE_FEATURES', '1') != '0'
DSQG_W_EBH_PAIR_MIXER = os.getenv('DWARF_DSQG_W_EBH_PAIR_MIXER', '0') == '1'
DSQG_W_EBH_PAIR_RANK = int(os.environ.get('DWARF_DSQG_W_EBH_PAIR_RANK', '64'))
DSQG_W_EBH_PAIR_GATE_INIT = float(os.environ.get('DWARF_DSQG_W_EBH_PAIR_GATE_INIT', '-2.5'))
DSQG_W_EBH_SOURCEWISE_PACKET = os.getenv('DWARF_DSQG_W_EBH_SOURCEWISE_PACKET', '0') == '1'
DSQG_W_EBH_TRITON_LANE_ACCUM = os.getenv('DWARF_DSQG_W_EBH_TRITON_LANE_ACCUM', '0') == '1'
DSQG_W_EVIDENCE_PRIOR = os.getenv('DWARF_DSQG_W_EVIDENCE_PRIOR', '0') == '1'
DSQG_W_EVIDENCE_PRIOR_CLIP = float(os.environ.get('DWARF_DSQG_W_EVIDENCE_PRIOR_CLIP', '2.0'))
DSQG_W_EVIDENCE_PRIOR_INIT_SCALE = float(os.environ.get('DWARF_DSQG_W_EVIDENCE_PRIOR_INIT_SCALE', '0.0'))
DSQG_W_CANDIDATE_QUOTAS = os.getenv('DWARF_DSQG_W_CANDIDATE_QUOTAS', '0') == '1'
DSQG_W_QUOTA_HISA_MAX = int(os.environ.get('DWARF_DSQG_W_QUOTA_HISA_MAX', '0'))
DSQG_W_CANDIDATE_WORKSPACE = os.getenv('DWARF_DSQG_W_CANDIDATE_WORKSPACE', '0') == '1'
DSQG_W_CANDIDATE_WORKSPACE_DIM = int(os.environ.get('DWARF_DSQG_W_CANDIDATE_WORKSPACE_DIM', '64'))
DSQG_W_CANDIDATE_WORKSPACE_PHASE_BANDS = int(os.environ.get('DWARF_DSQG_W_CANDIDATE_WORKSPACE_PHASE_BANDS', '4'))
DSQG_W_CANDIDATE_WORKSPACE_SCORE_FEATURES = os.getenv('DWARF_DSQG_W_CANDIDATE_WORKSPACE_SCORE_FEATURES', '1') != '0'
DSQG_W_CANDIDATE_WORKSPACE_QUERY_SCORES = os.getenv('DWARF_DSQG_W_CANDIDATE_WORKSPACE_QUERY_SCORES', '1') != '0'
DSQG_W_CANDIDATE_WORKSPACE_PAIR_TRANSFER = os.getenv('DWARF_DSQG_W_CANDIDATE_WORKSPACE_PAIR_TRANSFER', '0') == '1'
DSQG_W_CANDIDATE_WORKSPACE_PAIR_GATE_INIT = float(os.environ.get('DWARF_DSQG_W_CANDIDATE_WORKSPACE_PAIR_GATE_INIT', '-2.5'))
DSQG_W_DSR_CANDIDATES = os.getenv('DWARF_DSQG_W_DSR_CANDIDATES', '1') != '0'
DSQG_W_LOCAL_OFFSETS = _parse_int_tuple_env('DWARF_DSQG_W_LOCAL_OFFSETS', (1, 2, 4, 8))
DSQG_W_LONG_OFFSETS = _parse_int_tuple_env('DWARF_DSQG_W_LONG_OFFSETS', (16, 32, 64, 128, 256, 512, 1024, 2048))
DSQG_W_QUESTION_ENABLED = os.getenv('DWARF_DSQG_W_QUESTION', '0') == '1'
DSQG_W_K_QUESTION = int(os.environ.get('DWARF_DSQG_W_K_QUESTION', '4')) if DSQG_W_QUESTION_ENABLED else 0
DSQG_W_HISA_L3_ENABLED = os.getenv('DWARF_DSQG_W_HISA_L3', '0') == '1'
DSQG_W_K_HISA_EVIDENCE = int(os.environ.get('DWARF_DSQG_W_K_HISA_EVIDENCE', '8')) if DSQG_W_HISA_L3_ENABLED else 0
DSQG_W_K_L3_SKIP = int(os.environ.get('DWARF_DSQG_W_K_L3_SKIP', '4')) if DSQG_W_HISA_L3_ENABLED else 0


def _parse_dsqg_w_site_specs(raw: str | None):
    spec = 'final' if raw is None or str(raw).strip() == '' else str(raw).strip()
    out = []
    seen = set()
    for part in spec.split(','):
        item = part.strip().lower()
        if not item:
            continue
        if item in ('final', 'f'):
            value = 'final'
        else:
            if item.startswith('layer_'):
                item = item[len('layer_'):]
            value = int(item)
            if value < 0 or value >= NUM_LAYERS:
                raise ValueError(f'DWARF_DSQG_W_SITES layer index {value} outside [0, {NUM_LAYERS - 1}]')
        if value not in seen:
            out.append(value)
            seen.add(value)
    if not out:
        raise ValueError('DWARF_DSQG_W_SITES did not contain any site entries')
    return tuple(out)


def _dsqg_w_training_candidate_indices(x: torch.Tensor):
    if not DSQG_W_ENABLED:
        return None, None, None
    bsz, seq_len = x.shape
    device = x.device
    question_indices = None
    if DSQG_W_QUESTION_ENABLED and DSQG_W_K_QUESTION > 0:
        base = torch.linspace(0, max(seq_len - 1, 0), steps=DSQG_W_K_QUESTION, device=device)
        question_indices = base.round().to(torch.long).clamp_(0, max(seq_len - 1, 0)).unsqueeze(0).expand(bsz, -1).contiguous()
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    hisa_evidence_indices = None
    l3_skip_indices = None
    if DSQG_W_HISA_L3_ENABLED and DSQG_W_K_HISA_EVIDENCE > 0:
        offsets = [1, 8, 32, 128, 256, 512]
        while len(offsets) < DSQG_W_K_HISA_EVIDENCE:
            offsets.append(offsets[-1] * 2)
        rows = [(positions - off).clamp_min(0) for off in offsets[:DSQG_W_K_HISA_EVIDENCE]]
        hisa_evidence_indices = torch.stack(rows, dim=-1).unsqueeze(0).expand(bsz, -1, -1).contiguous()
    if DSQG_W_HISA_L3_ENABLED and DSQG_W_K_L3_SKIP > 0:
        offsets = [16, 64, 256, 1024]
        while len(offsets) < DSQG_W_K_L3_SKIP:
            offsets.append(offsets[-1] * 2)
        rows = [(positions - off).clamp_min(0) for off in offsets[:DSQG_W_K_L3_SKIP]]
        l3_skip_indices = torch.stack(rows, dim=-1).unsqueeze(0).expand(bsz, -1, -1).contiguous()
    return question_indices, hisa_evidence_indices, l3_skip_indices


DSQG_W_SITE_SPECS = _parse_dsqg_w_site_specs(os.getenv('DWARF_DSQG_W_SITES'))


def _dsqg_w_site_key(site):
    return 'final' if site == 'final' else f'layer_{int(site)}'


def _profile_range(name):
    if PROFILE_DSQG_W:
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


def _dsqg_w_candidate_path_label():
    parts = []
    if DSQG_W_HISA_L3_ENABLED and DSQG_W_DSR_CANDIDATES and not PURE_DSQG_BASELINE:
        parts.append('DSR_SELECTED')
    if DSQG_W_LOCAL_OFFSETS:
        parts.append('LOCAL')
    if DSQG_W_LONG_OFFSETS:
        parts.append('LONG')
    if DSQG_W_QUESTION_ENABLED:
        parts.append('QUESTION')
    if DSQG_W_HISA_L3_ENABLED and not DSQG_W_DSR_CANDIDATES:
        parts.extend(['HISA_EVIDENCE_FALLBACK', 'L3_SKIP'])
    elif DSQG_W_HISA_L3_ENABLED and DSQG_W_K_L3_SKIP > 0:
        parts.append('L3_SKIP')
    parts.append('NULL')
    return '_'.join(parts)


def _parse_q6_layer_spec(spec):
    if spec in ('', 'all', '*'):
        return None
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out

_Q6_G128_LAYER_FILTER = _parse_q6_layer_spec(Q6_G128_LAYER_SPEC)
Q6_G128_LAYERS = set(range(NUM_LAYERS)) if _Q6_G128_LAYER_FILTER is None else set(_Q6_G128_LAYER_FILTER)

def _q6_enabled_for_layer(layer_index: int) -> bool:
    return Q6_G128_ENABLED and (int(layer_index) in Q6_G128_LAYERS)


def _movt_plane_shift_for_dsqg_index(dsqg_index: int, head_dim: int, r_planes: int = R_PLANES) -> int:
    """Return the per-DSQG-layer MOVT plane shift; DSR/HISA layers do not consume slots."""
    segment = max(2, int(head_dim) // int(r_planes))
    slots = max(1, segment // 2)
    return 2 * (int(dsqg_index) % slots)

# =============================================================================
# LAYER LAYOUT: L=10, DSR/HISA@L3
# Pre-DSR: 1 triad (L0-2), Post-DSR: 2 full triads (L4-9)
# =============================================================================

_HISA_LAYER_LAYOUT = [
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L00
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L01
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, True),    # L02
    ('DSR', None, 0, 0, False),                    # L03: HierarchicalSparseAttentionV15HISA
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L04
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L05
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L06
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L07
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L08
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L09
]

_PURE_DSQG_LAYER_LAYOUT = [
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L00
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L01
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, True),    # L02
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L03: pure-DSQG v1 control slot
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L04
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L05
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L06
    ('A', GROUP_A, J_SMALL_A, J_LARGE_A, False),   # L07
    ('B', GROUP_B, J_SMALL_B, J_LARGE_B, False),   # L08
    ('C', GROUP_C, J_SMALL_C, J_LARGE_C, False),   # L09
]

_BASE_LAYER_LAYOUT = _PURE_DSQG_LAYER_LAYOUT if PURE_DSQG_BASELINE else _HISA_LAYER_LAYOUT
LAYER_LAYOUT = [
    (label, offsets, js, jl, has_if and PRE_HISA_EMA_ENABLED)
    for label, offsets, js, jl, has_if in _BASE_LAYER_LAYOUT
]

assert len(LAYER_LAYOUT) == NUM_LAYERS
if not PRE_HISA_EMA_ENABLED:
    print('  Pre-HISA causal EMA/preIF disabled (DWARF_PRE_HISA_EMA=0)')


def _env_float_or_none(name):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == '':
        return None
    return float(raw)


def _dsqg_w_legacy_mode_labels():
    labels = []
    if DSQG_W_EVIDENCE_BINDING_HUB and not DSQG_W_EBH_SOURCEWISE_PACKET:
        labels.append('legacy_materialized_ebh')
    if DSQG_W_EVIDENCE_BINDING_HUB and DSQG_W_EBH_SOURCEWISE_PACKET and not DSQG_W_EBH_SCORE_FEATURES:
        labels.append('legacy_packet_no_score')
    if DSQG_W_EBH_PAIR_MIXER:
        labels.append('legacy_ebh_pair_mixer')
    if DSQG_W_EVIDENCE_BINDING_HUB and DSQG_W_EBH_SOURCEWISE_PACKET and (DSQG_W_WIDTH_CELL or DSQG_W_TYPED_MIXER):
        labels.append('legacy_packet_semantic_approx')
    if os.getenv('DWARF_DSQG_W_PROJECTED_WIDTH_CONTROL', '0') == '1':
        labels.append('legacy_projected_width_control')
    if DSQG_W_FAST_EVIDENCE_MEAN and os.getenv('DWARF_DSQG_W_ALLOW_FAST_EVIDENCE_MEAN_BYPASS', '0') == '1':
        labels.append('legacy_fast_evidence_mean_bypass')
    if not PRE_HISA_EMA_ENABLED:
        labels.append('legacy_no_pre_hisa_ema')
    return labels


def _dsqg_w_lane_label():
    if not DSQG_W_ENABLED:
        return 'disabled'
    if _dsqg_w_legacy_mode_labels():
        return 'legacy_guarded'
    if (
        DSQG_W_SOURCEWISE
        and DSQG_W_TRITON_SOURCEWISE
        and DSQG_W_WIDTH_CELL
        and DSQG_W_TYPED_MIXER
        and not DSQG_W_EVIDENCE_BINDING_HUB
        and PRE_HISA_EMA_ENABLED
    ):
        return 'lane_a_no_ebh_lateral_open'
    if (
        DSQG_W_SOURCEWISE
        and DSQG_W_TRITON_SOURCEWISE
        and DSQG_W_EVIDENCE_BINDING_HUB
        and DSQG_W_EBH_SOURCEWISE_PACKET
        and DSQG_W_EBH_SCORE_FEATURES
        and DSQG_W_EBH_TRITON_LANE_ACCUM
        and not DSQG_W_WIDTH_CELL
        and not DSQG_W_TYPED_MIXER
        and PRE_HISA_EMA_ENABLED
    ):
        return 'lane_b_ebh_packet_triton_score'
    return 'other'


def _layer_layout_marker():
    payload = [(label, len(offsets) if offsets is not None else 0, js, jl, has_if) for label, offsets, js, jl, has_if in LAYER_LAYOUT]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]


# =============================================================================
# MODEL BLOCKS
# =============================================================================

def _amp_context(device):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()

def _unwrap_compiled_module(module):
    return getattr(module, '_orig_mod', module)

def _causal_ema(xi, ema_factor, floor=EMA_FLOOR):
    return _causal_ema_scan(xi, ema_factor, floor=floor)

def _agc_normalize(pool, eps=1e-6):
    D = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


def _q6_g128_triton_direct_ste_causal_gather(cache, offsets, *, seed):
    """Triton direct q6_g128 causal gather with BF16-gather STE gradients.

    The forward q6 values come from the Phase-3 Triton direct-gather kernel,
    which decodes resident q6 directly into the gathered [B,H,N,J,D] read output
    and does not allocate a BF16 full-sequence scratch cache. Gradients still use
    the matching BF16 causal gather path via a straight-through estimator so this
    smoke can train through the original q/k/v projections while exercising the
    production-shaped q6 read perturbation in the real trainer loop.
    """
    if not _Q6_LAYOUT_AVAILABLE:
        raise RuntimeError('DWARF_Q6_G128=1 requires kernels/q6_g128/layout.py')
    if not _Q6_TRITON_AVAILABLE:
        raise RuntimeError('DWARF_Q6_G128=1 requires kernels/q6_g128/decode.py')
    if not cache.is_cuda:
        raise RuntimeError('q6_g128 Triton direct-gather smoke path requires CUDA')
    if cache.shape[-1] != 64:
        raise ValueError(f'q6_g128 smoke path requires head_dim=64, got {cache.shape[-1]}')
    offsets = [int(o) for o in offsets]
    layout = _q6_layout_mod.pack_q6_g128_cache_layout(cache, seed=seed)
    q6_gather, mask, direct_report = _q6_triton_mod.triton_direct_decode_gather(
        layout, offsets, return_report=True
    )
    bf16_gather, bf16_mask = _q6_layout_mod.bf16_causal_gather(cache.to(torch.bfloat16), offsets)
    if not torch.equal(mask, bf16_mask):
        raise RuntimeError('q6 causal mask diverged from BF16 causal gather mask')
    gathered = bf16_gather.float() + (q6_gather.float() - bf16_gather.float()).detach()
    storage = layout.storage_report()
    return gathered, mask, {
        **storage,
        'read_implementation': 'q6_triton_direct_gather',
        'scratch_mode': 'direct_q6_decode_to_gathered_output',
        'peak_scratch_bytes': int(direct_report.get('peak_scratch_bytes', 0)),
        'peak_scratch_vs_full_scratch': float(direct_report.get('peak_scratch_vs_full_scratch', 0.0)),
        'output_bytes': int(q6_gather.numel() * q6_gather.element_size()),
    }


def _q6_g128_effective_stage_c_tile(seq_len: int, requested: int | None = None) -> int:
    """Bound Stage-C backward replay tiles so the replay never spans a full sequence."""
    seq_len = int(seq_len)
    requested = Q6_G128_STAGE_C_TILE if requested is None else int(requested)
    if seq_len <= 1:
        return 1
    max_bounded_tile = max(1, seq_len // 2)
    return min(max(1, requested), max_bounded_tile)


def _q6_g128_layout_from_saved(payload, scales, *, batch, heads, seq_len, head_dim, seed):
    padded_seq_len = int(scales.shape[2]) * int(_q6_layout_mod.TOKENS_PER_GROUP)
    return _q6_layout_mod.DwarfQ6G128CacheLayout(
        payload=payload,
        scales=scales,
        batch=int(batch),
        heads=int(heads),
        seq_len=int(seq_len),
        padded_seq_len=padded_seq_len,
        head_dim=int(head_dim),
        seed=int(seed),
    )


def _q6_g128_decode_gather_tile(layout, offsets, *, start, end):
    """Decode only one query tile's q6 causal-offset gather, never full [B,H,N,J,D]."""
    idx, valid = _q6_layout_mod.causal_offset_index(layout.seq_len, offsets, device=layout.payload.device)
    idx = idx[start:end]
    valid = valid[start:end]
    pair_idx = torch.div(idx, _q6_layout_mod.TOKENS_PER_GROUP, rounding_mode='floor')
    token_half = torch.remainder(idx, _q6_layout_mod.TOKENS_PER_GROUP)
    selected_payload = layout.payload[:, :, pair_idx, :]
    selected_scales = layout.scales[:, :, pair_idx]
    decoded_pairs = _q6_layout_mod._decode_pair_payload(selected_payload, selected_scales)
    gather_idx = token_half.reshape(1, 1, end - start, len(offsets), 1, 1).expand(
        layout.batch, layout.heads, end - start, len(offsets), 1, layout.head_dim
    )
    gathered = torch.gather(decoded_pairs, dim=4, index=gather_idx).squeeze(4).to(torch.bfloat16)
    gathered = gathered * valid.reshape(1, 1, end - start, len(offsets), 1).to(gathered.dtype)
    return gathered, idx, valid


def _q6_g128_decode_gather_tile_triton(layout, offsets, *, start, end, block=256):
    """Decode one query tile's q6 causal-offset gather directly with Triton.

    This is the Stage-D read-side replay bridge: it avoids the PyTorch pair-decode
    path in backward while still materializing only bounded [B,H,T,J,D] tiles.
    """
    if (
        not _Q6_TRITON_AVAILABLE
        or not layout.payload.is_cuda
        or not layout.scales.is_cuda
        or layout.head_dim != 64
    ):
        return _q6_g128_decode_gather_tile(layout, offsets, start=start, end=end)
    start = int(start)
    end = int(end)
    if start < 0 or end > layout.seq_len or start >= end:
        raise ValueError(f'expected non-empty decode tile within [0,{layout.seq_len}], got [{start},{end})')
    offsets = [int(o) for o in offsets]
    idx, valid = _q6_layout_mod.causal_offset_index(layout.seq_len, offsets, device=layout.payload.device)
    idx = idx[start:end]
    valid = valid[start:end]
    tile_tokens = end - start
    adjusted_offsets = torch.tensor([o - start for o in offsets], device=layout.payload.device, dtype=torch.int32)
    out = torch.empty(
        (layout.batch, layout.heads, tile_tokens, len(offsets), layout.head_dim),
        device=layout.payload.device,
        dtype=torch.bfloat16,
    )
    total_values = out.numel()
    grid = (triton.cdiv(total_values, int(block)),)
    _q6_triton_mod._decode_q6_g128_direct_gather_kernel[grid](
        layout.payload,
        layout.scales,
        adjusted_offsets,
        out,
        total_values,
        tile_tokens,
        len(offsets),
        layout.heads,
        layout.pair_groups,
        BLOCK=int(block),
    )
    return out, idx, valid


def _q6_g128_rotate_sparse_values_tile(values, y_tile, z_full, idx, valid, gated_phase_base,
                                       gated_phase_gain, *, j_small, plane_shift):
    j_val = int(values.shape[3])
    if int(j_small) >= j_val:
        return values
    out = values.clone()
    hd_segment = values.shape[-1] // R_PLANES
    for i in range(int(j_small), j_val):
        pi = i - int(j_small)
        kp = idx[:, i]
        val_i = valid[:, i]
        for r in range(R_PLANES):
            ch_a = r * hd_segment + int(plane_shift)
            ch_b = ch_a + 1
            z_i = z_full[:, :, kp, r]
            theta = (gated_phase_base[pi, :, r].reshape(1, -1, 1)
                     + gated_phase_gain[pi, :, r].reshape(1, -1, 1) * y_tile[:, :, :, r].float() * z_i.float())
            theta = torch.where(val_i.reshape(1, 1, -1), theta, torch.zeros_like(theta))
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            old_a = out[:, :, :, i, ch_a].clone()
            old_b = out[:, :, :, i, ch_b].clone()
            out[:, :, :, i, ch_a] = cos_t * old_a - sin_t * old_b
            out[:, :, :, i, ch_b] = sin_t * old_a + cos_t * old_b
    return out


def _q6_g128_dsqg_consume_tile(q_tile, k_g, v_g, valid, idx, pos_bias, scale_embed,
                               gated_phase_base, gated_phase_gain, y_tile, z_full,
                               *, j_small, plane_shift):
    b, h, t, d = q_tile.shape
    j_val = int(k_g.shape[3])
    qf = q_tile.float()
    sc = 1.0 / math.sqrt(float(d))
    scores = torch.einsum('bhtd,bhtjd->bhtj', qf, k_g.float()) * sc
    scores = scores + torch.einsum('bhtd,jd->bhtj', qf, scale_embed.float()) * sc
    scores = scores + pos_bias.float().transpose(0, 1).reshape(1, h, 1, j_val)
    mask = valid.reshape(1, 1, t, j_val)
    scores = scores.masked_fill(~mask, float('-inf'))
    max_scores = scores.amax(dim=-1, keepdim=True)
    all_invalid = ~torch.isfinite(max_scores)
    safe_max = torch.where(all_invalid, torch.zeros_like(max_scores), max_scores)
    exp_scores = torch.exp(scores - safe_max).masked_fill(~mask, 0.0)
    denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    probs = exp_scores / denom
    v_rot = _q6_g128_rotate_sparse_values_tile(
        v_g.float(), y_tile, z_full, idx, valid, gated_phase_base, gated_phase_gain,
        j_small=j_small, plane_shift=plane_shift,
    )
    return torch.sum(probs.unsqueeze(-1) * v_rot, dim=3).to(dtype=q_tile.dtype)


def _q6_g128_dsqg_consume_tile_backward_manual(
    q_tile, k_g, v_g, valid, idx, pos_bias, scale_embed,
    phase_gate, phase_base, phase_gain, y_tile, z_full, grad_out,
    *, j_small, plane_shift,
):
    """Manual backward for one Stage-C replay tile, avoiding autograd graph construction."""
    b, h, t, d = q_tile.shape
    j_val = int(k_g.shape[3])
    qf = q_tile.float()
    kf = k_g.float()
    vf = v_g.float()
    go = grad_out.float()
    mask = valid.reshape(1, 1, t, j_val)
    sc = 1.0 / math.sqrt(float(d))

    gate = torch.sigmoid(phase_gate.float())
    gated_phase_base = phase_base.float() * gate[:, None, None]
    gated_phase_gain = phase_gain.float() * gate[:, None, None]

    scores = torch.einsum('bhtd,bhtjd->bhtj', qf, kf) * sc
    scores = scores + torch.einsum('bhtd,jd->bhtj', qf, scale_embed.float()) * sc
    scores = scores + pos_bias.float().transpose(0, 1).reshape(1, h, 1, j_val)
    scores = scores.masked_fill(~mask, float('-inf'))
    max_scores = scores.amax(dim=-1, keepdim=True)
    all_invalid = ~torch.isfinite(max_scores)
    safe_max = torch.where(all_invalid, torch.zeros_like(max_scores), max_scores)
    exp_scores = torch.exp(scores - safe_max).masked_fill(~mask, 0.0)
    denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    probs = exp_scores / denom

    v_rot = _q6_g128_rotate_sparse_values_tile(
        vf, y_tile.float(), z_full.float(), idx, valid, gated_phase_base, gated_phase_gain,
        j_small=j_small, plane_shift=plane_shift,
    )
    grad_v_rot = probs.unsqueeze(-1) * go.unsqueeze(3)
    grad_probs = torch.sum(go.unsqueeze(3) * v_rot, dim=-1)
    grad_scores = probs * (grad_probs - torch.sum(grad_probs * probs, dim=-1, keepdim=True))
    grad_scores = grad_scores.masked_fill(~mask, 0.0)

    grad_q = (
        torch.einsum('bhtj,bhtjd->bhtd', grad_scores, kf)
        + torch.einsum('bhtj,jd->bhtd', grad_scores, scale_embed.float())
    ) * sc
    grad_k = grad_scores.unsqueeze(-1) * qf.unsqueeze(3) * sc
    grad_scale = torch.einsum('bhtj,bhtd->jd', grad_scores, qf) * sc
    grad_pos = grad_scores.sum(dim=(0, 2)).transpose(0, 1).contiguous()

    grad_v = grad_v_rot.clone()
    grad_y = torch.zeros_like(y_tile, dtype=torch.float32)
    grad_z = torch.zeros_like(z_full, dtype=torch.float32)
    grad_gated_base = torch.zeros_like(gated_phase_base, dtype=torch.float32)
    grad_gated_gain = torch.zeros_like(gated_phase_gain, dtype=torch.float32)

    if int(j_small) < j_val:
        hd_segment = d // R_PLANES
        for i in range(int(j_small), j_val):
            pi = i - int(j_small)
            kp = idx[:, i]
            val_i = valid[:, i].reshape(1, 1, t)
            for r in range(R_PLANES):
                ch_a = r * hd_segment + int(plane_shift)
                ch_b = ch_a + 1
                va = vf[:, :, :, i, ch_a]
                vb = vf[:, :, :, i, ch_b]
                y_r = y_tile[:, :, :, r].float()
                z_i = z_full[:, :, kp, r].float()
                gb = gated_phase_base[pi, :, r].reshape(1, h, 1)
                gg = gated_phase_gain[pi, :, r].reshape(1, h, 1)
                theta_raw = gb + gg * y_r * z_i
                theta = torch.where(val_i, theta_raw, torch.zeros_like(theta_raw))
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                go_a = grad_v_rot[:, :, :, i, ch_a]
                go_b = grad_v_rot[:, :, :, i, ch_b]
                grad_v[:, :, :, i, ch_a] = cos_t * go_a + sin_t * go_b
                grad_v[:, :, :, i, ch_b] = -sin_t * go_a + cos_t * go_b
                dtheta = ((-sin_t * va - cos_t * vb) * go_a
                          + (cos_t * va - sin_t * vb) * go_b)
                dtheta = dtheta * val_i.to(dtheta.dtype)
                grad_gated_base[pi, :, r] += dtheta.sum(dim=(0, 2))
                grad_gated_gain[pi, :, r] += (dtheta * y_r * z_i).sum(dim=(0, 2))
                grad_y[:, :, :, r] += dtheta * gg * z_i
                z_contrib = dtheta * gg * y_r
                grad_z[:, :, :, r].index_add_(2, kp, z_contrib)

    grad_phase_base = grad_gated_base * gate[:, None, None]
    grad_phase_gain = grad_gated_gain * gate[:, None, None]
    grad_gate = (grad_gated_base * phase_base.float() + grad_gated_gain * phase_gain.float()).sum(dim=(1, 2))
    grad_phase_gate = grad_gate * gate * (1.0 - gate)

    return (
        grad_q, grad_k, grad_v, grad_pos, grad_scale,
        grad_phase_gate, grad_phase_base, grad_phase_gain, grad_y, grad_z,
    )


def _q6_g128_dsqg_consume_tile_backward_stage_d_vectorized(
    q_tile, k_g, v_g, valid, idx, pos_bias, scale_embed,
    phase_gate, phase_base, phase_gain, y_tile, z_full, grad_out,
    *, j_small, plane_shift,
):
    """Stage-D first attempt: vectorize the MOVT-gradient side of tile backward.

    This keeps the same bounded-tile STE surface as Stage-C, but removes the
    Python loop over every large offset x R-plane in the MOVT portion.  It is
    intentionally still a safe fallback-friendly slice rather than a full Triton
    backward kernel.
    """
    b, h, t, d = q_tile.shape
    h_kv = int(k_g.shape[1])
    if int(v_g.shape[1]) != h_kv:
        raise RuntimeError(f'q6 Stage-D fallback K/V head mismatch: K={tuple(k_g.shape)} V={tuple(v_g.shape)}')
    if h_kv != h:
        if h % h_kv != 0:
            raise RuntimeError(f'q6 Stage-D fallback Hq={h} must be divisible by Hkv={h_kv}')
        kv_group = h // h_kv
    else:
        kv_group = 1
    j_val = int(k_g.shape[3])
    qf = q_tile.float()
    kf = k_g.float()
    vf = v_g.float()
    if kv_group != 1:
        # Hkv/GQA path: decoded q6 K/V are resident at Hkv, while q/pos/MOVT
        # tensors are indexed by query head.  Expand for per-query-head math in
        # this safe fallback and reduce gathered K/V gradients before returning.
        kf = kf.repeat_interleave(kv_group, dim=1).contiguous()
        vf = vf.repeat_interleave(kv_group, dim=1).contiguous()
    go = grad_out.float()
    mask = valid.reshape(1, 1, t, j_val)
    sc = 1.0 / math.sqrt(float(d))

    gate = torch.sigmoid(phase_gate.float())
    gated_phase_base = phase_base.float() * gate[:, None, None]
    gated_phase_gain = phase_gain.float() * gate[:, None, None]

    scores = torch.einsum('bhtd,bhtjd->bhtj', qf, kf) * sc
    scores = scores + torch.einsum('bhtd,jd->bhtj', qf, scale_embed.float()) * sc
    scores = scores + pos_bias.float().transpose(0, 1).reshape(1, h, 1, j_val)
    scores = scores.masked_fill(~mask, float('-inf'))
    max_scores = scores.amax(dim=-1, keepdim=True)
    all_invalid = ~torch.isfinite(max_scores)
    safe_max = torch.where(all_invalid, torch.zeros_like(max_scores), max_scores)
    exp_scores = torch.exp(scores - safe_max).masked_fill(~mask, 0.0)
    denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    probs = exp_scores / denom

    v_rot = _q6_g128_rotate_sparse_values_tile(
        vf, y_tile.float(), z_full.float(), idx, valid, gated_phase_base, gated_phase_gain,
        j_small=j_small, plane_shift=plane_shift,
    )
    grad_v_rot = probs.unsqueeze(-1) * go.unsqueeze(3)
    grad_probs = torch.sum(go.unsqueeze(3) * v_rot, dim=-1)
    grad_scores = probs * (grad_probs - torch.sum(grad_probs * probs, dim=-1, keepdim=True))
    grad_scores = grad_scores.masked_fill(~mask, 0.0)

    grad_q = (
        torch.einsum('bhtj,bhtjd->bhtd', grad_scores, kf)
        + torch.einsum('bhtj,jd->bhtd', grad_scores, scale_embed.float())
    ) * sc
    grad_k = grad_scores.unsqueeze(-1) * qf.unsqueeze(3) * sc
    grad_scale = torch.einsum('bhtj,bhtd->jd', grad_scores, qf) * sc
    grad_pos = grad_scores.sum(dim=(0, 2)).transpose(0, 1).contiguous()

    grad_v = grad_v_rot.clone()
    grad_y = torch.zeros_like(y_tile, dtype=torch.float32)
    grad_z = torch.zeros_like(z_full, dtype=torch.float32)
    grad_gated_base = torch.zeros_like(gated_phase_base, dtype=torch.float32)
    grad_gated_gain = torch.zeros_like(gated_phase_gain, dtype=torch.float32)

    j_small_i = int(j_small)
    if j_small_i < j_val:
        hd_segment = d // R_PLANES
        large = j_val - j_small_i
        device = vf.device
        r_idx = torch.arange(R_PLANES, device=device, dtype=torch.long)
        ch_a = r_idx * hd_segment + int(plane_shift)
        ch_b = ch_a + 1

        vf_large = vf[:, :, :, j_small_i:, :]
        grad_v_rot_large = grad_v_rot[:, :, :, j_small_i:, :]
        va = vf_large[..., ch_a]
        vb = vf_large[..., ch_b]
        go_a = grad_v_rot_large[..., ch_a]
        go_b = grad_v_rot_large[..., ch_b]

        kp = idx[:, j_small_i:].to(device=device, dtype=torch.long)
        valid_large = valid[:, j_small_i:].to(device=device)
        z_i = z_full[:, :, kp, :].float()
        y_r = y_tile.float().unsqueeze(3)
        gb = gated_phase_base.permute(1, 0, 2).reshape(1, h, 1, large, R_PLANES)
        gg = gated_phase_gain.permute(1, 0, 2).reshape(1, h, 1, large, R_PLANES)
        val = valid_large.reshape(1, 1, t, large, 1)
        theta_raw = gb + gg * y_r * z_i
        theta = torch.where(val, theta_raw, torch.zeros_like(theta_raw))
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        gv_a = cos_t * go_a + sin_t * go_b
        gv_b = -sin_t * go_a + cos_t * go_b
        grad_v_large = grad_v[:, :, :, j_small_i:, :]
        grad_v_large[..., ch_a] = gv_a
        grad_v_large[..., ch_b] = gv_b

        dtheta = ((-sin_t * va - cos_t * vb) * go_a
                  + (cos_t * va - sin_t * vb) * go_b)
        dtheta = dtheta * val.to(dtheta.dtype)
        grad_gated_base += dtheta.sum(dim=(0, 2)).permute(1, 0, 2).contiguous()
        grad_gated_gain += (dtheta * y_r * z_i).sum(dim=(0, 2)).permute(1, 0, 2).contiguous()
        grad_y += (dtheta * gg * z_i).sum(dim=3)

        z_contrib = dtheta * gg * y_r
        flat_idx = kp.reshape(t * large)
        flat_contrib = z_contrib.reshape(b, h, t * large, R_PLANES)
        scatter_idx = flat_idx.reshape(1, 1, t * large, 1).expand(b, h, t * large, R_PLANES)
        grad_z.scatter_add_(2, scatter_idx, flat_contrib)

    grad_phase_base = grad_gated_base * gate[:, None, None]
    grad_phase_gain = grad_gated_gain * gate[:, None, None]
    grad_gate = (grad_gated_base * phase_base.float() + grad_gated_gain * phase_gain.float()).sum(dim=(1, 2))
    grad_phase_gate = grad_gate * gate * (1.0 - gate)

    if kv_group != 1:
        grad_k = grad_k.reshape(b, h_kv, kv_group, t, j_val, d).sum(dim=2).contiguous()
        grad_v = grad_v.reshape(b, h_kv, kv_group, t, j_val, d).sum(dim=2).contiguous()

    return (
        grad_q, grad_k, grad_v, grad_pos, grad_scale,
        grad_phase_gate, grad_phase_base, grad_phase_gain, grad_y, grad_z,
    )


def _q6_g128_accumulate_gathered_grads_vectorized(grad_seq, gathered_grads, valid, idx):
    """Accumulate gathered [B,H,T,J,D] gradients into [B,H,N,D] without a Python J loop."""
    if grad_seq is None:
        return None
    b, h, t, j_val, d = gathered_grads.shape
    if t == 0 or j_val == 0:
        return grad_seq
    idx = idx.to(device=grad_seq.device, dtype=torch.long)
    valid = valid.to(device=grad_seq.device)
    contrib = gathered_grads.float() * valid.reshape(1, 1, t, j_val, 1).to(gathered_grads.dtype)
    flat_idx = idx.reshape(t * j_val)
    flat_contrib = contrib.reshape(b, h, t * j_val, d)
    scatter_idx = flat_idx.reshape(1, 1, t * j_val, 1).expand(b, h, t * j_val, d)
    grad_seq.scatter_add_(2, scatter_idx, flat_contrib)
    return grad_seq


if _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE:
    @triton.jit
    def _q6_g128_accumulate_gathered_grads_triton_kernel(
        GATHERED, VALID, IDX, GRAD,
        TOTAL: tl.constexpr,
        B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, T: tl.constexpr, J: tl.constexpr, D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < TOTAL
        d = offs % D
        tmp = offs // D
        j = tmp % J
        tmp = tmp // J
        t = tmp % T
        tmp = tmp // T
        h = tmp % H
        b = tmp // H
        is_valid = tl.load(VALID + t * J + j, mask=mask, other=0).to(tl.int1)
        token = tl.load(IDX + t * J + j, mask=mask, other=0).to(tl.int64)
        val = tl.load(GATHERED + offs, mask=mask & is_valid, other=0.0).to(tl.float32)
        grad_off = ((b * H + h) * N + token) * D + d
        tl.atomic_add(GRAD + grad_off, val, sem="relaxed", mask=mask & is_valid)


def _q6_g128_accumulate_gathered_grads_triton(grad_seq, gathered_grads, valid, idx):
    """Triton atomic-add accumulation for gathered q/k/v grads; falls back safely."""
    if (
        grad_seq is None
        or not _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE
        or not grad_seq.is_cuda
        or not gathered_grads.is_cuda
    ):
        return _q6_g128_accumulate_gathered_grads_vectorized(grad_seq, gathered_grads, valid, idx)
    gathered_grads = gathered_grads.contiguous()
    valid = valid.to(device=grad_seq.device).contiguous()
    idx = idx.to(device=grad_seq.device, dtype=torch.long).contiguous()
    b, h, t, j_val, d = (int(v) for v in gathered_grads.shape)
    if t == 0 or j_val == 0:
        return grad_seq
    if not grad_seq.is_contiguous():
        return _q6_g128_accumulate_gathered_grads_vectorized(grad_seq, gathered_grads, valid, idx)
    total = b * h * t * j_val * d
    block = 256
    grid = (triton.cdiv(total, block),)
    _q6_g128_accumulate_gathered_grads_triton_kernel[grid](
        gathered_grads, valid, idx, grad_seq,
        TOTAL=total, B=b, H=h, N=int(grad_seq.shape[2]), T=t, J=j_val, D=d,
        BLOCK=block,
        num_warps=4,
    )
    return grad_seq


if _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE:
    @triton.jit
    def _q6_g128_stage_e_decode_token_rows(
        payload_ptr, scales_ptr, bh, token_idx, row_valid, ds, dim_valid,
        PAIR_GROUPS: tl.constexpr, PAYLOAD_BYTES: tl.constexpr, HD: tl.constexpr,
    ):
        safe_t = tl.maximum(token_idx, 0)
        pair = safe_t >> 1
        half = safe_t & 1
        val_in_pair = half[:, None] * HD + ds[None, :]
        word_idx = val_in_pair >> 2
        lane = val_in_pair & 3
        payload_base = ((bh * PAIR_GROUPS + pair[:, None]) * PAYLOAD_BYTES + word_idx * 3)
        load_mask = row_valid[:, None] & dim_valid[None, :]
        b0 = tl.load(payload_ptr + payload_base + 0, mask=load_mask, other=0).to(tl.uint32)
        b1 = tl.load(payload_ptr + payload_base + 1, mask=load_mask, other=0).to(tl.uint32)
        b2 = tl.load(payload_ptr + payload_base + 2, mask=load_mask, other=0).to(tl.uint32)
        word = b0 | (b1 << 8) | (b2 << 16)
        code = ((word >> (lane * 6)) & 0x3F).to(tl.int32)
        signed = tl.where(code >= 32, code - 64, code).to(tl.float32)
        scale = tl.load(scales_ptr + bh * PAIR_GROUPS + pair, mask=row_valid, other=0.0).to(tl.float32)
        return (signed * scale[:, None]).to(tl.bfloat16).to(tl.float32)


    @triton.jit
    def _q6_g128_dsqg_backward_fused_core_kernel(
        Q, K_PAYLOAD, K_SCALES, V_PAYLOAD, V_SCALES, OFFSETS,
        POS_BIAS, SCALE_EMBED, PHASE_GATE, PHASE_BASE, PHASE_GAIN,
        Y_PRE, Z_PRE, GRAD_OUT,
        GRAD_Q, GRAD_K, GRAD_V, GRAD_POS, GRAD_SCALE,
        GRAD_GATED_BASE, GRAD_GATED_GAIN, GRAD_Y, GRAD_Z,
        N: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, KV_GROUP: tl.constexpr, HD: tl.constexpr,
        PAIR_GROUPS: tl.constexpr, PAYLOAD_BYTES: tl.constexpr,
        START: tl.constexpr, TILE_TOKENS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
        J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr, J_PAD: tl.constexpr,
        R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr,
    ):
        bh = tl.program_id(0)
        block_t = tl.program_id(1)
        b = bh // H_Q
        h = bh % H_Q
        kv_h = h // KV_GROUP
        kv_bh = b * H_KV + kv_h
        ts = block_t * BLOCK_N + tl.arange(0, BLOCK_N)
        ns = START + ts
        nm = (ts < TILE_TOKENS) & (ns < N)
        ds = tl.arange(0, BLOCK_HD)
        dm = ds < HD
        js = tl.arange(0, J_PAD)
        sc = 1.0 / (HD ** 0.5)
        q = tl.load(Q + ((b * H_Q + h) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        go = tl.load(GRAD_OUT + ((b * H_Q + h) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        scores = tl.full([BLOCK_N, J_PAD], float('-inf'), tl.float32)

        for i in range(J_VAL):
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            kt = _q6_g128_stage_e_decode_token_rows(
                K_PAYLOAD, K_SCALES, kv_bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            se_i = tl.load(SCALE_EMBED + i * HD + ds, mask=dm, other=0.0).to(tl.float32)
            s = tl.sum(q * kt, axis=1) * sc
            s += tl.sum(q * se_i[None, :], axis=1) * sc
            s += tl.load(POS_BIAS + i * H_Q + h).to(tl.float32)
            s = tl.where(valid, s, float('-inf'))
            scores = tl.where((js == i)[None, :], s[:, None], scores)

        max_score = tl.max(scores, axis=1)
        all_invalid = max_score == float('-inf')
        safe_max = tl.where(all_invalid, 0.0, max_score)
        exp_scores = tl.exp2((scores - safe_max[:, None]) * 1.4426950408889634)
        exp_scores = tl.where((js < J_VAL)[None, :], exp_scores, 0.0)
        denom = tl.sum(exp_scores, axis=1)
        safe_denom = tl.where(denom > 0.0, denom, 1.0)
        probs = exp_scores / safe_denom[:, None]

        grad_probs = tl.zeros([BLOCK_N, J_PAD], tl.float32)
        for i in range(J_VAL):
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            vt = _q6_g128_stage_e_decode_token_rows(
                V_PAYLOAD, V_SCALES, kv_bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            vt_rot = vt
            if i >= J_SMALL_VAL:
                slot = i - J_SMALL_VAL
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = ds == ch_a
                    mask_b = ds == ch_b
                    y_r = tl.load(Y_PRE + ((b * H_Q + h) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z_r = tl.load(Z_PRE + ((b * H_Q + h) * N + tl.maximum(kp, 0)) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gate = tl.sigmoid(tl.load(PHASE_GATE + slot).to(tl.float32))
                    gb = tl.load(PHASE_BASE + (slot * H_Q + h) * R_PLANES_VAL + r).to(tl.float32) * gate
                    gg = tl.load(PHASE_GAIN + (slot * H_Q + h) * R_PLANES_VAL + r).to(tl.float32) * gate
                    theta = tl.where(valid, gb + gg * y_r * z_r, 0.0)
                    cos_t = tl.cos(theta)
                    sin_t = tl.sin(theta)
                    va = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    vb = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    vt_rot = tl.where(mask_a[None, :], (cos_t * va - sin_t * vb)[:, None], vt_rot)
                    vt_rot = tl.where(mask_b[None, :], (sin_t * va + cos_t * vb)[:, None], vt_rot)
            gp = tl.sum(go * vt_rot, axis=1)
            grad_probs = tl.where((js == i)[None, :], gp[:, None], grad_probs)

        prob_dot = tl.sum(grad_probs * probs, axis=1)
        dq = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

        for i in range(J_VAL):
            p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)
            gp_i = tl.sum(grad_probs * (js == i)[None, :].to(tl.float32), axis=1)
            grad_s = p_i * (gp_i - prob_dot)
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            grad_s = tl.where(valid, grad_s, 0.0)
            kt = _q6_g128_stage_e_decode_token_rows(
                K_PAYLOAD, K_SCALES, kv_bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            vt = _q6_g128_stage_e_decode_token_rows(
                V_PAYLOAD, V_SCALES, kv_bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            se_i = tl.load(SCALE_EMBED + i * HD + ds, mask=dm, other=0.0).to(tl.float32)
            dq += grad_s[:, None] * (kt + se_i[None, :]) * sc
            gk = grad_s[:, None] * q * sc
            tl.atomic_add(GRAD_K + ((b * H_KV + kv_h) * N + kp[:, None]) * HD + ds[None, :], gk, sem='relaxed', mask=valid[:, None] & dm[None, :])
            gs = tl.sum(grad_s[:, None] * q, axis=0) * sc
            tl.atomic_add(GRAD_SCALE + i * HD + ds, gs, sem='relaxed', mask=dm)
            tl.atomic_add(GRAD_POS + i * H_Q + h, tl.sum(grad_s, axis=0), sem='relaxed')

            gv_rot = p_i[:, None] * go
            gv = gv_rot
            if i >= J_SMALL_VAL:
                slot = i - J_SMALL_VAL
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = ds == ch_a
                    mask_b = ds == ch_b
                    y_r = tl.load(Y_PRE + ((b * H_Q + h) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z_r = tl.load(Z_PRE + ((b * H_Q + h) * N + tl.maximum(kp, 0)) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gate = tl.sigmoid(tl.load(PHASE_GATE + slot).to(tl.float32))
                    gb_raw = tl.load(PHASE_BASE + (slot * H_Q + h) * R_PLANES_VAL + r).to(tl.float32)
                    gg_raw = tl.load(PHASE_GAIN + (slot * H_Q + h) * R_PLANES_VAL + r).to(tl.float32)
                    gb = gb_raw * gate
                    gg = gg_raw * gate
                    theta = tl.where(valid, gb + gg * y_r * z_r, 0.0)
                    cos_t = tl.cos(theta)
                    sin_t = tl.sin(theta)
                    va = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    vb = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    go_a = tl.sum(gv_rot * mask_a[None, :].to(tl.float32), axis=1)
                    go_b = tl.sum(gv_rot * mask_b[None, :].to(tl.float32), axis=1)
                    gv_a = cos_t * go_a + sin_t * go_b
                    gv_b = -sin_t * go_a + cos_t * go_b
                    gv = tl.where(mask_a[None, :], gv_a[:, None], gv)
                    gv = tl.where(mask_b[None, :], gv_b[:, None], gv)
                    dtheta = ((-sin_t * va - cos_t * vb) * go_a + (cos_t * va - sin_t * vb) * go_b)
                    dtheta = tl.where(valid, dtheta, 0.0)
                    tl.atomic_add(GRAD_GATED_BASE + (slot * H_Q + h) * R_PLANES_VAL + r, tl.sum(dtheta, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_GATED_GAIN + (slot * H_Q + h) * R_PLANES_VAL + r, tl.sum(dtheta * y_r * z_r, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_Y + ((b * H_Q + h) * N + ns) * R_PLANES_VAL + r, dtheta * gg * z_r, sem='relaxed', mask=nm)
                    tl.atomic_add(GRAD_Z + ((b * H_Q + h) * N + kp) * R_PLANES_VAL + r, dtheta * gg * y_r, sem='relaxed', mask=valid)
            tl.atomic_add(GRAD_V + ((b * H_KV + kv_h) * N + kp[:, None]) * HD + ds[None, :], gv, sem='relaxed', mask=valid[:, None] & dm[None, :])


        tl.store(GRAD_Q + ((b * H_Q + h) * N + ns[:, None]) * HD + ds[None, :], dq, mask=nm[:, None] & dm[None, :])


    @triton.jit
    def _q6_g128_dsqg_backward_fused_core_pair_q_kernel(
        Q, K_PAYLOAD, K_SCALES, V_PAYLOAD, V_SCALES, OFFSETS,
        POS_BIAS, SCALE_EMBED, PHASE_GATE, PHASE_BASE, PHASE_GAIN,
        Y_PRE, Z_PRE, GRAD_OUT,
        GRAD_Q, GRAD_K, GRAD_V, GRAD_POS, GRAD_SCALE,
        GRAD_GATED_BASE, GRAD_GATED_GAIN, GRAD_Y, GRAD_Z,
        N: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, HD: tl.constexpr,
        PAIR_GROUPS: tl.constexpr, PAYLOAD_BYTES: tl.constexpr,
        START: tl.constexpr, TILE_TOKENS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
        J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr, J_PAD: tl.constexpr,
        R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr,
    ):
        """Stage-F.3 paired-query-head fused backward for KV_GROUP=2.

        One program computes the two query heads that share a KV head. K/V q6
        rows are decoded once per offset and K/V sequence gradients are summed
        before one atomic scatter into the Hkv-shaped buffers. Query-head-local
        Q, pos-bias, phase, y, and z gradients remain separate.
        """
        bh_kv = tl.program_id(0)
        block_t = tl.program_id(1)
        b = bh_kv // H_KV
        kv_h = bh_kv % H_KV
        h0 = kv_h * 2
        h1 = h0 + 1
        ts = block_t * BLOCK_N + tl.arange(0, BLOCK_N)
        ns = START + ts
        nm = (ts < TILE_TOKENS) & (ns < N)
        ds = tl.arange(0, BLOCK_HD)
        dm = ds < HD
        js = tl.arange(0, J_PAD)
        sc = 1.0 / (HD ** 0.5)

        q0 = tl.load(Q + ((b * H_Q + h0) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        q1 = tl.load(Q + ((b * H_Q + h1) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        go0 = tl.load(GRAD_OUT + ((b * H_Q + h0) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        go1 = tl.load(GRAD_OUT + ((b * H_Q + h1) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        scores0 = tl.full([BLOCK_N, J_PAD], float('-inf'), tl.float32)
        scores1 = tl.full([BLOCK_N, J_PAD], float('-inf'), tl.float32)

        for i in range(J_VAL):
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            kt = _q6_g128_stage_e_decode_token_rows(
                K_PAYLOAD, K_SCALES, bh_kv, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            se_i = tl.load(SCALE_EMBED + i * HD + ds, mask=dm, other=0.0).to(tl.float32)
            s0 = tl.sum(q0 * kt, axis=1) * sc
            s0 += tl.sum(q0 * se_i[None, :], axis=1) * sc
            s0 += tl.load(POS_BIAS + i * H_Q + h0).to(tl.float32)
            s0 = tl.where(valid, s0, float('-inf'))
            scores0 = tl.where((js == i)[None, :], s0[:, None], scores0)
            s1 = tl.sum(q1 * kt, axis=1) * sc
            s1 += tl.sum(q1 * se_i[None, :], axis=1) * sc
            s1 += tl.load(POS_BIAS + i * H_Q + h1).to(tl.float32)
            s1 = tl.where(valid, s1, float('-inf'))
            scores1 = tl.where((js == i)[None, :], s1[:, None], scores1)

        max0 = tl.max(scores0, axis=1)
        max1 = tl.max(scores1, axis=1)
        invalid0 = max0 == float('-inf')
        invalid1 = max1 == float('-inf')
        safe0 = tl.where(invalid0, 0.0, max0)
        safe1 = tl.where(invalid1, 0.0, max1)
        exp0 = tl.exp2((scores0 - safe0[:, None]) * 1.4426950408889634)
        exp1 = tl.exp2((scores1 - safe1[:, None]) * 1.4426950408889634)
        exp0 = tl.where((js < J_VAL)[None, :], exp0, 0.0)
        exp1 = tl.where((js < J_VAL)[None, :], exp1, 0.0)
        den0 = tl.sum(exp0, axis=1)
        den1 = tl.sum(exp1, axis=1)
        safe_den0 = tl.where(den0 > 0.0, den0, 1.0)
        safe_den1 = tl.where(den1 > 0.0, den1, 1.0)
        probs0 = exp0 / safe_den0[:, None]
        probs1 = exp1 / safe_den1[:, None]


        grad_probs0 = tl.zeros([BLOCK_N, J_PAD], tl.float32)
        grad_probs1 = tl.zeros([BLOCK_N, J_PAD], tl.float32)
        for i in range(J_VAL):
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            vt = _q6_g128_stage_e_decode_token_rows(
                V_PAYLOAD, V_SCALES, bh_kv, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            vt0 = vt
            vt1 = vt
            if i >= J_SMALL_VAL:
                slot = i - J_SMALL_VAL
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = ds == ch_a
                    mask_b = ds == ch_b
                    va = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    vb = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    z_idx = tl.maximum(kp, 0)
                    gate = tl.sigmoid(tl.load(PHASE_GATE + slot).to(tl.float32))

                    y0 = tl.load(Y_PRE + ((b * H_Q + h0) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z0 = tl.load(Z_PRE + ((b * H_Q + h0) * N + z_idx) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gb0 = tl.load(PHASE_BASE + (slot * H_Q + h0) * R_PLANES_VAL + r).to(tl.float32) * gate
                    gg0 = tl.load(PHASE_GAIN + (slot * H_Q + h0) * R_PLANES_VAL + r).to(tl.float32) * gate
                    th0 = tl.where(valid, gb0 + gg0 * y0 * z0, 0.0)
                    c0 = tl.cos(th0)
                    s0 = tl.sin(th0)
                    vt0 = tl.where(mask_a[None, :], (c0 * va - s0 * vb)[:, None], vt0)
                    vt0 = tl.where(mask_b[None, :], (s0 * va + c0 * vb)[:, None], vt0)

                    y1 = tl.load(Y_PRE + ((b * H_Q + h1) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z1 = tl.load(Z_PRE + ((b * H_Q + h1) * N + z_idx) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gb1 = tl.load(PHASE_BASE + (slot * H_Q + h1) * R_PLANES_VAL + r).to(tl.float32) * gate
                    gg1 = tl.load(PHASE_GAIN + (slot * H_Q + h1) * R_PLANES_VAL + r).to(tl.float32) * gate
                    th1 = tl.where(valid, gb1 + gg1 * y1 * z1, 0.0)
                    c1 = tl.cos(th1)
                    s1 = tl.sin(th1)
                    vt1 = tl.where(mask_a[None, :], (c1 * va - s1 * vb)[:, None], vt1)
                    vt1 = tl.where(mask_b[None, :], (s1 * va + c1 * vb)[:, None], vt1)
            gp0 = tl.sum(go0 * vt0, axis=1)
            gp1 = tl.sum(go1 * vt1, axis=1)
            grad_probs0 = tl.where((js == i)[None, :], gp0[:, None], grad_probs0)
            grad_probs1 = tl.where((js == i)[None, :], gp1[:, None], grad_probs1)

        prob_dot0 = tl.sum(grad_probs0 * probs0, axis=1)
        prob_dot1 = tl.sum(grad_probs1 * probs1, axis=1)
        dq0 = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)
        dq1 = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

        for i in range(J_VAL):
            p0 = tl.sum(probs0 * (js == i)[None, :].to(tl.float32), axis=1)
            p1 = tl.sum(probs1 * (js == i)[None, :].to(tl.float32), axis=1)
            gp0 = tl.sum(grad_probs0 * (js == i)[None, :].to(tl.float32), axis=1)
            gp1 = tl.sum(grad_probs1 * (js == i)[None, :].to(tl.float32), axis=1)
            gs0 = p0 * (gp0 - prob_dot0)
            gs1 = p1 * (gp1 - prob_dot1)
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            gs0 = tl.where(valid, gs0, 0.0)
            gs1 = tl.where(valid, gs1, 0.0)
            kt = _q6_g128_stage_e_decode_token_rows(
                K_PAYLOAD, K_SCALES, bh_kv, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            vt = _q6_g128_stage_e_decode_token_rows(
                V_PAYLOAD, V_SCALES, bh_kv, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            se_i = tl.load(SCALE_EMBED + i * HD + ds, mask=dm, other=0.0).to(tl.float32)
            dq0 += gs0[:, None] * (kt + se_i[None, :]) * sc
            dq1 += gs1[:, None] * (kt + se_i[None, :]) * sc
            gk = (gs0[:, None] * q0 + gs1[:, None] * q1) * sc
            tl.atomic_add(GRAD_K + ((b * H_KV + kv_h) * N + kp[:, None]) * HD + ds[None, :], gk, sem='relaxed', mask=valid[:, None] & dm[None, :])
            gscale = (tl.sum(gs0[:, None] * q0, axis=0) + tl.sum(gs1[:, None] * q1, axis=0)) * sc
            tl.atomic_add(GRAD_SCALE + i * HD + ds, gscale, sem='relaxed', mask=dm)
            tl.atomic_add(GRAD_POS + i * H_Q + h0, tl.sum(gs0, axis=0), sem='relaxed')
            tl.atomic_add(GRAD_POS + i * H_Q + h1, tl.sum(gs1, axis=0), sem='relaxed')

            gv_rot0 = p0[:, None] * go0
            gv_rot1 = p1[:, None] * go1
            gv0 = gv_rot0
            gv1 = gv_rot1
            if i >= J_SMALL_VAL:
                slot = i - J_SMALL_VAL
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = ds == ch_a
                    mask_b = ds == ch_b
                    va = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    vb = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    z_idx = tl.maximum(kp, 0)
                    gate = tl.sigmoid(tl.load(PHASE_GATE + slot).to(tl.float32))

                    y0 = tl.load(Y_PRE + ((b * H_Q + h0) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z0 = tl.load(Z_PRE + ((b * H_Q + h0) * N + z_idx) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gb0_raw = tl.load(PHASE_BASE + (slot * H_Q + h0) * R_PLANES_VAL + r).to(tl.float32)
                    gg0_raw = tl.load(PHASE_GAIN + (slot * H_Q + h0) * R_PLANES_VAL + r).to(tl.float32)
                    gb0 = gb0_raw * gate
                    gg0 = gg0_raw * gate
                    th0 = tl.where(valid, gb0 + gg0 * y0 * z0, 0.0)
                    c0 = tl.cos(th0)
                    s0 = tl.sin(th0)
                    go0a = tl.sum(gv_rot0 * mask_a[None, :].to(tl.float32), axis=1)
                    go0b = tl.sum(gv_rot0 * mask_b[None, :].to(tl.float32), axis=1)
                    gv0a = c0 * go0a + s0 * go0b
                    gv0b = -s0 * go0a + c0 * go0b
                    gv0 = tl.where(mask_a[None, :], gv0a[:, None], gv0)
                    gv0 = tl.where(mask_b[None, :], gv0b[:, None], gv0)
                    dth0 = ((-s0 * va - c0 * vb) * go0a + (c0 * va - s0 * vb) * go0b)
                    dth0 = tl.where(valid, dth0, 0.0)
                    tl.atomic_add(GRAD_GATED_BASE + (slot * H_Q + h0) * R_PLANES_VAL + r, tl.sum(dth0, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_GATED_GAIN + (slot * H_Q + h0) * R_PLANES_VAL + r, tl.sum(dth0 * y0 * z0, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_Y + ((b * H_Q + h0) * N + ns) * R_PLANES_VAL + r, dth0 * gg0 * z0, sem='relaxed', mask=nm)
                    tl.atomic_add(GRAD_Z + ((b * H_Q + h0) * N + kp) * R_PLANES_VAL + r, dth0 * gg0 * y0, sem='relaxed', mask=valid)

                    y1 = tl.load(Y_PRE + ((b * H_Q + h1) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z1 = tl.load(Z_PRE + ((b * H_Q + h1) * N + z_idx) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gb1_raw = tl.load(PHASE_BASE + (slot * H_Q + h1) * R_PLANES_VAL + r).to(tl.float32)
                    gg1_raw = tl.load(PHASE_GAIN + (slot * H_Q + h1) * R_PLANES_VAL + r).to(tl.float32)
                    gb1 = gb1_raw * gate
                    gg1 = gg1_raw * gate
                    th1 = tl.where(valid, gb1 + gg1 * y1 * z1, 0.0)
                    c1 = tl.cos(th1)
                    s1 = tl.sin(th1)
                    go1a = tl.sum(gv_rot1 * mask_a[None, :].to(tl.float32), axis=1)
                    go1b = tl.sum(gv_rot1 * mask_b[None, :].to(tl.float32), axis=1)
                    gv1a = c1 * go1a + s1 * go1b
                    gv1b = -s1 * go1a + c1 * go1b
                    gv1 = tl.where(mask_a[None, :], gv1a[:, None], gv1)
                    gv1 = tl.where(mask_b[None, :], gv1b[:, None], gv1)
                    dth1 = ((-s1 * va - c1 * vb) * go1a + (c1 * va - s1 * vb) * go1b)
                    dth1 = tl.where(valid, dth1, 0.0)
                    tl.atomic_add(GRAD_GATED_BASE + (slot * H_Q + h1) * R_PLANES_VAL + r, tl.sum(dth1, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_GATED_GAIN + (slot * H_Q + h1) * R_PLANES_VAL + r, tl.sum(dth1 * y1 * z1, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_Y + ((b * H_Q + h1) * N + ns) * R_PLANES_VAL + r, dth1 * gg1 * z1, sem='relaxed', mask=nm)
                    tl.atomic_add(GRAD_Z + ((b * H_Q + h1) * N + kp) * R_PLANES_VAL + r, dth1 * gg1 * y1, sem='relaxed', mask=valid)
            tl.atomic_add(GRAD_V + ((b * H_KV + kv_h) * N + kp[:, None]) * HD + ds[None, :], gv0 + gv1, sem='relaxed', mask=valid[:, None] & dm[None, :])

        tl.store(GRAD_Q + ((b * H_Q + h0) * N + ns[:, None]) * HD + ds[None, :], dq0, mask=nm[:, None] & dm[None, :])
        tl.store(GRAD_Q + ((b * H_Q + h1) * N + ns[:, None]) * HD + ds[None, :], dq1, mask=nm[:, None] & dm[None, :])


    @triton.jit
    def _q6_g128_dsqg_backward_fused_core_pair_q_sparse_movt_kernel(
        Q, K_PAYLOAD, K_SCALES, V_PAYLOAD, V_SCALES, OFFSETS,
        POS_BIAS, SCALE_EMBED, PHASE_GATE, PHASE_BASE, PHASE_GAIN,
        Y_PRE, Z_PRE, GRAD_OUT,
        GRAD_Q, GRAD_K, GRAD_V, GRAD_POS, GRAD_SCALE,
        GRAD_GATED_BASE, GRAD_GATED_GAIN, GRAD_Y, GRAD_Z,
        PROB_SCRATCH,
        N: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, HD: tl.constexpr,
        PAIR_GROUPS: tl.constexpr, PAYLOAD_BYTES: tl.constexpr,
        START: tl.constexpr, TILE_TOKENS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
        J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr, J_PAD: tl.constexpr,
        R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr,
        SCRATCH_BLOCKS: tl.constexpr,
        SPLIT_MOVT_CORR: tl.constexpr,
        SPLIT_VPHASE: tl.constexpr,
        STORE_PROB_SCRATCH: tl.constexpr,
    ):
        """Stage-F.3 paired-query-head fused backward with sparse MOVT corrections.

        One program computes the two query heads that share a KV head. K/V q6
        rows are decoded once per offset and K/V sequence gradients are summed
        before one atomic scatter into the Hkv-shaped buffers. Query-head-local
        Q, pos-bias, phase, y, and z gradients remain separate.
        """
        bh_kv = tl.program_id(0)
        block_t = tl.program_id(1)
        b = bh_kv // H_KV
        kv_h = bh_kv % H_KV
        h0 = kv_h * 2
        h1 = h0 + 1
        ts = block_t * BLOCK_N + tl.arange(0, BLOCK_N)
        ns = START + ts
        nm = (ts < TILE_TOKENS) & (ns < N)
        ds = tl.arange(0, BLOCK_HD)
        dm = ds < HD
        js = tl.arange(0, J_PAD)
        sc = 1.0 / (HD ** 0.5)

        q0 = tl.load(Q + ((b * H_Q + h0) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        q1 = tl.load(Q + ((b * H_Q + h1) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        go0 = tl.load(GRAD_OUT + ((b * H_Q + h0) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        go1 = tl.load(GRAD_OUT + ((b * H_Q + h1) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        scores0 = tl.full([BLOCK_N, J_PAD], float('-inf'), tl.float32)
        scores1 = tl.full([BLOCK_N, J_PAD], float('-inf'), tl.float32)

        for i in range(J_VAL):
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            kt = _q6_g128_stage_e_decode_token_rows(
                K_PAYLOAD, K_SCALES, bh_kv, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            se_i = tl.load(SCALE_EMBED + i * HD + ds, mask=dm, other=0.0).to(tl.float32)
            s0 = tl.sum(q0 * kt, axis=1) * sc
            s0 += tl.sum(q0 * se_i[None, :], axis=1) * sc
            s0 += tl.load(POS_BIAS + i * H_Q + h0).to(tl.float32)
            s0 = tl.where(valid, s0, float('-inf'))
            scores0 = tl.where((js == i)[None, :], s0[:, None], scores0)
            s1 = tl.sum(q1 * kt, axis=1) * sc
            s1 += tl.sum(q1 * se_i[None, :], axis=1) * sc
            s1 += tl.load(POS_BIAS + i * H_Q + h1).to(tl.float32)
            s1 = tl.where(valid, s1, float('-inf'))
            scores1 = tl.where((js == i)[None, :], s1[:, None], scores1)

        max0 = tl.max(scores0, axis=1)
        max1 = tl.max(scores1, axis=1)
        invalid0 = max0 == float('-inf')
        invalid1 = max1 == float('-inf')
        safe0 = tl.where(invalid0, 0.0, max0)
        safe1 = tl.where(invalid1, 0.0, max1)
        exp0 = tl.exp2((scores0 - safe0[:, None]) * 1.4426950408889634)
        exp1 = tl.exp2((scores1 - safe1[:, None]) * 1.4426950408889634)
        exp0 = tl.where((js < J_VAL)[None, :], exp0, 0.0)
        exp1 = tl.where((js < J_VAL)[None, :], exp1, 0.0)
        den0 = tl.sum(exp0, axis=1)
        den1 = tl.sum(exp1, axis=1)
        safe_den0 = tl.where(den0 > 0.0, den0, 1.0)
        safe_den1 = tl.where(den1 > 0.0, den1, 1.0)
        probs0 = exp0 / safe_den0[:, None]
        probs1 = exp1 / safe_den1[:, None]

        if STORE_PROB_SCRATCH:
            for i in range(J_SMALL_VAL, J_VAL):
                slot = i - J_SMALL_VAL
                p0_s = tl.sum(probs0 * (js == i)[None, :].to(tl.float32), axis=1)
                p1_s = tl.sum(probs1 * (js == i)[None, :].to(tl.float32), axis=1)
                scratch_base = (((bh_kv * SCRATCH_BLOCKS + block_t) * J_LARGE_VAL + slot) * 2) * BLOCK_N + ts
                tl.store(PROB_SCRATCH + scratch_base, p0_s, mask=nm)
                tl.store(PROB_SCRATCH + scratch_base + BLOCK_N, p1_s, mask=nm)

        grad_probs0 = tl.zeros([BLOCK_N, J_PAD], tl.float32)
        grad_probs1 = tl.zeros([BLOCK_N, J_PAD], tl.float32)
        for i in range(J_VAL):
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            vt = _q6_g128_stage_e_decode_token_rows(
                V_PAYLOAD, V_SCALES, bh_kv, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            gp0 = tl.sum(go0 * vt, axis=1)
            gp1 = tl.sum(go1 * vt, axis=1)
            if i >= J_SMALL_VAL:
                slot = i - J_SMALL_VAL
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = ds == ch_a
                    mask_b = ds == ch_b
                    va = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    vb = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    go0a_raw = tl.sum(go0 * mask_a[None, :].to(tl.float32), axis=1)
                    go0b_raw = tl.sum(go0 * mask_b[None, :].to(tl.float32), axis=1)
                    go1a_raw = tl.sum(go1 * mask_a[None, :].to(tl.float32), axis=1)
                    go1b_raw = tl.sum(go1 * mask_b[None, :].to(tl.float32), axis=1)
                    z_idx = tl.maximum(kp, 0)
                    gate = tl.sigmoid(tl.load(PHASE_GATE + slot).to(tl.float32))

                    y0 = tl.load(Y_PRE + ((b * H_Q + h0) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z0 = tl.load(Z_PRE + ((b * H_Q + h0) * N + z_idx) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gb0 = tl.load(PHASE_BASE + (slot * H_Q + h0) * R_PLANES_VAL + r).to(tl.float32) * gate
                    gg0 = tl.load(PHASE_GAIN + (slot * H_Q + h0) * R_PLANES_VAL + r).to(tl.float32) * gate
                    th0 = tl.where(valid, gb0 + gg0 * y0 * z0, 0.0)
                    c0 = tl.cos(th0)
                    s0 = tl.sin(th0)
                    rot0a = c0 * va - s0 * vb
                    rot0b = s0 * va + c0 * vb
                    gp0 += go0a_raw * (rot0a - va) + go0b_raw * (rot0b - vb)

                    y1 = tl.load(Y_PRE + ((b * H_Q + h1) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z1 = tl.load(Z_PRE + ((b * H_Q + h1) * N + z_idx) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gb1 = tl.load(PHASE_BASE + (slot * H_Q + h1) * R_PLANES_VAL + r).to(tl.float32) * gate
                    gg1 = tl.load(PHASE_GAIN + (slot * H_Q + h1) * R_PLANES_VAL + r).to(tl.float32) * gate
                    th1 = tl.where(valid, gb1 + gg1 * y1 * z1, 0.0)
                    c1 = tl.cos(th1)
                    s1 = tl.sin(th1)
                    rot1a = c1 * va - s1 * vb
                    rot1b = s1 * va + c1 * vb
                    gp1 += go1a_raw * (rot1a - va) + go1b_raw * (rot1b - vb)
            grad_probs0 = tl.where((js == i)[None, :], gp0[:, None], grad_probs0)
            grad_probs1 = tl.where((js == i)[None, :], gp1[:, None], grad_probs1)

        prob_dot0 = tl.sum(grad_probs0 * probs0, axis=1)
        prob_dot1 = tl.sum(grad_probs1 * probs1, axis=1)
        dq0 = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)
        dq1 = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

        for i in range(J_VAL):
            p0 = tl.sum(probs0 * (js == i)[None, :].to(tl.float32), axis=1)
            p1 = tl.sum(probs1 * (js == i)[None, :].to(tl.float32), axis=1)
            gp0 = tl.sum(grad_probs0 * (js == i)[None, :].to(tl.float32), axis=1)
            gp1 = tl.sum(grad_probs1 * (js == i)[None, :].to(tl.float32), axis=1)
            gs0 = p0 * (gp0 - prob_dot0)
            gs1 = p1 * (gp1 - prob_dot1)
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            gs0 = tl.where(valid, gs0, 0.0)
            gs1 = tl.where(valid, gs1, 0.0)
            kt = _q6_g128_stage_e_decode_token_rows(
                K_PAYLOAD, K_SCALES, bh_kv, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            vt = _q6_g128_stage_e_decode_token_rows(
                V_PAYLOAD, V_SCALES, bh_kv, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            se_i = tl.load(SCALE_EMBED + i * HD + ds, mask=dm, other=0.0).to(tl.float32)
            dq0 += gs0[:, None] * (kt + se_i[None, :]) * sc
            dq1 += gs1[:, None] * (kt + se_i[None, :]) * sc
            gk = (gs0[:, None] * q0 + gs1[:, None] * q1) * sc
            tl.atomic_add(GRAD_K + ((b * H_KV + kv_h) * N + kp[:, None]) * HD + ds[None, :], gk, sem='relaxed', mask=valid[:, None] & dm[None, :])
            gscale = (tl.sum(gs0[:, None] * q0, axis=0) + tl.sum(gs1[:, None] * q1, axis=0)) * sc
            tl.atomic_add(GRAD_SCALE + i * HD + ds, gscale, sem='relaxed', mask=dm)
            tl.atomic_add(GRAD_POS + i * H_Q + h0, tl.sum(gs0, axis=0), sem='relaxed')
            tl.atomic_add(GRAD_POS + i * H_Q + h1, tl.sum(gs1, axis=0), sem='relaxed')

            if not SPLIT_VPHASE:
                gv_base = p0[:, None] * go0 + p1[:, None] * go1
                tl.atomic_add(GRAD_V + ((b * H_KV + kv_h) * N + kp[:, None]) * HD + ds[None, :], gv_base, sem='relaxed', mask=valid[:, None] & dm[None, :])
            if (not SPLIT_MOVT_CORR) and i >= J_SMALL_VAL:
                slot = i - J_SMALL_VAL
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = ds == ch_a
                    mask_b = ds == ch_b
                    va = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    vb = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    z_idx = tl.maximum(kp, 0)
                    gate = tl.sigmoid(tl.load(PHASE_GATE + slot).to(tl.float32))

                    y0 = tl.load(Y_PRE + ((b * H_Q + h0) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z0 = tl.load(Z_PRE + ((b * H_Q + h0) * N + z_idx) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gb0_raw = tl.load(PHASE_BASE + (slot * H_Q + h0) * R_PLANES_VAL + r).to(tl.float32)
                    gg0_raw = tl.load(PHASE_GAIN + (slot * H_Q + h0) * R_PLANES_VAL + r).to(tl.float32)
                    gb0 = gb0_raw * gate
                    gg0 = gg0_raw * gate
                    th0 = tl.where(valid, gb0 + gg0 * y0 * z0, 0.0)
                    c0 = tl.cos(th0)
                    s0 = tl.sin(th0)
                    go0a_raw = tl.sum(go0 * mask_a[None, :].to(tl.float32), axis=1)
                    go0b_raw = tl.sum(go0 * mask_b[None, :].to(tl.float32), axis=1)
                    go0a = p0 * go0a_raw
                    go0b = p0 * go0b_raw
                    gv0a = c0 * go0a + s0 * go0b
                    gv0b = -s0 * go0a + c0 * go0b
                    dth0 = ((-s0 * va - c0 * vb) * go0a + (c0 * va - s0 * vb) * go0b)
                    dth0 = tl.where(valid, dth0, 0.0)
                    tl.atomic_add(GRAD_GATED_BASE + (slot * H_Q + h0) * R_PLANES_VAL + r, tl.sum(dth0, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_GATED_GAIN + (slot * H_Q + h0) * R_PLANES_VAL + r, tl.sum(dth0 * y0 * z0, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_Y + ((b * H_Q + h0) * N + ns) * R_PLANES_VAL + r, dth0 * gg0 * z0, sem='relaxed', mask=nm)
                    tl.atomic_add(GRAD_Z + ((b * H_Q + h0) * N + kp) * R_PLANES_VAL + r, dth0 * gg0 * y0, sem='relaxed', mask=valid)

                    y1 = tl.load(Y_PRE + ((b * H_Q + h1) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z1 = tl.load(Z_PRE + ((b * H_Q + h1) * N + z_idx) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gb1_raw = tl.load(PHASE_BASE + (slot * H_Q + h1) * R_PLANES_VAL + r).to(tl.float32)
                    gg1_raw = tl.load(PHASE_GAIN + (slot * H_Q + h1) * R_PLANES_VAL + r).to(tl.float32)
                    gb1 = gb1_raw * gate
                    gg1 = gg1_raw * gate
                    th1 = tl.where(valid, gb1 + gg1 * y1 * z1, 0.0)
                    c1 = tl.cos(th1)
                    s1 = tl.sin(th1)
                    go1a_raw = tl.sum(go1 * mask_a[None, :].to(tl.float32), axis=1)
                    go1b_raw = tl.sum(go1 * mask_b[None, :].to(tl.float32), axis=1)
                    go1a = p1 * go1a_raw
                    go1b = p1 * go1b_raw
                    gv1a = c1 * go1a + s1 * go1b
                    gv1b = -s1 * go1a + c1 * go1b
                    dth1 = ((-s1 * va - c1 * vb) * go1a + (c1 * va - s1 * vb) * go1b)
                    dth1 = tl.where(valid, dth1, 0.0)
                    tl.atomic_add(GRAD_GATED_BASE + (slot * H_Q + h1) * R_PLANES_VAL + r, tl.sum(dth1, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_GATED_GAIN + (slot * H_Q + h1) * R_PLANES_VAL + r, tl.sum(dth1 * y1 * z1, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_Y + ((b * H_Q + h1) * N + ns) * R_PLANES_VAL + r, dth1 * gg1 * z1, sem='relaxed', mask=nm)
                    tl.atomic_add(GRAD_Z + ((b * H_Q + h1) * N + kp) * R_PLANES_VAL + r, dth1 * gg1 * y1, sem='relaxed', mask=valid)

                    corr_a = (gv0a - go0a) + (gv1a - go1a)
                    corr_b = (gv0b - go0b) + (gv1b - go1b)
                    tl.atomic_add(GRAD_V + ((b * H_KV + kv_h) * N + kp) * HD + ch_a, corr_a, sem='relaxed', mask=valid)
                    tl.atomic_add(GRAD_V + ((b * H_KV + kv_h) * N + kp) * HD + ch_b, corr_b, sem='relaxed', mask=valid)

        tl.store(GRAD_Q + ((b * H_Q + h0) * N + ns[:, None]) * HD + ds[None, :], dq0, mask=nm[:, None] & dm[None, :])
        tl.store(GRAD_Q + ((b * H_Q + h1) * N + ns[:, None]) * HD + ds[None, :], dq1, mask=nm[:, None] & dm[None, :])


    @triton.jit
    def _q6_g128_dsqg_backward_fused_core_pair_q_movt_corr_sidecar_kernel(
        Q, K_PAYLOAD, K_SCALES, V_PAYLOAD, V_SCALES, OFFSETS,
        POS_BIAS, SCALE_EMBED, PHASE_GATE, PHASE_BASE, PHASE_GAIN,
        Y_PRE, Z_PRE, GRAD_OUT,
        GRAD_V, GRAD_GATED_BASE, GRAD_GATED_GAIN, GRAD_Y, GRAD_Z,
        PROB_SCRATCH,
        N: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr, HD: tl.constexpr,
        PAIR_GROUPS: tl.constexpr, PAYLOAD_BYTES: tl.constexpr,
        START: tl.constexpr, TILE_TOKENS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
        J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr, J_PAD: tl.constexpr,
        R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr,
        SCRATCH_BLOCKS: tl.constexpr,
        INCLUDE_BASE_V: tl.constexpr,
        USE_PROB_SCRATCH: tl.constexpr,
        SIDECAR_PAIR_DIRECT: tl.constexpr,
    ):
        """Stage-F.4 sidecar for sparse MOVT value/phase corrections.

        The primary sparse-MOVT kernel computes score/prob replay, K/Q/score
        gradients, and the dense/base V scatter.  This specialist recomputes the
        score softmax once, then touches only the 2*R MOVT channels for sparse
        offsets and emits the V correction plus phase/y/z reductions.  It is a
        deliberately different live-state schedule: no grad_probs, no dq/gk, no
        full-width V gradient state in the MOVT correction path.
        """
        bh_kv = tl.program_id(0)
        block_t = tl.program_id(1)
        b = bh_kv // H_KV
        kv_h = bh_kv % H_KV
        h0 = kv_h * 2
        h1 = h0 + 1
        ts = block_t * BLOCK_N + tl.arange(0, BLOCK_N)
        ns = START + ts
        nm = (ts < TILE_TOKENS) & (ns < N)
        ds = tl.arange(0, BLOCK_HD)
        dm = ds < HD
        js = tl.arange(0, J_PAD)
        sc = 1.0 / (HD ** 0.5)

        if not USE_PROB_SCRATCH:
            q0 = tl.load(Q + ((b * H_Q + h0) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
            q1 = tl.load(Q + ((b * H_Q + h1) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
            scores0 = tl.full([BLOCK_N, J_PAD], float('-inf'), tl.float32)
            scores1 = tl.full([BLOCK_N, J_PAD], float('-inf'), tl.float32)

            for i in range(J_VAL):
                delta = tl.load(OFFSETS + i).to(tl.int32)
                kp = ns.to(tl.int32) - delta
                valid = nm & (kp >= 0) & (kp < N)
                kt = _q6_g128_stage_e_decode_token_rows(
                    K_PAYLOAD, K_SCALES, bh_kv, kp, valid, ds, dm,
                    PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
                )
                se_i = tl.load(SCALE_EMBED + i * HD + ds, mask=dm, other=0.0).to(tl.float32)
                s0 = tl.sum(q0 * kt, axis=1) * sc
                s0 += tl.sum(q0 * se_i[None, :], axis=1) * sc
                s0 += tl.load(POS_BIAS + i * H_Q + h0).to(tl.float32)
                s0 = tl.where(valid, s0, float('-inf'))
                scores0 = tl.where((js == i)[None, :], s0[:, None], scores0)
                s1 = tl.sum(q1 * kt, axis=1) * sc
                s1 += tl.sum(q1 * se_i[None, :], axis=1) * sc
                s1 += tl.load(POS_BIAS + i * H_Q + h1).to(tl.float32)
                s1 = tl.where(valid, s1, float('-inf'))
                scores1 = tl.where((js == i)[None, :], s1[:, None], scores1)

            max0 = tl.max(scores0, axis=1)
            max1 = tl.max(scores1, axis=1)
            safe0 = tl.where(max0 == float('-inf'), 0.0, max0)
            safe1 = tl.where(max1 == float('-inf'), 0.0, max1)
            exp0 = tl.exp2((scores0 - safe0[:, None]) * 1.4426950408889634)
            exp1 = tl.exp2((scores1 - safe1[:, None]) * 1.4426950408889634)
            exp0 = tl.where((js < J_VAL)[None, :], exp0, 0.0)
            exp1 = tl.where((js < J_VAL)[None, :], exp1, 0.0)
            den0 = tl.sum(exp0, axis=1)
            den1 = tl.sum(exp1, axis=1)
            probs0 = exp0 / tl.where(den0 > 0.0, den0, 1.0)[:, None]
            probs1 = exp1 / tl.where(den1 > 0.0, den1, 1.0)[:, None]

        if INCLUDE_BASE_V:
            go0 = tl.load(GRAD_OUT + ((b * H_Q + h0) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
            go1 = tl.load(GRAD_OUT + ((b * H_Q + h1) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
            for i in range(J_VAL):
                p0 = tl.sum(probs0 * (js == i)[None, :].to(tl.float32), axis=1)
                p1 = tl.sum(probs1 * (js == i)[None, :].to(tl.float32), axis=1)
                delta = tl.load(OFFSETS + i).to(tl.int32)
                kp = ns.to(tl.int32) - delta
                valid = nm & (kp >= 0) & (kp < N)
                gv_base = p0[:, None] * go0 + p1[:, None] * go1
                tl.atomic_add(GRAD_V + ((b * H_KV + kv_h) * N + kp[:, None]) * HD + ds[None, :], gv_base, sem='relaxed', mask=valid[:, None] & dm[None, :])

        pair_ds0 = tl.arange(0, 2)

        for i in range(J_SMALL_VAL, J_VAL):
            slot = i - J_SMALL_VAL
            if USE_PROB_SCRATCH:
                scratch_base = (((bh_kv * SCRATCH_BLOCKS + block_t) * J_LARGE_VAL + slot) * 2) * BLOCK_N + ts
                p0 = tl.load(PROB_SCRATCH + scratch_base, mask=nm, other=0.0).to(tl.float32)
                p1 = tl.load(PROB_SCRATCH + scratch_base + BLOCK_N, mask=nm, other=0.0).to(tl.float32)
            else:
                p0 = tl.sum(probs0 * (js == i)[None, :].to(tl.float32), axis=1)
                p1 = tl.sum(probs1 * (js == i)[None, :].to(tl.float32), axis=1)
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            z_idx = tl.maximum(kp, 0)
            gate = tl.sigmoid(tl.load(PHASE_GATE + slot).to(tl.float32))
            for r in range(R_PLANES_VAL):
                ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                ch_b = ch_a + 1
                pair_ds = ch_a + pair_ds0
                vt_pair = _q6_g128_stage_e_decode_token_rows(
                    V_PAYLOAD, V_SCALES, bh_kv, kp, valid, pair_ds, pair_ds < HD,
                    PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
                )
                va = tl.sum(vt_pair * (pair_ds0 == 0)[None, :].to(tl.float32), axis=1)
                vb = tl.sum(vt_pair * (pair_ds0 == 1)[None, :].to(tl.float32), axis=1)

                go0a_raw = tl.load(GRAD_OUT + ((b * H_Q + h0) * N + ns) * HD + ch_a, mask=nm, other=0.0).to(tl.float32)
                go0b_raw = tl.load(GRAD_OUT + ((b * H_Q + h0) * N + ns) * HD + ch_b, mask=nm, other=0.0).to(tl.float32)
                go1a_raw = tl.load(GRAD_OUT + ((b * H_Q + h1) * N + ns) * HD + ch_a, mask=nm, other=0.0).to(tl.float32)
                go1b_raw = tl.load(GRAD_OUT + ((b * H_Q + h1) * N + ns) * HD + ch_b, mask=nm, other=0.0).to(tl.float32)

                y0 = tl.load(Y_PRE + ((b * H_Q + h0) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                z0 = tl.load(Z_PRE + ((b * H_Q + h0) * N + z_idx) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                gb0_raw = tl.load(PHASE_BASE + (slot * H_Q + h0) * R_PLANES_VAL + r).to(tl.float32)
                gg0_raw = tl.load(PHASE_GAIN + (slot * H_Q + h0) * R_PLANES_VAL + r).to(tl.float32)
                gg0 = gg0_raw * gate
                th0 = tl.where(valid, gb0_raw * gate + gg0 * y0 * z0, 0.0)
                c0 = tl.cos(th0)
                s0 = tl.sin(th0)
                go0a = p0 * go0a_raw
                go0b = p0 * go0b_raw
                gv0a = c0 * go0a + s0 * go0b
                gv0b = -s0 * go0a + c0 * go0b
                dth0 = ((-s0 * va - c0 * vb) * go0a + (c0 * va - s0 * vb) * go0b)
                dth0 = tl.where(valid, dth0, 0.0)
                tl.atomic_add(GRAD_GATED_BASE + (slot * H_Q + h0) * R_PLANES_VAL + r, tl.sum(dth0, axis=0), sem='relaxed')
                tl.atomic_add(GRAD_GATED_GAIN + (slot * H_Q + h0) * R_PLANES_VAL + r, tl.sum(dth0 * y0 * z0, axis=0), sem='relaxed')
                tl.atomic_add(GRAD_Y + ((b * H_Q + h0) * N + ns) * R_PLANES_VAL + r, dth0 * gg0 * z0, sem='relaxed', mask=nm)
                tl.atomic_add(GRAD_Z + ((b * H_Q + h0) * N + kp) * R_PLANES_VAL + r, dth0 * gg0 * y0, sem='relaxed', mask=valid)

                y1 = tl.load(Y_PRE + ((b * H_Q + h1) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                z1 = tl.load(Z_PRE + ((b * H_Q + h1) * N + z_idx) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                gb1_raw = tl.load(PHASE_BASE + (slot * H_Q + h1) * R_PLANES_VAL + r).to(tl.float32)
                gg1_raw = tl.load(PHASE_GAIN + (slot * H_Q + h1) * R_PLANES_VAL + r).to(tl.float32)
                gg1 = gg1_raw * gate
                th1 = tl.where(valid, gb1_raw * gate + gg1 * y1 * z1, 0.0)
                c1 = tl.cos(th1)
                s1 = tl.sin(th1)
                go1a = p1 * go1a_raw
                go1b = p1 * go1b_raw
                gv1a = c1 * go1a + s1 * go1b
                gv1b = -s1 * go1a + c1 * go1b
                dth1 = ((-s1 * va - c1 * vb) * go1a + (c1 * va - s1 * vb) * go1b)
                dth1 = tl.where(valid, dth1, 0.0)
                tl.atomic_add(GRAD_GATED_BASE + (slot * H_Q + h1) * R_PLANES_VAL + r, tl.sum(dth1, axis=0), sem='relaxed')
                tl.atomic_add(GRAD_GATED_GAIN + (slot * H_Q + h1) * R_PLANES_VAL + r, tl.sum(dth1 * y1 * z1, axis=0), sem='relaxed')
                tl.atomic_add(GRAD_Y + ((b * H_Q + h1) * N + ns) * R_PLANES_VAL + r, dth1 * gg1 * z1, sem='relaxed', mask=nm)
                tl.atomic_add(GRAD_Z + ((b * H_Q + h1) * N + kp) * R_PLANES_VAL + r, dth1 * gg1 * y1, sem='relaxed', mask=valid)

                corr_a = (gv0a - go0a) + (gv1a - go1a)
                corr_b = (gv0b - go0b) + (gv1b - go1b)
                if SIDECAR_PAIR_DIRECT:
                    corr_pair = tl.where((pair_ds0 == 0)[None, :], corr_a[:, None], corr_b[:, None])
                    tl.atomic_add(
                        GRAD_V + ((b * H_KV + kv_h) * N + kp[:, None]) * HD + pair_ds[None, :],
                        corr_pair, sem='relaxed', mask=valid[:, None] & (pair_ds < HD)[None, :]
                    )
                else:
                    tl.atomic_add(GRAD_V + ((b * H_KV + kv_h) * N + kp) * HD + ch_a, corr_a, sem='relaxed', mask=valid)
                    tl.atomic_add(GRAD_V + ((b * H_KV + kv_h) * N + kp) * HD + ch_b, corr_b, sem='relaxed', mask=valid)


    @triton.jit
    def _q6_g128_dsqg_backward_fused_core_large_only_kernel(
        Q, K_PAYLOAD, K_SCALES, V_PAYLOAD, V_SCALES, OFFSETS,
        POS_BIAS, SCALE_EMBED, PHASE_GATE, PHASE_BASE, PHASE_GAIN,
        Y_PRE, Z_PRE, GRAD_OUT,
        GRAD_Q, GRAD_K, GRAD_V, GRAD_POS, GRAD_SCALE,
        GRAD_GATED_BASE, GRAD_GATED_GAIN, GRAD_Y, GRAD_Z,
        N: tl.constexpr, H: tl.constexpr, HD: tl.constexpr,
        PAIR_GROUPS: tl.constexpr, PAYLOAD_BYTES: tl.constexpr,
        START: tl.constexpr, TILE_TOKENS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
        J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr, J_PAD: tl.constexpr,
        R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr,
    ):
        """Stage-E group-specialized large-offset-only core for GROUP_B/C (J_SMALL=0)."""
        bh = tl.program_id(0)
        block_t = tl.program_id(1)
        b = bh // H
        h = bh % H
        ts = block_t * BLOCK_N + tl.arange(0, BLOCK_N)
        ns = START + ts
        nm = (ts < TILE_TOKENS) & (ns < N)
        ds = tl.arange(0, BLOCK_HD)
        dm = ds < HD
        js = tl.arange(0, J_PAD)
        sc = 1.0 / (HD ** 0.5)
        q = tl.load(Q + ((b * H + h) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        go = tl.load(GRAD_OUT + ((b * H + h) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        scores = tl.full([BLOCK_N, J_PAD], float('-inf'), tl.float32)

        for i in range(J_VAL):
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            kt = _q6_g128_stage_e_decode_token_rows(
                K_PAYLOAD, K_SCALES, bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            se_i = tl.load(SCALE_EMBED + i * HD + ds, mask=dm, other=0.0).to(tl.float32)
            s = tl.sum(q * kt, axis=1) * sc
            s += tl.sum(q * se_i[None, :], axis=1) * sc
            s += tl.load(POS_BIAS + i * H + h).to(tl.float32)
            s = tl.where(valid, s, float('-inf'))
            scores = tl.where((js == i)[None, :], s[:, None], scores)

        max_score = tl.max(scores, axis=1)
        all_invalid = max_score == float('-inf')
        safe_max = tl.where(all_invalid, 0.0, max_score)
        exp_scores = tl.exp2((scores - safe_max[:, None]) * 1.4426950408889634)
        exp_scores = tl.where((js < J_VAL)[None, :], exp_scores, 0.0)
        denom = tl.sum(exp_scores, axis=1)
        safe_denom = tl.where(denom > 0.0, denom, 1.0)
        probs = exp_scores / safe_denom[:, None]

        grad_probs = tl.zeros([BLOCK_N, J_PAD], tl.float32)
        for i in range(J_VAL):
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            vt = _q6_g128_stage_e_decode_token_rows(
                V_PAYLOAD, V_SCALES, bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            vt_rot = vt
            if True:
                slot = i
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = ds == ch_a
                    mask_b = ds == ch_b
                    y_r = tl.load(Y_PRE + ((b * H + h) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z_r = tl.load(Z_PRE + ((b * H + h) * N + tl.maximum(kp, 0)) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gate = tl.sigmoid(tl.load(PHASE_GATE + slot).to(tl.float32))
                    gb = tl.load(PHASE_BASE + (slot * H + h) * R_PLANES_VAL + r).to(tl.float32) * gate
                    gg = tl.load(PHASE_GAIN + (slot * H + h) * R_PLANES_VAL + r).to(tl.float32) * gate
                    theta = tl.where(valid, gb + gg * y_r * z_r, 0.0)
                    cos_t = tl.cos(theta)
                    sin_t = tl.sin(theta)
                    va = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    vb = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    vt_rot = tl.where(mask_a[None, :], (cos_t * va - sin_t * vb)[:, None], vt_rot)
                    vt_rot = tl.where(mask_b[None, :], (sin_t * va + cos_t * vb)[:, None], vt_rot)
            gp = tl.sum(go * vt_rot, axis=1)
            grad_probs = tl.where((js == i)[None, :], gp[:, None], grad_probs)

        prob_dot = tl.sum(grad_probs * probs, axis=1)
        dq = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

        for i in range(J_VAL):
            p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)
            gp_i = tl.sum(grad_probs * (js == i)[None, :].to(tl.float32), axis=1)
            grad_s = p_i * (gp_i - prob_dot)
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            grad_s = tl.where(valid, grad_s, 0.0)
            kt = _q6_g128_stage_e_decode_token_rows(
                K_PAYLOAD, K_SCALES, bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            vt = _q6_g128_stage_e_decode_token_rows(
                V_PAYLOAD, V_SCALES, bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            se_i = tl.load(SCALE_EMBED + i * HD + ds, mask=dm, other=0.0).to(tl.float32)
            dq += grad_s[:, None] * (kt + se_i[None, :]) * sc
            gk = grad_s[:, None] * q * sc
            tl.atomic_add(GRAD_K + ((b * H + h) * N + kp[:, None]) * HD + ds[None, :], gk, sem='relaxed', mask=valid[:, None] & dm[None, :])
            gs = tl.sum(grad_s[:, None] * q, axis=0) * sc
            tl.atomic_add(GRAD_SCALE + i * HD + ds, gs, sem='relaxed', mask=dm)
            tl.atomic_add(GRAD_POS + i * H + h, tl.sum(grad_s, axis=0), sem='relaxed')

            gv_rot = p_i[:, None] * go
            gv = gv_rot
            if True:
                slot = i
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = ds == ch_a
                    mask_b = ds == ch_b
                    y_r = tl.load(Y_PRE + ((b * H + h) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z_r = tl.load(Z_PRE + ((b * H + h) * N + tl.maximum(kp, 0)) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gate = tl.sigmoid(tl.load(PHASE_GATE + slot).to(tl.float32))
                    gb_raw = tl.load(PHASE_BASE + (slot * H + h) * R_PLANES_VAL + r).to(tl.float32)
                    gg_raw = tl.load(PHASE_GAIN + (slot * H + h) * R_PLANES_VAL + r).to(tl.float32)
                    gb = gb_raw * gate
                    gg = gg_raw * gate
                    theta = tl.where(valid, gb + gg * y_r * z_r, 0.0)
                    cos_t = tl.cos(theta)
                    sin_t = tl.sin(theta)
                    va = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    vb = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    go_a = tl.sum(gv_rot * mask_a[None, :].to(tl.float32), axis=1)
                    go_b = tl.sum(gv_rot * mask_b[None, :].to(tl.float32), axis=1)
                    gv_a = cos_t * go_a + sin_t * go_b
                    gv_b = -sin_t * go_a + cos_t * go_b
                    gv = tl.where(mask_a[None, :], gv_a[:, None], gv)
                    gv = tl.where(mask_b[None, :], gv_b[:, None], gv)
                    dtheta = ((-sin_t * va - cos_t * vb) * go_a + (cos_t * va - sin_t * vb) * go_b)
                    dtheta = tl.where(valid, dtheta, 0.0)
                    tl.atomic_add(GRAD_GATED_BASE + (slot * H + h) * R_PLANES_VAL + r, tl.sum(dtheta, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_GATED_GAIN + (slot * H + h) * R_PLANES_VAL + r, tl.sum(dtheta * y_r * z_r, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_Y + ((b * H + h) * N + ns) * R_PLANES_VAL + r, dtheta * gg * z_r, sem='relaxed', mask=nm)
                    tl.atomic_add(GRAD_Z + ((b * H + h) * N + kp) * R_PLANES_VAL + r, dtheta * gg * y_r, sem='relaxed', mask=valid)
            tl.atomic_add(GRAD_V + ((b * H + h) * N + kp[:, None]) * HD + ds[None, :], gv, sem='relaxed', mask=valid[:, None] & dm[None, :])


        tl.store(GRAD_Q + ((b * H + h) * N + ns[:, None]) * HD + ds[None, :], dq, mask=nm[:, None] & dm[None, :])


    @triton.jit
    def _q6_g128_dsqg_backward_fused_core_scores_only_kernel(
        Q, K_PAYLOAD, K_SCALES, V_PAYLOAD, V_SCALES, OFFSETS,
        POS_BIAS, SCALE_EMBED, PHASE_GATE, PHASE_BASE, PHASE_GAIN,
        Y_PRE, Z_PRE, GRAD_OUT,
        GRAD_Q, GRAD_K, GRAD_V, GRAD_POS, GRAD_SCALE,
        GRAD_GATED_BASE, GRAD_GATED_GAIN, GRAD_Y, GRAD_Z,
        N: tl.constexpr, H: tl.constexpr, HD: tl.constexpr,
        PAIR_GROUPS: tl.constexpr, PAYLOAD_BYTES: tl.constexpr,
        START: tl.constexpr, TILE_TOKENS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
        J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr, J_PAD: tl.constexpr,
        R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr,
    ):
        """Lower-register Stage-E variant: retain scores only, stream probs/grad-probs."""
        bh = tl.program_id(0)
        block_t = tl.program_id(1)
        b = bh // H
        h = bh % H
        ts = block_t * BLOCK_N + tl.arange(0, BLOCK_N)
        ns = START + ts
        nm = (ts < TILE_TOKENS) & (ns < N)
        ds = tl.arange(0, BLOCK_HD)
        dm = ds < HD
        js = tl.arange(0, J_PAD)
        sc = 1.0 / (HD ** 0.5)
        q = tl.load(Q + ((b * H + h) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        go = tl.load(GRAD_OUT + ((b * H + h) * N + ns[:, None]) * HD + ds[None, :], mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        scores = tl.full([BLOCK_N, J_PAD], float('-inf'), tl.float32)

        for i in range(J_VAL):
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            kt = _q6_g128_stage_e_decode_token_rows(
                K_PAYLOAD, K_SCALES, bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            se_i = tl.load(SCALE_EMBED + i * HD + ds, mask=dm, other=0.0).to(tl.float32)
            s = tl.sum(q * kt, axis=1) * sc
            s += tl.sum(q * se_i[None, :], axis=1) * sc
            s += tl.load(POS_BIAS + i * H + h).to(tl.float32)
            s = tl.where(valid, s, float('-inf'))
            scores = tl.where((js == i)[None, :], s[:, None], scores)

        max_score = tl.max(scores, axis=1)
        all_invalid = max_score == float('-inf')
        safe_max = tl.where(all_invalid, 0.0, max_score)
        denom = tl.zeros([BLOCK_N], tl.float32)
        grad_num = tl.zeros([BLOCK_N], tl.float32)

        for i in range(J_VAL):
            s_i = tl.sum(tl.where((js == i)[None, :], scores, 0.0), axis=1)
            e_i = tl.exp2((s_i - safe_max) * 1.4426950408889634)
            e_i = tl.where(s_i == float('-inf'), 0.0, e_i)
            denom += e_i

            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            vt = _q6_g128_stage_e_decode_token_rows(
                V_PAYLOAD, V_SCALES, bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            vt_rot = vt
            if i >= J_SMALL_VAL:
                slot = i - J_SMALL_VAL
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = ds == ch_a
                    mask_b = ds == ch_b
                    y_r = tl.load(Y_PRE + ((b * H + h) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z_r = tl.load(Z_PRE + ((b * H + h) * N + tl.maximum(kp, 0)) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gate = tl.sigmoid(tl.load(PHASE_GATE + slot).to(tl.float32))
                    gb = tl.load(PHASE_BASE + (slot * H + h) * R_PLANES_VAL + r).to(tl.float32) * gate
                    gg = tl.load(PHASE_GAIN + (slot * H + h) * R_PLANES_VAL + r).to(tl.float32) * gate
                    theta = tl.where(valid, gb + gg * y_r * z_r, 0.0)
                    cos_t = tl.cos(theta)
                    sin_t = tl.sin(theta)
                    va = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    vb = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    vt_rot = tl.where(mask_a[None, :], (cos_t * va - sin_t * vb)[:, None], vt_rot)
                    vt_rot = tl.where(mask_b[None, :], (sin_t * va + cos_t * vb)[:, None], vt_rot)
            gp = tl.sum(go * vt_rot, axis=1)
            grad_num += e_i * gp

        safe_denom = tl.where(denom > 0.0, denom, 1.0)
        prob_dot = grad_num / safe_denom
        dq = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

        for i in range(J_VAL):
            s_i = tl.sum(tl.where((js == i)[None, :], scores, 0.0), axis=1)
            e_i = tl.exp2((s_i - safe_max) * 1.4426950408889634)
            e_i = tl.where(s_i == float('-inf'), 0.0, e_i)
            p_i = e_i / safe_denom
            delta = tl.load(OFFSETS + i).to(tl.int32)
            kp = ns.to(tl.int32) - delta
            valid = nm & (kp >= 0) & (kp < N)
            kt = _q6_g128_stage_e_decode_token_rows(
                K_PAYLOAD, K_SCALES, bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            vt = _q6_g128_stage_e_decode_token_rows(
                V_PAYLOAD, V_SCALES, bh, kp, valid, ds, dm,
                PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
            )
            gv_rot = p_i[:, None] * go
            vt_rot = vt
            gv = gv_rot
            if i >= J_SMALL_VAL:
                slot = i - J_SMALL_VAL
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = ds == ch_a
                    mask_b = ds == ch_b
                    y_r = tl.load(Y_PRE + ((b * H + h) * N + ns) * R_PLANES_VAL + r, mask=nm, other=0.0).to(tl.float32)
                    z_r = tl.load(Z_PRE + ((b * H + h) * N + tl.maximum(kp, 0)) * R_PLANES_VAL + r, mask=valid, other=0.0).to(tl.float32)
                    gate = tl.sigmoid(tl.load(PHASE_GATE + slot).to(tl.float32))
                    gb_raw = tl.load(PHASE_BASE + (slot * H + h) * R_PLANES_VAL + r).to(tl.float32)
                    gg_raw = tl.load(PHASE_GAIN + (slot * H + h) * R_PLANES_VAL + r).to(tl.float32)
                    gb = gb_raw * gate
                    gg = gg_raw * gate
                    theta = tl.where(valid, gb + gg * y_r * z_r, 0.0)
                    cos_t = tl.cos(theta)
                    sin_t = tl.sin(theta)
                    va = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    vb = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    vt_rot = tl.where(mask_a[None, :], (cos_t * va - sin_t * vb)[:, None], vt_rot)
                    vt_rot = tl.where(mask_b[None, :], (sin_t * va + cos_t * vb)[:, None], vt_rot)
                    go_a = tl.sum(gv_rot * mask_a[None, :].to(tl.float32), axis=1)
                    go_b = tl.sum(gv_rot * mask_b[None, :].to(tl.float32), axis=1)
                    gv_a = cos_t * go_a + sin_t * go_b
                    gv_b = -sin_t * go_a + cos_t * go_b
                    gv = tl.where(mask_a[None, :], gv_a[:, None], gv)
                    gv = tl.where(mask_b[None, :], gv_b[:, None], gv)
                    dtheta = ((-sin_t * va - cos_t * vb) * go_a + (cos_t * va - sin_t * vb) * go_b)
                    dtheta = tl.where(valid, dtheta, 0.0)
                    tl.atomic_add(GRAD_GATED_BASE + (slot * H + h) * R_PLANES_VAL + r, tl.sum(dtheta, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_GATED_GAIN + (slot * H + h) * R_PLANES_VAL + r, tl.sum(dtheta * y_r * z_r, axis=0), sem='relaxed')
                    tl.atomic_add(GRAD_Y + ((b * H + h) * N + ns) * R_PLANES_VAL + r, dtheta * gg * z_r, sem='relaxed', mask=nm)
                    tl.atomic_add(GRAD_Z + ((b * H + h) * N + kp) * R_PLANES_VAL + r, dtheta * gg * y_r, sem='relaxed', mask=valid)
            gp_i = tl.sum(go * vt_rot, axis=1)
            grad_s = p_i * (gp_i - prob_dot)
            grad_s = tl.where(valid, grad_s, 0.0)
            se_i = tl.load(SCALE_EMBED + i * HD + ds, mask=dm, other=0.0).to(tl.float32)
            dq += grad_s[:, None] * (kt + se_i[None, :]) * sc
            gk = grad_s[:, None] * q * sc
            tl.atomic_add(GRAD_K + ((b * H + h) * N + kp[:, None]) * HD + ds[None, :], gk, sem='relaxed', mask=valid[:, None] & dm[None, :])
            gs = tl.sum(grad_s[:, None] * q, axis=0) * sc
            tl.atomic_add(GRAD_SCALE + i * HD + ds, gs, sem='relaxed', mask=dm)
            tl.atomic_add(GRAD_POS + i * H + h, tl.sum(grad_s, axis=0), sem='relaxed')
            tl.atomic_add(GRAD_V + ((b * H + h) * N + kp[:, None]) * HD + ds[None, :], gv, sem='relaxed', mask=valid[:, None] & dm[None, :])

        tl.store(GRAD_Q + ((b * H + h) * N + ns[:, None]) * HD + ds[None, :], dq, mask=nm[:, None] & dm[None, :])


def _q6_g128_dsqg_backward_fused_core_triton(
    q, k_layout, v_layout, offsets, *, start, end,
    pos_bias, scale_embed, phase_gate, phase_base, phase_gain,
    y_pre, z_pre, grad_out,
    grad_q, grad_k, grad_v, grad_pos, grad_scale,
    grad_gated_base, grad_gated_gain, grad_y, grad_z,
    j_small, plane_shift, block_n=32,
):
    """Stage-E fused tile backward core: q6 decode + DSQG backward + sequence scatter.

    This is intentionally an opt-in experimental larger fusion boundary. It emits
    gradients directly into sequence/reduction buffers and does not materialize
    q6 K/V replay tiles or gathered K/V gradient tiles.
    """
    if (
        not _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE
        or not q.is_cuda
        or not k_layout.payload.is_cuda
        or not v_layout.payload.is_cuda
        or int(q.shape[-1]) != 64
    ):
        k_g, idx, valid = _q6_g128_decode_gather_tile_triton(k_layout, offsets, start=start, end=end)
        v_g, idx_v, valid_v = _q6_g128_decode_gather_tile_triton(v_layout, offsets, start=start, end=end)
        if not torch.equal(idx, idx_v) or not torch.equal(valid, valid_v):
            raise RuntimeError('q6 Stage-E fallback K/V masks diverged')
        out = _q6_g128_dsqg_consume_tile_backward_stage_d_vectorized(
            q[:, :, start:end, :], k_g.float(), v_g.float(), valid, idx,
            pos_bias, scale_embed, phase_gate, phase_base, phase_gain,
            y_pre[:, :, start:end, :], z_pre, grad_out[:, :, start:end, :],
            j_small=j_small, plane_shift=plane_shift,
        )
        grad_q[:, :, start:end, :] += out[0].float()
        _q6_g128_accumulate_gathered_grads_triton(grad_k, out[1], valid, idx)
        _q6_g128_accumulate_gathered_grads_triton(grad_v, out[2], valid, idx)
        grad_pos += out[3].float()
        grad_scale += out[4].float()
        gate = torch.sigmoid(phase_gate.float())
        grad_gated_base += out[6].float() / gate[:, None, None].clamp_min(1e-20)
        grad_gated_gain += out[7].float() / gate[:, None, None].clamp_min(1e-20)
        grad_y[:, :, start:end, :] += out[8].float()
        grad_z += out[9].float()
        return

    offsets_dev = torch.tensor([int(o) for o in offsets], device=q.device, dtype=torch.int32).contiguous()
    start = int(start)
    end = int(end)
    tile_tokens = end - start
    if tile_tokens <= 0:
        return
    q_c = q.contiguous()
    go_c = grad_out.contiguous()
    y_c = y_pre.contiguous()
    z_c = z_pre.contiguous()
    pos_c = pos_bias.contiguous()
    scale_c = scale_embed.contiguous()
    pg_c = phase_gate.contiguous()
    pb_c = phase_base.contiguous()
    pgn_c = phase_gain.contiguous()
    b, h, n, d = (int(v) for v in q_c.shape)
    h_kv = int(k_layout.heads)
    if h % h_kv != 0:
        raise ValueError(f'q6 Stage-E Hq={h} must be divisible by Hkv={h_kv}')
    kv_group = h // h_kv
    j_val = len(offsets)
    j_large = j_val - int(j_small)
    block_n = int(block_n)
    block_hd = _next_pow2(d)
    j_pad = max(16, _next_pow2(j_val))
    use_pair_backward = bool(
        Q6_G128_STAGE_F3_PAIR_BACKWARD
        and not Q6_G128_STAGE_E_SCORES_ONLY
        and not Q6_G128_STAGE_E_GROUP_SPECIALIZE
        and h_kv * 2 == h
        and kv_group == 2
    )
    if use_pair_backward:
        tile_blocks = int(math.ceil(tile_tokens / block_n))
        grid_pair = (b * h_kv, tile_blocks)
        if Q6_G128_STAGE_F3_SPARSE_MOVT:
            use_split_vphase = bool(Q6_G128_STAGE_F3_SPLIT_MOVT_CORR and Q6_G128_STAGE_F3_SPLIT_VPHASE)
            use_prob_scratch = bool(
                Q6_G128_STAGE_F3_SPLIT_MOVT_CORR
                and Q6_G128_STAGE_F3_SPLIT_MOVT_PROB_SCRATCH
                and not use_split_vphase
                and j_large > 0
            )
            prob_scratch = (
                torch.empty((b * h_kv * tile_blocks * j_large * 2 * block_n,), device=q_c.device, dtype=torch.float32)
                if use_prob_scratch else torch.empty((1,), device=q_c.device, dtype=torch.float32)
            )
            _q6_g128_dsqg_backward_fused_core_pair_q_sparse_movt_kernel[grid_pair](
                q_c, k_layout.payload.contiguous(), k_layout.scales.contiguous(),
                v_layout.payload.contiguous(), v_layout.scales.contiguous(), offsets_dev,
                pos_c, scale_c, pg_c, pb_c, pgn_c, y_c, z_c, go_c,
                grad_q, grad_k, grad_v, grad_pos, grad_scale,
                grad_gated_base, grad_gated_gain, grad_y, grad_z,
                prob_scratch,
                N=n, H_Q=h, H_KV=h_kv, HD=d, PAIR_GROUPS=k_layout.pair_groups,
                PAYLOAD_BYTES=_q6_layout_mod.PAYLOAD_BYTES_PER_GROUP,
                START=start, TILE_TOKENS=tile_tokens,
                BLOCK_N=block_n, BLOCK_HD=block_hd,
                J_VAL=j_val, J_SMALL_VAL=int(j_small), J_LARGE_VAL=j_large, J_PAD=j_pad,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=int(plane_shift),
                SCRATCH_BLOCKS=tile_blocks,
                SPLIT_MOVT_CORR=Q6_G128_STAGE_F3_SPLIT_MOVT_CORR,
                SPLIT_VPHASE=use_split_vphase,
                STORE_PROB_SCRATCH=use_prob_scratch,
                num_warps=Q6_G128_STAGE_E_NUM_WARPS, num_stages=2,
            )
            if Q6_G128_STAGE_F3_SPLIT_MOVT_CORR:
                _q6_g128_dsqg_backward_fused_core_pair_q_movt_corr_sidecar_kernel[grid_pair](
                    q_c, k_layout.payload.contiguous(), k_layout.scales.contiguous(),
                    v_layout.payload.contiguous(), v_layout.scales.contiguous(), offsets_dev,
                    pos_c, scale_c, pg_c, pb_c, pgn_c, y_c, z_c, go_c,
                    grad_v, grad_gated_base, grad_gated_gain, grad_y, grad_z,
                    prob_scratch,
                    N=n, H_Q=h, H_KV=h_kv, HD=d, PAIR_GROUPS=k_layout.pair_groups,
                    PAYLOAD_BYTES=_q6_layout_mod.PAYLOAD_BYTES_PER_GROUP,
                    START=start, TILE_TOKENS=tile_tokens,
                    BLOCK_N=block_n, BLOCK_HD=block_hd,
                    J_VAL=j_val, J_SMALL_VAL=int(j_small), J_LARGE_VAL=j_large, J_PAD=j_pad,
                    R_PLANES_VAL=R_PLANES, PLANE_SHIFT=int(plane_shift),
                    SCRATCH_BLOCKS=tile_blocks,
                    INCLUDE_BASE_V=Q6_G128_STAGE_F3_SPLIT_VPHASE,
                    USE_PROB_SCRATCH=use_prob_scratch,
                    SIDECAR_PAIR_DIRECT=Q6_G128_STAGE_F3_SIDECAR_PAIR_DIRECT,
                    num_warps=Q6_G128_STAGE_E_NUM_WARPS, num_stages=2,
                )
        else:
            _q6_g128_dsqg_backward_fused_core_pair_q_kernel[grid_pair](
                q_c, k_layout.payload.contiguous(), k_layout.scales.contiguous(),
                v_layout.payload.contiguous(), v_layout.scales.contiguous(), offsets_dev,
                pos_c, scale_c, pg_c, pb_c, pgn_c, y_c, z_c, go_c,
                grad_q, grad_k, grad_v, grad_pos, grad_scale,
                grad_gated_base, grad_gated_gain, grad_y, grad_z,
                N=n, H_Q=h, H_KV=h_kv, HD=d, PAIR_GROUPS=k_layout.pair_groups,
                PAYLOAD_BYTES=_q6_layout_mod.PAYLOAD_BYTES_PER_GROUP,
                START=start, TILE_TOKENS=tile_tokens,
                BLOCK_N=block_n, BLOCK_HD=block_hd,
                J_VAL=j_val, J_SMALL_VAL=int(j_small), J_LARGE_VAL=j_large, J_PAD=j_pad,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=int(plane_shift),
                num_warps=Q6_G128_STAGE_E_NUM_WARPS, num_stages=2,
            )
        return
    grid = (b * h, triton.cdiv(tile_tokens, block_n))
    use_large_only_specialist = (
        Q6_G128_STAGE_E_GROUP_SPECIALIZE
        and not Q6_G128_STAGE_E_SCORES_ONLY
        and int(j_small) == 0
        and j_val == 32
    )
    kernel = (_q6_g128_dsqg_backward_fused_core_scores_only_kernel
              if Q6_G128_STAGE_E_SCORES_ONLY
              else _q6_g128_dsqg_backward_fused_core_large_only_kernel
              if use_large_only_specialist
              else _q6_g128_dsqg_backward_fused_core_kernel)
    if kernel is _q6_g128_dsqg_backward_fused_core_kernel:
        kernel[grid](
            q_c, k_layout.payload.contiguous(), k_layout.scales.contiguous(),
            v_layout.payload.contiguous(), v_layout.scales.contiguous(), offsets_dev,
            pos_c, scale_c, pg_c, pb_c, pgn_c, y_c, z_c, go_c,
            grad_q, grad_k, grad_v, grad_pos, grad_scale,
            grad_gated_base, grad_gated_gain, grad_y, grad_z,
            N=n, H_Q=h, H_KV=h_kv, KV_GROUP=kv_group, HD=d, PAIR_GROUPS=k_layout.pair_groups,
            PAYLOAD_BYTES=_q6_layout_mod.PAYLOAD_BYTES_PER_GROUP,
            START=start, TILE_TOKENS=tile_tokens,
            BLOCK_N=block_n, BLOCK_HD=block_hd,
            J_VAL=j_val, J_SMALL_VAL=int(j_small), J_LARGE_VAL=j_large, J_PAD=j_pad,
            R_PLANES_VAL=R_PLANES, PLANE_SHIFT=int(plane_shift),
            num_warps=Q6_G128_STAGE_E_NUM_WARPS, num_stages=2,
        )
    else:
        kernel[grid](
            q_c, k_layout.payload.contiguous(), k_layout.scales.contiguous(),
            v_layout.payload.contiguous(), v_layout.scales.contiguous(), offsets_dev,
            pos_c, scale_c, pg_c, pb_c, pgn_c, y_c, z_c, go_c,
            grad_q, grad_k, grad_v, grad_pos, grad_scale,
            grad_gated_base, grad_gated_gain, grad_y, grad_z,
            N=n, H=h, HD=d, PAIR_GROUPS=k_layout.pair_groups,
            PAYLOAD_BYTES=_q6_layout_mod.PAYLOAD_BYTES_PER_GROUP,
            START=start, TILE_TOKENS=tile_tokens,
            BLOCK_N=block_n, BLOCK_HD=block_hd,
            J_VAL=j_val, J_SMALL_VAL=int(j_small), J_LARGE_VAL=j_large, J_PAD=j_pad,
            R_PLANES_VAL=R_PLANES, PLANE_SHIFT=int(plane_shift),
            num_warps=Q6_G128_STAGE_E_NUM_WARPS, num_stages=2,
        )


class _Q6G128FusedConsumeStageC(torch.autograd.Function):
    """Stage-C fused q6 direct-consume forward with tiled STE recompute backward.

    Forward saves compact q6 K/V payloads from the exact stochastic pack used for
    the fused forward. Backward replays the direct-gather PyTorch consumer one
    query tile at a time, so no full [B,H,N,J,D] gathered K/V tensor is saved or
    recreated while preserving the existing BF16-gather STE gradient surface.
    """

    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed, phase_gate, phase_base, phase_gain,
                y_pre, z_pre, offsets, j_small, plane_shift, seed, block_n, tile_tokens):
        offsets = tuple(int(o) for o in offsets)
        k_layout = _q6_layout_mod.pack_q6_g128_cache_layout(k.detach(), seed=int(seed))
        v_layout = _q6_layout_mod.pack_q6_g128_cache_layout(v.detach(), seed=int(seed) + 1)
        offsets_dev = torch.tensor(offsets, device=q.device, dtype=torch.int32)
        gate = torch.sigmoid(phase_gate).float()[:, None, None]
        gated_phase_base = (phase_base.float() * gate).contiguous()
        gated_phase_gain = (phase_gain.float() * gate).contiguous()
        out, _lse, valid_counts = _q6_fused_mod.triton_q6_g128_dsqg_direct_consume(
            q.to(torch.bfloat16).contiguous(),
            k_layout,
            v_layout,
            offsets_dev,
            pos_bias.float().contiguous(),
            scale_embed.float().contiguous(),
            gated_phase_base,
            gated_phase_gain,
            y_pre.float().contiguous(),
            z_pre.float().contiguous(),
            j_small=int(j_small),
            plane_shift=int(plane_shift),
            block_n=int(block_n),
            return_report=False,
            pair_q_heads=bool(Q6_G128_STAGE_F2_PAIR_REUSE),
        )
        expected_counts = torch.tensor(
            [sum(1 for o in offsets if i - o >= 0) for i in range(q.shape[2])],
            device=q.device,
            dtype=torch.int32,
        )
        if not torch.equal(valid_counts, expected_counts):
            raise RuntimeError('q6 Stage-C fused direct-consume causal valid counts diverged')
        ctx.save_for_backward(
            q.detach(), y_pre.detach(), z_pre.detach(),
            pos_bias.detach(), scale_embed.detach(), phase_gate.detach(), phase_base.detach(), phase_gain.detach(),
            k_layout.payload, k_layout.scales, v_layout.payload, v_layout.scales,
        )
        ctx.offsets = offsets
        ctx.j_small = int(j_small)
        ctx.plane_shift = int(plane_shift)
        ctx.seed = int(seed)
        ctx.block_n = int(block_n)
        ctx.tile_tokens = _q6_g128_effective_stage_c_tile(q.shape[2], tile_tokens)
        ctx.shape_meta = (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3]))
        ctx.kv_shape_meta = (int(k.shape[0]), int(k.shape[1]), int(k.shape[2]), int(k.shape[3]))
        return out.float()

    @staticmethod
    def backward(ctx, grad_out):
        (q_saved, y_saved, z_saved, pos_saved, scale_saved, phase_gate_saved,
         phase_base_saved, phase_gain_saved, k_payload, k_scales, v_payload, v_scales) = ctx.saved_tensors
        b, h, n, d = ctx.shape_meta
        b_kv, h_kv, n_kv, d_kv = ctx.kv_shape_meta
        if (b_kv, n_kv, d_kv) != (b, n, d) or h % h_kv != 0:
            raise RuntimeError(f'q6 Stage-F.1 invalid q/kv shape meta: q={ctx.shape_meta}, kv={ctx.kv_shape_meta}')
        offsets = list(ctx.offsets)
        k_layout = _q6_g128_layout_from_saved(k_payload, k_scales, batch=b_kv, heads=h_kv, seq_len=n_kv, head_dim=d_kv, seed=ctx.seed)
        v_layout = _q6_g128_layout_from_saved(v_payload, v_scales, batch=b_kv, heads=h_kv, seq_len=n_kv, head_dim=d_kv, seed=ctx.seed + 1)
        needs = ctx.needs_input_grad

        grad_q = torch.zeros_like(q_saved, dtype=torch.float32) if needs[0] else None
        grad_k = torch.zeros((b_kv, h_kv, n_kv, d_kv), device=q_saved.device, dtype=torch.float32) if needs[1] else None
        grad_v = torch.zeros((b_kv, h_kv, n_kv, d_kv), device=q_saved.device, dtype=torch.float32) if needs[2] else None
        grad_pos = torch.zeros_like(pos_saved, dtype=torch.float32) if needs[3] else None
        grad_scale = torch.zeros_like(scale_saved, dtype=torch.float32) if needs[4] else None
        grad_phase_gate = torch.zeros_like(phase_gate_saved, dtype=torch.float32) if needs[5] else None
        grad_phase_base = torch.zeros_like(phase_base_saved, dtype=torch.float32) if needs[6] else None
        grad_phase_gain = torch.zeros_like(phase_gain_saved, dtype=torch.float32) if needs[7] else None
        grad_y = torch.zeros_like(y_saved, dtype=torch.float32) if needs[8] else None
        grad_z = torch.zeros_like(z_saved, dtype=torch.float32) if needs[9] else None
        use_stage_e = (
            Q6_G128_STAGE_E_BACKWARD
            and Q6_G128_STAGE_D_BACKWARD
            and _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE
            and q_saved.is_cuda
            and all(bool(needs[i]) for i in range(10))
        )
        grad_gated_base = torch.zeros_like(phase_base_saved, dtype=torch.float32) if use_stage_e else None
        grad_gated_gain = torch.zeros_like(phase_gain_saved, dtype=torch.float32) if use_stage_e else None

        for start in range(0, n, ctx.tile_tokens):
            end = min(n, start + ctx.tile_tokens)
            if use_stage_e:
                _q6_g128_dsqg_backward_fused_core_triton(
                    q_saved, k_layout, v_layout, offsets, start=start, end=end,
                    pos_bias=pos_saved, scale_embed=scale_saved,
                    phase_gate=phase_gate_saved, phase_base=phase_base_saved, phase_gain=phase_gain_saved,
                    y_pre=y_saved, z_pre=z_saved, grad_out=grad_out,
                    grad_q=grad_q, grad_k=grad_k, grad_v=grad_v, grad_pos=grad_pos, grad_scale=grad_scale,
                    grad_gated_base=grad_gated_base, grad_gated_gain=grad_gated_gain,
                    grad_y=grad_y, grad_z=grad_z,
                    j_small=ctx.j_small, plane_shift=ctx.plane_shift,
                    block_n=ctx.block_n,
                )
                continue
            q_tile = q_saved[:, :, start:end, :]
            y_tile = y_saved[:, :, start:end, :]
            decode_fn = (_q6_g128_decode_gather_tile_triton
                         if Q6_G128_STAGE_D_BACKWARD
                         else _q6_g128_decode_gather_tile)
            k_q6, idx, valid = decode_fn(k_layout, offsets, start=start, end=end)
            v_q6, idx_v, valid_v = decode_fn(v_layout, offsets, start=start, end=end)
            if not torch.equal(idx, idx_v) or not torch.equal(valid, valid_v):
                raise RuntimeError('q6 Stage-C K/V tile masks diverged')
            backward_fn = (_q6_g128_dsqg_consume_tile_backward_stage_d_vectorized
                           if Q6_G128_STAGE_D_BACKWARD
                           else _q6_g128_dsqg_consume_tile_backward_manual)
            (g_q_tile, g_k_g, g_v_g, g_pos, g_scale, g_phase_gate,
             g_phase_base, g_phase_gain, g_y_tile, g_z) = backward_fn(
                q_tile, k_q6.float(), v_q6.float(), valid, idx,
                pos_saved, scale_saved, phase_gate_saved, phase_base_saved, phase_gain_saved,
                y_tile, z_saved, grad_out[:, :, start:end, :],
                j_small=ctx.j_small, plane_shift=ctx.plane_shift,
            )

            if needs[0]:
                grad_q[:, :, start:end, :] += g_q_tile.float()
            if needs[1]:
                if Q6_G128_STAGE_D_BACKWARD:
                    _q6_g128_accumulate_gathered_grads_triton(grad_k, g_k_g, valid, idx)
                else:
                    for j in range(len(offsets)):
                        contrib = g_k_g[:, :, :, j, :].float() * valid[:, j].reshape(1, 1, end - start, 1).to(g_k_g.dtype)
                        grad_k.index_add_(2, idx[:, j], contrib)
            if needs[2]:
                if Q6_G128_STAGE_D_BACKWARD:
                    _q6_g128_accumulate_gathered_grads_triton(grad_v, g_v_g, valid, idx)
                else:
                    for j in range(len(offsets)):
                        contrib = g_v_g[:, :, :, j, :].float() * valid[:, j].reshape(1, 1, end - start, 1).to(g_v_g.dtype)
                        grad_v.index_add_(2, idx[:, j], contrib)
            if needs[3]:
                grad_pos += g_pos.float()
            if needs[4]:
                grad_scale += g_scale.float()
            if needs[5]:
                grad_phase_gate += g_phase_gate.float()
            if needs[6]:
                grad_phase_base += g_phase_base.float()
            if needs[7]:
                grad_phase_gain += g_phase_gain.float()
            if needs[8]:
                grad_y[:, :, start:end, :] += g_y_tile.float()
            if needs[9]:
                grad_z += g_z.float()

        if use_stage_e:
            gate = torch.sigmoid(phase_gate_saved.float())
            if needs[6]:
                grad_phase_base += grad_gated_base * gate[:, None, None]
            if needs[7]:
                grad_phase_gain += grad_gated_gain * gate[:, None, None]
            if needs[5]:
                grad_gate = (grad_gated_base * phase_base_saved.float() + grad_gated_gain * phase_gain_saved.float()).sum(dim=(1, 2))
                grad_phase_gate += grad_gate * gate * (1.0 - gate)

        return (
            grad_q, grad_k, grad_v, grad_pos, grad_scale, grad_phase_gate,
            grad_phase_base, grad_phase_gain, grad_y, grad_z,
            None, None, None, None, None, None,
        )


def _q6_g128_fused_report_from_shape(q, offsets, *, stage_c=False, kv_heads=None):
    b, h, n, d = (int(v) for v in q.shape)
    h_kv = int(kv_heads) if kv_heads is not None else h
    kv_group_size = h // h_kv if h_kv else 0
    pair_groups = int(math.ceil(n / _q6_layout_mod.TOKENS_PER_GROUP))
    one_layout_bytes = b * h_kv * pair_groups * (_q6_layout_mod.PAYLOAD_BYTES_PER_GROUP + _q6_layout_mod.SCALE_BYTES_PER_GROUP)
    resident_q6_bytes = 2 * one_layout_bytes
    values = b * h * n * d
    bf16_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()
    one_gather_bytes = b * h * n * len(offsets) * d * bf16_bytes
    report_j_small = sum(1 for o in offsets if int(o) <= 28)
    report_large_only_specialist = bool(
        stage_c and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD
        and Q6_G128_STAGE_E_GROUP_SPECIALIZE and not Q6_G128_STAGE_E_SCORES_ONLY
        and report_j_small == 0 and len(offsets) == 32 and _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE
    )
    tile_tokens = _q6_g128_effective_stage_c_tile(n) if stage_c else 0
    tile_gather_bytes = 2 * b * h * tile_tokens * len(offsets) * d * bf16_bytes if stage_c else 0
    replay_tiles = int(math.ceil(n / tile_tokens)) if stage_c and tile_tokens > 0 else 0
    full_kv_gather_bytes = 2 * one_gather_bytes
    return {
        'read_implementation': (
            'q6_triton_fused_direct_consume_stage_e_fused_bwd'
            if stage_c and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD else
            'q6_triton_fused_direct_consume_stage_d_vectorized_bwd'
            if stage_c and Q6_G128_STAGE_D_BACKWARD else
            'q6_triton_fused_direct_consume_stage_c' if stage_c else 'q6_triton_fused_direct_consume'
        ),
        'scratch_mode': 'direct_q6_decode_to_dsqg_output',
        'peak_scratch_bytes': 0,
        'peak_scratch_vs_full_scratch': 0.0,
        'gather_output_bytes': 0,
        'avoided_gather_bytes': int(full_kv_gather_bytes),
        'attention_output_bytes': int(values * bf16_bytes),
        'lse_bytes': int(b * h * n * torch.tensor([], dtype=torch.float32).element_size()),
        'materialized_gather_bytes': 0,
        'resident_q6_bytes': int(resident_q6_bytes),
        'num_query_heads': int(h),
        'num_kv_heads': int(h_kv),
        'kv_group_size': int(kv_group_size),
        'compression_vs_bf16': float((2 * values * 2) / resident_q6_bytes),
        'stage_c_backward_tile_tokens': int(tile_tokens),
        'stage_c_backward_replay_tiles': int(replay_tiles),
        'stage_c_backward_tile_gather_bytes': int(tile_gather_bytes),
        'stage_c_backward_tile_gather_vs_full': float(tile_gather_bytes / full_kv_gather_bytes) if full_kv_gather_bytes else 0.0,
        'stage_d_backward_enabled': bool(stage_c and Q6_G128_STAGE_D_BACKWARD),
        'stage_d_sequence_scatter': (
            'triton_atomic' if stage_c and Q6_G128_STAGE_D_BACKWARD and _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE
            else 'torch_vectorized' if stage_c and Q6_G128_STAGE_D_BACKWARD
            else 'python_offset_loop'
        ),
        'stage_d_tile_decode': (
            'stage_e_fused_decode_inside_backward_core' if stage_c and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD else
            'triton_direct_tile' if stage_c and Q6_G128_STAGE_D_BACKWARD and _Q6_TRITON_AVAILABLE
            else 'pytorch_pair_decode'
        ),
        'stage_e_fused_backward_enabled': bool(stage_c and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD),
        'stage_e_group_specialize_enabled': bool(Q6_G128_STAGE_E_GROUP_SPECIALIZE),
        'stage_e_group_specialized_large_only': report_large_only_specialist,
        'stage_f2_pair_forward_enabled': bool(stage_c and Q6_G128_STAGE_F2_PAIR_REUSE and h_kv * 2 == h),
        'stage_f2_pair_forward_core': (
            'triton_pair_query_head_forward_reuse'
            if stage_c and Q6_G128_STAGE_F2_PAIR_REUSE and h_kv * 2 == h else 'disabled'
        ),
        'stage_f3_pair_backward_enabled': bool(
            stage_c and Q6_G128_STAGE_F3_PAIR_BACKWARD and h_kv * 2 == h
            and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD
            and not Q6_G128_STAGE_E_SCORES_ONLY and not Q6_G128_STAGE_E_GROUP_SPECIALIZE
        ),
        'stage_f3_pair_backward_core': (
            'triton_pair_query_head_sparse_movt_split_vphase_correction'
            if stage_c and Q6_G128_STAGE_F3_PAIR_BACKWARD and Q6_G128_STAGE_F3_SPARSE_MOVT
            and Q6_G128_STAGE_F3_SPLIT_MOVT_CORR and Q6_G128_STAGE_F3_SPLIT_VPHASE and h_kv * 2 == h
            and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD
            and not Q6_G128_STAGE_E_SCORES_ONLY and not Q6_G128_STAGE_E_GROUP_SPECIALIZE
            else 'triton_pair_query_head_sparse_movt_split_correction'
            if stage_c and Q6_G128_STAGE_F3_PAIR_BACKWARD and Q6_G128_STAGE_F3_SPARSE_MOVT
            and Q6_G128_STAGE_F3_SPLIT_MOVT_CORR and h_kv * 2 == h
            and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD
            and not Q6_G128_STAGE_E_SCORES_ONLY and not Q6_G128_STAGE_E_GROUP_SPECIALIZE
            else 'triton_pair_query_head_sparse_movt_correction'
            if stage_c and Q6_G128_STAGE_F3_PAIR_BACKWARD and Q6_G128_STAGE_F3_SPARSE_MOVT and h_kv * 2 == h
            and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD
            and not Q6_G128_STAGE_E_SCORES_ONLY and not Q6_G128_STAGE_E_GROUP_SPECIALIZE
            else 'triton_pair_query_head_backward_scatter_reuse'
            if stage_c and Q6_G128_STAGE_F3_PAIR_BACKWARD and h_kv * 2 == h
            and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD
            and not Q6_G128_STAGE_E_SCORES_ONLY and not Q6_G128_STAGE_E_GROUP_SPECIALIZE
            else 'disabled'
        ),
        'stage_f3_split_movt_prob_scratch_enabled': bool(Q6_G128_STAGE_F3_SPLIT_MOVT_PROB_SCRATCH),
        'stage_f3_sidecar_pair_direct_enabled': bool(Q6_G128_STAGE_F3_SIDECAR_PAIR_DIRECT),
        'stage_e_num_warps': int(Q6_G128_STAGE_E_NUM_WARPS),
        'stage_e_backward_core': (
            'triton_kv_head_aware_decode_softmax_movt_scatter'
            if stage_c and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD and _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE and h_kv != h else
            'triton_scores_only_decode_softmax_movt_scatter'
            if stage_c and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD and Q6_G128_STAGE_E_SCORES_ONLY and _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE else
            'triton_large_only_decode_softmax_movt_scatter'
            if report_large_only_specialist else
            'triton_decode_softmax_movt_scatter'
            if stage_c and Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD and _Q6_STAGE_D_TRITON_SCATTER_AVAILABLE
            else 'disabled'
        ),
    }


def _q6_g128_triton_fused_direct_consume(q, k, v, offsets, module, y_pre, z_pre, *, seed):
    """q6 fused direct-consume forward path with Stage-C tiled STE backward in training."""
    if not _Q6_LAYOUT_AVAILABLE:
        raise RuntimeError('DWARF_Q6_G128_FUSED_CONSUME=1 requires kernels/q6_g128/layout.py')
    if not _Q6_FUSED_CONSUME_AVAILABLE:
        raise RuntimeError('DWARF_Q6_G128_FUSED_CONSUME=1 requires kernels/q6_g128/fused_consume.py')
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise RuntimeError('q6_g128 fused direct-consume path requires CUDA')
    if q.shape[-1] != 64 or k.shape[-1] != 64 or v.shape[-1] != 64:
        raise ValueError(f'q6_g128 fused consume requires head_dim=64, got q/k/v={q.shape[-1]}/{k.shape[-1]}/{v.shape[-1]}')

    offsets = [int(o) for o in offsets]
    pos_bias_scale = getattr(module, 'pos_bias_scale', None)
    effective_pos_bias = module.pos_bias if pos_bias_scale is None else module.pos_bias * pos_bias_scale.to(
        device=module.pos_bias.device,
        dtype=module.pos_bias.dtype,
    )
    if module.training or torch.is_grad_enabled():
        out = _Q6G128FusedConsumeStageC.apply(
            q, k, v, effective_pos_bias, module.scale_embed, module.phase_gate,
            module.phase_base, module.phase_gain, y_pre, z_pre, tuple(offsets),
            int(module.j_small), int(module.plane_shift), int(seed),
            int(Q6_G128_FUSED_BLOCK_N), int(Q6_G128_STAGE_C_TILE),
        )
        return out, _q6_g128_fused_report_from_shape(q, offsets, stage_c=True, kv_heads=int(k.shape[1]))

    offsets_dev = torch.tensor(offsets, device=q.device, dtype=torch.int32)
    k_layout = _q6_layout_mod.pack_q6_g128_cache_layout(k.detach(), seed=seed)
    v_layout = _q6_layout_mod.pack_q6_g128_cache_layout(v.detach(), seed=seed + 1)
    phase_gate = torch.sigmoid(module.phase_gate).float()[:, None, None]
    gated_phase_base = (module.phase_base.float() * phase_gate).contiguous()
    gated_phase_gain = (module.phase_gain.float() * phase_gate).contiguous()
    out, lse, valid_counts, fused_report = _q6_fused_mod.triton_q6_g128_dsqg_direct_consume(
        q.to(torch.bfloat16).contiguous(),
        k_layout,
        v_layout,
        offsets_dev,
        effective_pos_bias.float().contiguous(),
        module.scale_embed.float().contiguous(),
        gated_phase_base,
        gated_phase_gain,
        y_pre.float().contiguous(),
        z_pre.float().contiguous(),
        j_small=module.j_small,
        plane_shift=module.plane_shift,
        block_n=Q6_G128_FUSED_BLOCK_N,
        return_report=True,
        pair_q_heads=bool(Q6_G128_STAGE_F2_PAIR_REUSE),
    )
    expected_counts = torch.tensor(
        [sum(1 for o in offsets if i - o >= 0) for i in range(q.shape[2])],
        device=q.device,
        dtype=torch.int32,
    )
    if not torch.equal(valid_counts, expected_counts):
        raise RuntimeError('q6 fused direct-consume causal valid counts diverged')
    return out.float(), {
        **fused_report,
        'read_implementation': 'q6_triton_fused_direct_consume',
        'scratch_mode': 'direct_q6_decode_to_dsqg_output',
        'peak_scratch_bytes': 0,
        'peak_scratch_vs_full_scratch': 0.0,
        'gather_output_bytes': 0,
        'avoided_gather_bytes': int(fused_report.get('avoided_kv_gather_bytes', 0)),
        'attention_output_bytes': int(fused_report.get('attention_output_bytes', 0)),
        'lse_bytes': int(fused_report.get('lse_bytes', 0)),
        'materialized_gather_bytes': int(fused_report.get('materialized_gather_bytes', 0)),
        'compression_vs_bf16': float(fused_report.get('resident_q6_compression_vs_bf16', 0.0)),
        'stage_c_backward_tile_tokens': 0,
    }


class DSQGAttentionV19Q6G128Smoke(DSQGAttentionV19):
    """q6_g128 Triton direct-gather K/V smoke path for the base_v1 trainer clone.

    This is a correctness/stability integration spike, not the final fused
    consume/recompute implementation. It preserves the DSQG module parameters
    and MOVT semantics, decodes q6 resident K/V through the Phase-3 Triton
    direct-gather read path, and keeps BF16-gather STE gradients so q/k/v
    projections remain trainable. Expected outcome: real trainer CE/tok/s signal
    and scratch-read accounting, not a total production VRAM win yet.
    """

    q6_g128_smoke_path = True

    def __init__(self, *args, q6_layer_index=0, q6_seed=Q6_G128_SEED, **kwargs):
        super().__init__(*args, **kwargs)
        self.q6_layer_index = int(q6_layer_index)
        self.q6_seed = int(q6_seed) + 1009 * int(q6_layer_index)
        requested_kv_heads = int(Q6_G128_NUM_KV_HEADS)
        if requested_kv_heads == NUM_HEADS and self.num_heads != NUM_HEADS:
            requested_kv_heads = self.num_heads
        if self.num_heads % requested_kv_heads != 0:
            raise ValueError(f'q6 num_kv_heads={requested_kv_heads} must divide num_heads={self.num_heads}')
        self.num_kv_heads = requested_kv_heads
        self.kv_group_size = self.num_heads // self.num_kv_heads
        model_dim = self.num_heads * self.head_dim
        if self.num_kv_heads != self.num_heads:
            old_qkv = self.qkv_proj
            del self.qkv_proj
            self.q_proj = nn.Linear(model_dim, model_dim, bias=True)
            self.k_proj = nn.Linear(model_dim, self.num_kv_heads * self.head_dim, bias=True)
            self.v_proj = nn.Linear(model_dim, self.num_kv_heads * self.head_dim, bias=True)
            with torch.no_grad():
                q_w, k_w, v_w = old_qkv.weight.split(model_dim, dim=0)
                q_b, k_b, v_b = old_qkv.bias.split(model_dim, dim=0)
                self.q_proj.weight.copy_(q_w)
                self.q_proj.bias.copy_(q_b)
                k_w = k_w.view(self.num_heads, self.head_dim, model_dim)
                v_w = v_w.view(self.num_heads, self.head_dim, model_dim)
                k_b = k_b.view(self.num_heads, self.head_dim)
                v_b = v_b.view(self.num_heads, self.head_dim)
                k_w = k_w.view(self.num_kv_heads, self.kv_group_size, self.head_dim, model_dim).mean(dim=1)
                v_w = v_w.view(self.num_kv_heads, self.kv_group_size, self.head_dim, model_dim).mean(dim=1)
                k_b = k_b.view(self.num_kv_heads, self.kv_group_size, self.head_dim).mean(dim=1)
                v_b = v_b.view(self.num_kv_heads, self.kv_group_size, self.head_dim).mean(dim=1)
                self.k_proj.weight.copy_(k_w.reshape(self.num_kv_heads * self.head_dim, model_dim))
                self.v_proj.weight.copy_(v_w.reshape(self.num_kv_heads * self.head_dim, model_dim))
                self.k_proj.bias.copy_(k_b.reshape(self.num_kv_heads * self.head_dim))
                self.v_proj.bias.copy_(v_b.reshape(self.num_kv_heads * self.head_dim))
        self._q6_last_report = {}

    def _rotate_sparse_values(self, values, y_pre, z_pre, idx, valid):
        if self.j_large <= 0:
            return values
        out = values.clone()
        gate = torch.sigmoid(self.phase_gate).float()[:, None, None]
        gated_phase_base = (self.phase_base.float() * gate)
        gated_phase_gain = (self.phase_gain.float() * gate)
        hd_segment = self.head_dim // R_PLANES
        # idx/valid are [N,J]; z_pre gather by each offset is cheap at smoke scale.
        for i in range(self.j_small, self.j_val):
            pi = i - self.j_small
            kp = idx[:, i]
            val_i = valid[:, i]
            for r in range(R_PLANES):
                ch_a = r * hd_segment + self.plane_shift
                ch_b = ch_a + 1
                z_i = z_pre[:, :, kp, r]
                theta = (gated_phase_base[pi, :, r].reshape(1, -1, 1)
                         + gated_phase_gain[pi, :, r].reshape(1, -1, 1) * y_pre[:, :, :, r].float() * z_i.float())
                theta = torch.where(val_i.reshape(1, 1, -1), theta, torch.zeros_like(theta))
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                old_a = out[:, :, :, i, ch_a].clone()
                old_b = out[:, :, :, i, ch_b].clone()
                out[:, :, :, i, ch_a] = cos_t * old_a - sin_t * old_b
                out[:, :, :, i, ch_b] = sin_t * old_a + cos_t * old_b
        return out

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim

        HKV = int(getattr(self, 'num_kv_heads', H))
        kv_group = int(getattr(self, 'kv_group_size', 1))
        if HKV == H:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.split(D, dim=-1)
            q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
            k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
            v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        else:
            q = self.q_proj(x).view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
            k = self.k_proj(x).view(B, N, HKV, HD).permute(0, 2, 1, 3).contiguous()
            v = self.v_proj(x).view(B, N, HKV, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            theta_k = NPCI_THETA_MAX * torch.tanh(self.npci_theta_k)
            theta_v = NPCI_THETA_MAX * torch.tanh(self.npci_theta_v)
            if HKV != H:
                k_delta = k_delta.view(B, HKV, kv_group, N, HD).mean(dim=2).contiguous()
                v_delta = v_delta.view(B, HKV, kv_group, N, HD).mean(dim=2).contiguous()
                theta_k = theta_k.view(HKV, kv_group).mean(dim=1)
                theta_v = theta_v.view(HKV, kv_group).mean(dim=1)
            k = npci_rotate(k, k_delta, theta_k).to(dtype=q.dtype).contiguous()
            v = npci_rotate(v, v_delta, theta_v).to(dtype=q.dtype).contiguous()

        q_norm = _rms_normalize_last(q)
        k_norm = _rms_normalize_last(k)
        qp_norm = F.normalize(self.query_probes.float(), dim=-1)
        kp_norm = F.normalize(self.key_probes.float(), dim=-1)
        probe_scale = 1.0 / math.sqrt(float(HD))
        y_pre = (torch.einsum('bhnd,rd->bhnr', q_norm, qp_norm) * probe_scale).contiguous()
        z_kv_pre = (torch.einsum('bhnd,rd->bhnr', k_norm, kp_norm) * probe_scale).contiguous()
        z_pre = z_kv_pre if HKV == H else z_kv_pre.repeat_interleave(kv_group, dim=1).contiguous()

        offsets = [int(v) for v in self.offsets_dev.detach().cpu().tolist()]
        if Q6_G128_FUSED_CONSUME:
            out, fused_report = _q6_g128_triton_fused_direct_consume(
                q, k, v, offsets, self, y_pre, z_pre, seed=self.q6_seed
            )
            out = out.to(dtype=q.dtype) * self.if_gain.view(1, H, 1, 1)
            out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
            gate = torch.sigmoid(self.gate_proj(x))
            self._q6_last_report = {
                'k_resident_q6_bytes': int(fused_report['resident_q6_bytes'] // 2),
                'v_resident_q6_bytes': int(fused_report['resident_q6_bytes'] - fused_report['resident_q6_bytes'] // 2),
                'resident_q6_bytes': int(fused_report['resident_q6_bytes']),
                'forward_values': int(B * H * N * HD * 2),
                'compression_vs_bf16': float(fused_report['compression_vs_bf16']),
                'num_query_heads': int(H),
                'num_kv_heads': int(HKV),
                'kv_group_size': int(kv_group),
                'read_implementation': fused_report['read_implementation'],
                'scratch_mode': fused_report['scratch_mode'],
                'peak_scratch_bytes': int(fused_report['peak_scratch_bytes']),
                'gather_output_bytes': int(fused_report['gather_output_bytes']),
                'materialized_gather_bytes': int(fused_report['materialized_gather_bytes']),
                'avoided_gather_bytes': int(fused_report['avoided_gather_bytes']),
                'attention_output_bytes': int(fused_report['attention_output_bytes']),
                'lse_bytes': int(fused_report['lse_bytes']),
                'fused_block_n': int(fused_report.get('block_n', Q6_G128_FUSED_BLOCK_N)),
                'stage_c_backward_tile_tokens': int(fused_report.get('stage_c_backward_tile_tokens', 0)),
                'stage_c_backward_replay_tiles': int(fused_report.get('stage_c_backward_replay_tiles', 0)),
                'stage_c_backward_tile_gather_bytes': int(fused_report.get('stage_c_backward_tile_gather_bytes', 0)),
                'stage_c_backward_tile_gather_vs_full': float(fused_report.get('stage_c_backward_tile_gather_vs_full', 0.0)),
                'stage_d_backward_enabled': bool(fused_report.get('stage_d_backward_enabled', False)),
                'stage_d_sequence_scatter': str(fused_report.get('stage_d_sequence_scatter', 'none')),
                'stage_d_tile_decode': str(fused_report.get('stage_d_tile_decode', 'none')),
                'stage_e_fused_backward_enabled': bool(fused_report.get('stage_e_fused_backward_enabled', False)),
                'stage_e_backward_core': str(fused_report.get('stage_e_backward_core', 'disabled')),
                'stage_f2_pair_forward_enabled': bool(fused_report.get('stage_f2_pair_forward_enabled', False)),
                'stage_f2_pair_forward_core': str(fused_report.get('stage_f2_pair_forward_core', 'disabled')),
                'stage_f3_pair_backward_enabled': bool(fused_report.get('stage_f3_pair_backward_enabled', False)),
                'stage_f3_pair_backward_core': str(fused_report.get('stage_f3_pair_backward_core', 'disabled')),
                'stage_f3_split_movt_prob_scratch_enabled': bool(fused_report.get('stage_f3_split_movt_prob_scratch_enabled', False)),
                'stage_f3_sidecar_pair_direct_enabled': bool(fused_report.get('stage_f3_sidecar_pair_direct_enabled', False)),
            }
            return self.dropout(self.out_proj(out_flat * gate))

        idx, valid = _q6_layout_mod.causal_offset_index(N, offsets, device=q.device)
        k_g, mask, k_report = _q6_g128_triton_direct_ste_causal_gather(k, offsets, seed=self.q6_seed)
        v_g, mask_v, v_report = _q6_g128_triton_direct_ste_causal_gather(v, offsets, seed=self.q6_seed + 1)
        if not torch.equal(mask, mask_v):
            raise RuntimeError('q6 K/V causal masks diverged')
        if HKV != H:
            k_g = k_g.repeat_interleave(kv_group, dim=1).contiguous()
            v_g = v_g.repeat_interleave(kv_group, dim=1).contiguous()

        qf = q.float()
        sc = 1.0 / math.sqrt(float(HD))
        scores = torch.einsum('bhnd,bhnjd->bhnj', qf, k_g.float()) * sc
        scores = scores + torch.einsum('bhnd,jd->bhnj', qf, self.scale_embed.float()) * sc
        scores = scores + (self.pos_bias.float() * self.pos_bias_scale.to(device=self.pos_bias.device).float()).transpose(0, 1).reshape(1, H, 1, self.j_val)
        scores = scores.masked_fill(~mask.reshape(1, 1, N, self.j_val), float('-inf'))

        max_scores = scores.amax(dim=-1, keepdim=True)
        all_invalid = ~torch.isfinite(max_scores)
        safe_max = torch.where(all_invalid, torch.zeros_like(max_scores), max_scores)
        exp_scores = torch.exp(scores - safe_max).masked_fill(~mask.reshape(1, 1, N, self.j_val), 0.0)
        denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        probs = exp_scores / denom

        v_rot = self._rotate_sparse_values(v_g.float(), y_pre, z_pre, idx, mask)
        out = torch.sum(probs.unsqueeze(-1) * v_rot, dim=3).to(dtype=q.dtype)
        out = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        q6_resident_bytes = int(k_report['total_bytes'] + v_report['total_bytes'])
        bf16_kv_baseline_bytes = int(2 * B * H * N * HD * torch.tensor([], dtype=torch.bfloat16).element_size())
        gather_bytes = int(k_report['output_bytes'] + v_report['output_bytes'])
        if HKV != H:
            gather_bytes *= kv_group
        self._q6_last_report = {
            'k_resident_q6_bytes': int(k_report['total_bytes']),
            'v_resident_q6_bytes': int(v_report['total_bytes']),
            'resident_q6_bytes': q6_resident_bytes,
            'forward_values': int(2 * B * H * N * HD),
            'compression_vs_bf16': float(bf16_kv_baseline_bytes / q6_resident_bytes),
            'num_query_heads': int(H),
            'num_kv_heads': int(HKV),
            'kv_group_size': int(kv_group),
            'read_implementation': 'q6_gqa_direct_gather_reference' if HKV != H else k_report['read_implementation'],
            'scratch_mode': k_report['scratch_mode'],
            'peak_scratch_bytes': int(k_report['peak_scratch_bytes'] + v_report['peak_scratch_bytes']),
            'gather_output_bytes': gather_bytes,
            'materialized_gather_bytes': gather_bytes,
        }
        return self.dropout(self.out_proj(out_flat * gate))


_DSQG_TYPES = (DSQGAttentionV19, DSQGAttentionV19Q6G128Smoke)


class FFN(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlockTriadic(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 offsets, j_small, j_large, group_label,
                 dropout=0.1, interference=False, plane_shift=0, layer_index=-1):
        super().__init__()
        self.interference = interference
        self.group_label = group_label
        self.layer_index = int(layer_index)
        self.plane_shift = int(plane_shift)
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.norm1 = _LayerNorm(embedding_dim)
        self.norm2 = _LayerNorm(embedding_dim)
        attn_cls = DSQGAttentionV19Q6G128Smoke if _q6_enabled_for_layer(self.layer_index) else DSQGAttentionV19
        attn_kwargs = {'q6_layer_index': self.layer_index} if attn_cls is DSQGAttentionV19Q6G128Smoke else {}
        self.attn = attn_cls(
            embedding_dim, num_heads, offsets, j_small, j_large,
            seq_len=seq_len, dropout=dropout, plane_shift=self.plane_shift,
            **attn_kwargs)
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm = _LayerNorm(embedding_dim)
            self.inter_gate = nn.Linear(embedding_dim, embedding_dim)
            self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
            self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)
            self.ema_factor = nn.Parameter(torch.full((1,), EMA_INIT))

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD = self.num_heads, self.head_dim
            pool = _causal_ema(xi, self.ema_factor.abs() + EMA_FLOOR, floor=EMA_FLOOR)
            pool = _agc_normalize(pool)
            inter = torch.sigmoid(self.inter_gate(xi)) * pool
            k_delta = (self.inter_k_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            v_delta = (self.inter_v_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            kv_inject = (k_delta, v_delta)
        x = x + self.attn(self.norm1(x), kv_inject=kv_inject)
        x = x + self.ffn(self.norm2(x))
        return x


class DSRBlock(nn.Module):
    """DSR block: selected HISA implementation + FFN + output gate."""
    def __init__(self, embedding_dim, num_heads, ffn_dim, head_dim,
                 num_chunks, top_k_chunks, dropout=0.1, hisa_top_m_tokens=32):
        super().__init__()
        self.norm1 = _LayerNorm(embedding_dim)
        self.norm2 = _LayerNorm(embedding_dim)
        self.attn = HISA_IMPL_CLS(
            D=embedding_dim, H=num_heads, hd=head_dim,
            num_chunks=num_chunks, top_k_chunks=top_k_chunks,
            hisa_top_m_tokens=hisa_top_m_tokens,
        )
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim)
        self.ffn = FFN(embedding_dim, ffn_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        gate = torch.sigmoid(self.gate_proj(x))
        x = x + self.drop(attn_out * gate)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# MODEL
# =============================================================================

class TriadicJ96Dsr(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, ffn_dim, seq_len,
                 dsr_layer, scale_embed_init_val=0.15, dropout=0.1,
                 num_chunks=32, top_k_chunks=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.dsr_layer = dsr_layer
        self.head_dim = embedding_dim // num_heads

        blocks = []
        dsqg_idx = 0
        for i, (label, offsets, js, jl, has_if) in enumerate(LAYER_LAYOUT):
            if label == 'DSR':
                blocks.append(DSRBlock(
                    embedding_dim, num_heads, ffn_dim,
                    self.head_dim, num_chunks, top_k_chunks, dropout,
                    hisa_top_m_tokens=HISA_TOP_M_TOKENS))
            else:
                plane_shift = (_movt_plane_shift_for_dsqg_index(dsqg_idx, self.head_dim, R_PLANES)
                               if STAGGER_MOVT_PLANES else 0)
                blocks.append(DSQGBlockTriadic(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    offsets, js, jl, group_label=label,
                    dropout=dropout, interference=has_if,
                    plane_shift=plane_shift, layer_index=i))
                dsqg_idx += 1
        self.blocks = nn.ModuleList(blocks)
        self.norm = _LayerNorm(embedding_dim)
        self.out = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight
        self.dsqg_w_enabled = bool(DSQG_W_ENABLED)
        self.dsqg_w_sourcewise_enabled = bool(DSQG_W_SOURCEWISE)
        self.dsqg_w_config = None
        self.dsqg_w_candidate_provider = None
        self.dsqg_w = None
        self.dsqg_w_blocks = nn.ModuleDict()
        self.dsqg_w_site_specs = tuple(DSQG_W_SITE_SPECS)
        self.dsqg_w_site_keys = tuple(_dsqg_w_site_key(site) for site in self.dsqg_w_site_specs)
        self.dsqg_w_layer_site_map = {
            int(site): _dsqg_w_site_key(site)
            for site in self.dsqg_w_site_specs
            if site != 'final'
        }
        self.dsqg_w_has_final_site = 'final' in self.dsqg_w_site_specs
        self.dsqg_w_last_telemetry = {}
        self._dsqg_w_forward_counter = 0
        self._dsqg_w_active_site_key = None
        self._dsqg_w_metadata_cache = {}
        self._init_weights(scale_embed_init_val)
        if self.dsqg_w_enabled:
            self.dsqg_w_config = DSQGWConfig(
                d=embedding_dim,
                n_heads=num_heads,
                bottleneck=DSQG_W_BOTTLENECK,
                max_candidates=DSQG_W_MAX_CANDIDATES,
                gate_init=DSQG_W_GATE_INIT,
                fuse_init_std=DSQG_W_FUSE_INIT_STD,
                local_offsets=DSQG_W_LOCAL_OFFSETS,
                long_offsets=DSQG_W_LONG_OFFSETS,
                # QUESTION/cue candidates are opt-in so the identity path can
                # remain LOCAL/LONG/NULL until explicitly audited.
                k_question=DSQG_W_K_QUESTION,
                k_hisa_evidence=DSQG_W_K_HISA_EVIDENCE,
                k_chunk=0,
                k_l3_skip=(
                    1
                    if (
                        DSQG_W_FAST_EVIDENCE_MEAN
                        and os.getenv('DWARF_DSQG_W_ALLOW_FAST_EVIDENCE_MEAN_BYPASS', '0') != '1'
                        and DSQG_W_K_L3_SKIP <= 0
                    )
                    else DSQG_W_K_L3_SKIP
                ),
                use_width_cell=DSQG_W_WIDTH_CELL,
                width_bottleneck=DSQG_W_WIDTH_BOTTLENECK,
                width_gate_init=DSQG_W_WIDTH_GATE_INIT,
                width_entropy_floor=DSQG_W_WIDTH_ENTROPY_FLOOR,
                width_entropy_weight=DSQG_W_WIDTH_ENTROPY_WEIGHT,
                use_typed_mixer=DSQG_W_TYPED_MIXER,
                typed_mixer_bottleneck=DSQG_W_TYPED_MIXER_BOTTLENECK,
                typed_mixer_gate_init=DSQG_W_TYPED_MIXER_GATE_INIT,
                use_query_type_bias=DSQG_W_QUERY_TYPE_BIAS,
                typed_hisa_reps=DSQG_W_TYPED_HISA_REPS,
                use_evidence_binding_hub=DSQG_W_EVIDENCE_BINDING_HUB,
                ebh_bottleneck=DSQG_W_EBH_BOTTLENECK,
                ebh_gate_init=DSQG_W_EBH_GATE_INIT,
                ebh_phase_bands=DSQG_W_EBH_PHASE_BANDS,
                ebh_score_features=DSQG_W_EBH_SCORE_FEATURES,
                ebh_pair_mixer=DSQG_W_EBH_PAIR_MIXER,
                ebh_pair_rank=DSQG_W_EBH_PAIR_RANK,
                ebh_pair_gate_init=DSQG_W_EBH_PAIR_GATE_INIT,
                use_evidence_prior=DSQG_W_EVIDENCE_PRIOR,
                evidence_prior_clip=DSQG_W_EVIDENCE_PRIOR_CLIP,
                evidence_prior_init_scale=DSQG_W_EVIDENCE_PRIOR_INIT_SCALE,
                use_candidate_quotas=DSQG_W_CANDIDATE_QUOTAS,
                quota_hisa_max=DSQG_W_QUOTA_HISA_MAX,
                use_candidate_workspace=DSQG_W_CANDIDATE_WORKSPACE,
                candidate_workspace_dim=DSQG_W_CANDIDATE_WORKSPACE_DIM,
                candidate_workspace_phase_bands=DSQG_W_CANDIDATE_WORKSPACE_PHASE_BANDS,
                candidate_workspace_score_features=DSQG_W_CANDIDATE_WORKSPACE_SCORE_FEATURES,
                candidate_workspace_query_scores=DSQG_W_CANDIDATE_WORKSPACE_QUERY_SCORES,
                candidate_workspace_pair_transfer=DSQG_W_CANDIDATE_WORKSPACE_PAIR_TRANSFER,
                candidate_workspace_pair_gate_init=DSQG_W_CANDIDATE_WORKSPACE_PAIR_GATE_INIT,
            )
            self.dsqg_w_candidate_provider = CandidateProvider(self.dsqg_w_config)
            for site_key in self.dsqg_w_site_keys:
                self.dsqg_w_blocks[site_key] = DSQGWBlock.from_config(self.dsqg_w_config)
            if self.dsqg_w_has_final_site:
                self.dsqg_w = self.dsqg_w_blocks['final']
            elif self.dsqg_w_site_keys:
                self.dsqg_w = self.dsqg_w_blocks[self.dsqg_w_site_keys[0]]

    def _init_weights(self, scale_embed_init_val):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        for m in self.modules():
            if hasattr(m, 'gate_proj') and isinstance(m.gate_proj, nn.Linear):
                nn.init.constant_(m.gate_proj.bias, 0.0)
        global _DSQG_TYPES
        for m in self.modules():
            if isinstance(m, _DSQG_TYPES):
                nn.init.normal_(m.phase_base, 0.0, 0.01)
                if hasattr(m, 'reset_phase_probes_'):
                    m.reset_phase_probes_()
                nn.init.normal_(m.phase_gain, 0.0, 0.001)
                nn.init.zeros_(m.phase_gate)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def _should_checkpoint_block(self, block_idx):
        if CHECKPOINT_STRATEGY == 'none':
            return False
        if CHECKPOINT_STRATEGY == 'all':
            return True
        if CHECKPOINT_STRATEGY == 'every_other':
            return block_idx % 2 == 0
        if CHECKPOINT_STRATEGY == 'full_attn':
            return block_idx == self.dsr_layer
        return False

    def _ckpt(self, block, x):
        if _SAC_AVAILABLE:
            return grad_ckpt(block, x, use_reentrant=False,
                             context_fn=partial(create_selective_checkpoint_contexts,
                                                _sac_policy_fn))
        return grad_ckpt(block, x, use_reentrant=False)

    def _begin_dsqg_w_forward(self):
        if not self.dsqg_w_enabled or not self.training:
            self._dsqg_w_active_site_key = None
            return
        mode = DSQG_W_ACTIVE_SITE_MODE
        if mode in ('', 'all', 'always'):
            self._dsqg_w_active_site_key = None
            return
        if mode in ('cycle', 'round_robin', 'round-robin'):
            site_count = max(1, len(self.dsqg_w_site_keys))
            active_idx = int(self._dsqg_w_forward_counter) % site_count
            self._dsqg_w_active_site_key = self.dsqg_w_site_keys[active_idx]
            self._dsqg_w_forward_counter += 1
            return
        if mode in self.dsqg_w_site_keys:
            self._dsqg_w_active_site_key = mode
            return
        raise ValueError(
            'DWARF_DSQG_W_ACTIVE_SITE_MODE must be all, cycle, round_robin, '
            f'or one of {self.dsqg_w_site_keys}; got {mode!r}'
        )

    def _dsqg_w_site_active(self, site_key):
        active_site = self._dsqg_w_active_site_key
        return active_site is None or str(site_key) == str(active_site)

    def _forward_trunk(
        self,
        idx,
        *,
        collect_l3_state=False,
        dsqg_w_question_indices=None,
        dsqg_w_hisa_evidence_indices=None,
        dsqg_w_hisa_evidence_scores=None,
        dsqg_w_l3_skip_indices=None,
    ):
        x = self.drop(self.embedding(idx))
        l3_state = None
        dsr_hisa_evidence_indices = None
        dsr_hisa_evidence_scores = None
        for i, block in enumerate(self.blocks):
            if self.training and self._should_checkpoint_block(i):
                x = self._ckpt(block, x)
            else:
                x = block(x)
            if i == self.dsr_layer:
                dsr_hisa_evidence_indices, dsr_hisa_evidence_scores = self._dsqg_w_dsr_selected_candidates(idx.shape[1])
            if collect_l3_state and i == self.dsr_layer:
                l3_state = x
            if (
                self.dsqg_w_enabled
                and i in self.dsqg_w_layer_site_map
                and self._dsqg_w_site_active(self.dsqg_w_layer_site_map[i])
            ):
                x = self._apply_dsqg_w_recomposer(
                    x,
                    site_key=self.dsqg_w_layer_site_map[i],
                    question_indices=dsqg_w_question_indices,
                    l3_states=l3_state,
                    hisa_evidence_indices=(
                        dsr_hisa_evidence_indices
                        if dsr_hisa_evidence_indices is not None
                        else dsqg_w_hisa_evidence_indices
                    ),
                    hisa_evidence_scores=(
                        dsr_hisa_evidence_scores
                        if dsr_hisa_evidence_scores is not None
                        else dsqg_w_hisa_evidence_scores
                    ),
                    l3_skip_indices=dsqg_w_l3_skip_indices,
                )
        if collect_l3_state:
            return (
                x,
                l3_state if l3_state is not None else x,
                dsr_hisa_evidence_indices,
                dsr_hisa_evidence_scores,
            )
        return x

    def _dsqg_w_dsr_selected_candidates(self, seq_len):
        if (
            not self.dsqg_w_enabled
            or not DSQG_W_HISA_L3_ENABLED
            or not DSQG_W_DSR_CANDIDATES
            or DSQG_W_K_HISA_EVIDENCE <= 0
        ):
            return None, None
        if _pack_hisa_selected_tokens_for_dsqg_w is None:
            return None, None
        if not (0 <= int(self.dsr_layer) < len(self.blocks)):
            return None, None
        dsr_block = self.blocks[int(self.dsr_layer)]
        attn = getattr(dsr_block, 'attn', None)
        token_idx = getattr(attn, '_last_token_idx_packed', None)
        token_scores = getattr(attn, '_last_token_scores_packed', None)
        chunk_size = getattr(attn, '_last_chunk_size', None)
        if token_idx is None or token_scores is None or chunk_size is None:
            return None, None
        return _pack_hisa_selected_tokens_for_dsqg_w(
            token_idx,
            token_scores,
            seq_len=int(seq_len),
            chunk_size=int(chunk_size),
            max_candidates=int(DSQG_W_K_HISA_EVIDENCE),
        )

    @staticmethod
    def _dsqg_w_tensor_cache_key(tensor):
        if tensor is None:
            return None
        return (
            int(tensor.data_ptr()),
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
        )

    def _dsqg_w_metadata_cache_key(
        self,
        x,
        *,
        question_indices=None,
        hisa_evidence_indices=None,
        hisa_evidence_scores=None,
        l3_skip_indices=None,
    ):
        return (
            tuple(x.shape),
            str(x.dtype),
            str(x.device),
            self._dsqg_w_tensor_cache_key(question_indices),
            self._dsqg_w_tensor_cache_key(hisa_evidence_indices),
            self._dsqg_w_tensor_cache_key(hisa_evidence_scores),
            self._dsqg_w_tensor_cache_key(l3_skip_indices),
        )

    def _get_or_build_dsqg_w_metadata(
        self,
        x,
        *,
        l3_states=None,
        question_indices=None,
        hisa_evidence_indices=None,
        hisa_evidence_scores=None,
        l3_skip_indices=None,
    ):
        key = self._dsqg_w_metadata_cache_key(
            x,
            question_indices=question_indices,
            hisa_evidence_indices=hisa_evidence_indices,
            hisa_evidence_scores=hisa_evidence_scores,
            l3_skip_indices=l3_skip_indices,
        )
        cached = self._dsqg_w_metadata_cache.get(key)
        if cached is not None:
            return cached, True
        with _profile_range('dsqg_w/candidate_metadata_build'):
            candidates = self.dsqg_w_candidate_provider.build_metadata(
                x,
                l3_states=l3_states,
                question_indices=question_indices,
                hisa_evidence_indices=hisa_evidence_indices,
                hisa_evidence_scores=hisa_evidence_scores,
                l3_skip_indices=l3_skip_indices,
            )
        self._dsqg_w_metadata_cache[key] = candidates
        return candidates, False

    @staticmethod
    def _dsqg_w_expand_candidate_indices(indices, *, bsz, seq_len, device):
        if indices is None:
            return None
        values = indices.to(device=device, dtype=torch.long)
        if values.ndim == 1:
            values = values.reshape(1, 1, -1).expand(bsz, seq_len, -1)
        elif values.ndim == 2:
            values = values.reshape(values.shape[0], 1, values.shape[1]).expand(-1, seq_len, -1)
        elif values.ndim != 3:
            raise ValueError('DSQG-W fast evidence indices must be rank 1, 2, or 3')
        if values.shape[0] == 1 and bsz != 1:
            values = values.expand(bsz, -1, -1)
        if values.shape[1] == 1 and seq_len != 1:
            values = values.expand(-1, seq_len, -1)
        if values.shape[:2] != (bsz, seq_len):
            raise ValueError('DSQG-W fast evidence indices must broadcast to [B,T,K]')
        return values

    @staticmethod
    def _dsqg_w_gather_fast_evidence(states, indices, positions):
        if indices is None or indices.shape[-1] == 0:
            return None, None
        bsz, seq_len = states.shape[:2]
        valid = (indices >= 0) & (indices <= positions)
        gather_tokens = indices.clamp(0, max(seq_len - 1, 0))
        batch_offsets = torch.arange(bsz, device=states.device, dtype=torch.long).reshape(bsz, 1, 1) * seq_len
        flat_indices = (batch_offsets + gather_tokens).reshape(-1)
        gathered = states.reshape(bsz * seq_len, states.shape[-1]).index_select(0, flat_indices)
        gathered = gathered.reshape(bsz, seq_len, indices.shape[-1], states.shape[-1])
        return gathered * valid[..., None].to(gathered.dtype), valid

    def _apply_dsqg_w_fast_evidence_mean(
        self,
        x,
        *,
        block,
        l3_states=None,
        question_indices=None,
        hisa_evidence_indices=None,
        l3_skip_indices=None,
    ):
        bsz, seq_len, d = x.shape
        device = x.device
        positions = torch.arange(seq_len, device=device, dtype=torch.long).reshape(1, seq_len, 1).expand(bsz, -1, -1)
        final_base = x
        l3_base = l3_states if l3_states is not None else x
        groups = []
        masks = []
        for states, indices in (
            (final_base, question_indices),
            (l3_base, hisa_evidence_indices),
            (l3_base, l3_skip_indices),
        ):
            expanded = self._dsqg_w_expand_candidate_indices(indices, bsz=bsz, seq_len=seq_len, device=device)
            gathered, valid = self._dsqg_w_gather_fast_evidence(states, expanded, positions) if expanded is not None else (None, None)
            if gathered is not None and valid is not None:
                groups.append(gathered)
                masks.append(valid)
        if groups:
            evidence = torch.cat(groups, dim=2)
            evidence_mask = torch.cat(masks, dim=2)
            count = evidence_mask.sum(dim=2, keepdim=True).clamp_min(1).to(evidence.dtype)
            read = evidence.sum(dim=2) / count
            slot_count = float(evidence.shape[2])
            valid_mean = evidence_mask.sum(dim=2).float().mean()
        elif l3_states is not None:
            read = l3_base
            slot_count = 1.0
            valid_mean = x.new_tensor(1.0)
        else:
            read = x
            slot_count = 0.0
            valid_mean = x.new_tensor(0.0)
        gate = torch.sigmoid(block.gate).reshape(1, 1, d)
        out = x + gate * (read - x)
        telemetry = {
            'dsqg_w_fast_evidence_mean': x.new_tensor(1.0).detach(),
            'dsqg_w_valid_candidate_count': valid_mean.detach(),
            'dsqg_w_candidate_slot_count': x.new_tensor(slot_count).detach(),
            'dsqg_w_gate_mean': gate.mean().detach(),
            'dsqg_w_delta_norm': (read - x).norm(dim=-1).mean().detach(),
            'dsqg_w_x_norm': x.norm(dim=-1).mean().detach(),
            'dsqg_w_delta_to_x_ratio': ((read - x).norm(dim=-1).mean() / x.norm(dim=-1).mean().clamp_min(1e-8)).detach(),
            'dsqg_w_read_norm': read.norm(dim=-1).mean().detach(),
        }
        return out, telemetry

    def _apply_dsqg_w_recomposer(
        self,
        x,
        *,
        site_key='final',
        question_indices=None,
        l3_states=None,
        hisa_evidence_indices=None,
        hisa_evidence_scores=None,
        l3_skip_indices=None,
    ):
        if not self.dsqg_w_enabled:
            self.dsqg_w_last_telemetry = {}
            return x
        if self.dsqg_w is None or self.dsqg_w_candidate_provider is None:
            raise RuntimeError('DSQG-W is enabled but its block/provider was not initialized')
        if site_key not in self.dsqg_w_blocks:
            return x
        block = self.dsqg_w_blocks[site_key]
        allow_fast_mean_bypass = os.getenv('DWARF_DSQG_W_ALLOW_FAST_EVIDENCE_MEAN_BYPASS', '0') == '1'
        force_trainable_candidate_path = DSQG_W_FAST_EVIDENCE_MEAN and not allow_fast_mean_bypass
        if force_trainable_candidate_path and l3_states is not None and l3_skip_indices is None and self.dsqg_w_config.k_l3_skip > 0:
            seq_len = int(x.shape[1])
            l3_skip_indices = torch.arange(seq_len, device=x.device, dtype=torch.long).reshape(1, seq_len, 1)
        effective_detach_recomposer = DSQG_W_DETACH_RECOMPOSER and not force_trainable_candidate_path
        recomposer_x = x.detach() if effective_detach_recomposer else x
        recomposer_l3_states = l3_states.detach() if effective_detach_recomposer and l3_states is not None else l3_states
        grad_context = torch.no_grad() if effective_detach_recomposer else contextlib.nullcontext()
        if DSQG_W_FAST_EVIDENCE_MEAN and allow_fast_mean_bypass:
            with _profile_range(f'dsqg_w/site={site_key}'):
                with grad_context:
                    x_out, telemetry = self._apply_dsqg_w_fast_evidence_mean(
                        recomposer_x,
                        block=block,
                        l3_states=recomposer_l3_states,
                        question_indices=question_indices,
                        hisa_evidence_indices=hisa_evidence_indices,
                        l3_skip_indices=l3_skip_indices,
                    )
            merged_telemetry = dict(telemetry)
            merged_telemetry['dsqg_w_site_key'] = site_key
            merged_telemetry['dsqg_w_metadata_cache_hit'] = x.new_tensor(0.0).detach()
            merged_telemetry['dsqg_w_detached_recomposer'] = x.new_tensor(1.0 if effective_detach_recomposer else 0.0).detach()
            merged_telemetry['dsqg_w_active_site_cycle'] = x.new_tensor(1.0 if self._dsqg_w_active_site_key is not None else 0.0).detach()
            merged_telemetry['dsqg_w_fast_evidence_mean_bypass'] = x.new_tensor(1.0).detach()
            self.dsqg_w_last_telemetry = merged_telemetry
            if effective_detach_recomposer:
                return x + (x_out - recomposer_x).detach()
            return x_out
        with _profile_range(f'dsqg_w/site={site_key}'):
            with grad_context:
                if self.dsqg_w_sourcewise_enabled:
                    candidates, metadata_cache_hit = self._get_or_build_dsqg_w_metadata(
                        recomposer_x,
                        l3_states=recomposer_l3_states,
                        question_indices=question_indices,
                        hisa_evidence_indices=hisa_evidence_indices,
                        hisa_evidence_scores=hisa_evidence_scores,
                        l3_skip_indices=l3_skip_indices,
                    )
                    x_out, telemetry = block.forward_sourcewise(
                        recomposer_x,
                        candidates.cand_token_indices,
                        candidates.cand_types,
                        candidates.cand_sources,
                        candidates.cand_mask,
                        l3_states=recomposer_l3_states,
                        cand_scores=candidates.cand_scores,
                        evidence_bits=candidates.evidence_bits,
                        evidence_count=candidates.evidence_count,
                        candidate_distances=candidates.candidate_distances,
                        needed_source_ids=candidates.active_source_ids,
                    )
                else:
                    with _profile_range('dsqg_w/candidate_materialized_build'):
                        candidates = self.dsqg_w_candidate_provider.build(
                            recomposer_x,
                            l3_states=recomposer_l3_states,
                            question_indices=question_indices,
                            hisa_evidence_indices=hisa_evidence_indices,
                            hisa_evidence_scores=hisa_evidence_scores,
                            l3_skip_indices=l3_skip_indices,
                        )
                    metadata_cache_hit = False
                    x_out, telemetry = block(
                        recomposer_x,
                        candidates.cand_states,
                        candidates.cand_types,
                        candidates.cand_sources,
                        candidates.cand_mask,
                        cand_scores=candidates.cand_scores,
                        evidence_bits=candidates.evidence_bits,
                        evidence_count=candidates.evidence_count,
                        candidate_distances=candidates.candidate_distances,
                    )
        merged_telemetry = dict(candidates.telemetry)
        merged_telemetry.update(telemetry)
        merged_telemetry['dsqg_w_site_key'] = site_key
        merged_telemetry['dsqg_w_metadata_cache_hit'] = x.new_tensor(1.0 if metadata_cache_hit else 0.0).detach()
        merged_telemetry['dsqg_w_detached_recomposer'] = x.new_tensor(1.0 if effective_detach_recomposer else 0.0).detach()
        merged_telemetry['dsqg_w_active_site_cycle'] = x.new_tensor(1.0 if self._dsqg_w_active_site_key is not None else 0.0).detach()
        merged_telemetry['dsqg_w_fast_evidence_mean_requested'] = x.new_tensor(1.0 if DSQG_W_FAST_EVIDENCE_MEAN else 0.0).detach()
        merged_telemetry['dsqg_w_fast_evidence_mean_bypass'] = x.new_tensor(0.0).detach()
        merged_telemetry['dsqg_w_force_trainable_candidate_path'] = x.new_tensor(1.0 if force_trainable_candidate_path else 0.0).detach()
        self.dsqg_w_last_telemetry = merged_telemetry
        if effective_detach_recomposer:
            return x + (x_out - recomposer_x).detach()
        return x_out

    def forward(
        self,
        idx,
        *,
        dsqg_w_question_indices=None,
        dsqg_w_hisa_evidence_indices=None,
        dsqg_w_l3_skip_indices=None,
    ):
        self._dsqg_w_metadata_cache = {}
        self._begin_dsqg_w_forward()
        if self.dsqg_w_enabled:
            trunk_out, l3_state, dsr_hisa_indices, dsr_hisa_scores = self._forward_trunk(
                idx,
                collect_l3_state=True,
                dsqg_w_question_indices=dsqg_w_question_indices,
                dsqg_w_hisa_evidence_indices=dsqg_w_hisa_evidence_indices,
                dsqg_w_l3_skip_indices=dsqg_w_l3_skip_indices,
            )
        else:
            trunk_out = self._forward_trunk(idx)
            l3_state = None
        x = trunk_out
        if self.dsqg_w_enabled and self.dsqg_w_has_final_site and self._dsqg_w_site_active('final'):
            x = self._apply_dsqg_w_recomposer(
                trunk_out,
                site_key='final',
                question_indices=dsqg_w_question_indices,
                l3_states=l3_state,
                hisa_evidence_indices=(
                    dsr_hisa_indices if dsr_hisa_indices is not None else dsqg_w_hisa_evidence_indices
                ),
                hisa_evidence_scores=(
                    dsr_hisa_scores if dsr_hisa_scores is not None else None
                ),
                l3_skip_indices=dsqg_w_l3_skip_indices,
            )
        return self.out(self.norm(x))

    def forward_hidden(
        self,
        idx,
        *,
        dsqg_w_question_indices=None,
        dsqg_w_hisa_evidence_indices=None,
        dsqg_w_l3_skip_indices=None,
    ):
        self._dsqg_w_metadata_cache = {}
        self._begin_dsqg_w_forward()
        if self.dsqg_w_enabled:
            trunk_out, l3_state, dsr_hisa_indices, dsr_hisa_scores = self._forward_trunk(
                idx,
                collect_l3_state=True,
                dsqg_w_question_indices=dsqg_w_question_indices,
                dsqg_w_hisa_evidence_indices=dsqg_w_hisa_evidence_indices,
                dsqg_w_l3_skip_indices=dsqg_w_l3_skip_indices,
            )
        else:
            trunk_out = self._forward_trunk(idx)
            l3_state = None
        x = trunk_out
        if self.dsqg_w_enabled and self.dsqg_w_has_final_site and self._dsqg_w_site_active('final'):
            x = self._apply_dsqg_w_recomposer(
                trunk_out,
                site_key='final',
                question_indices=dsqg_w_question_indices,
                l3_states=l3_state,
                hisa_evidence_indices=(
                    dsr_hisa_indices if dsr_hisa_indices is not None else dsqg_w_hisa_evidence_indices
                ),
                hisa_evidence_scores=(
                    dsr_hisa_scores if dsr_hisa_scores is not None else None
                ),
                l3_skip_indices=dsqg_w_l3_skip_indices,
            )
        return self.norm(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_embed_parameters(self):
        for m in self.modules():
            if isinstance(m, _DSQG_TYPES):
                yield m.scale_embed

    def non_scale_embed_parameters(self):
        exclude_ids = {id(p) for p in self.scale_embed_parameters()}
        exclude_ids.update(id(p) for p in self.phase_parameters())
        exclude_ids.update(id(p) for p in self.npci_theta_parameters())
        for p in self.parameters():
            if id(p) not in exclude_ids:
                yield p

    def phase_parameters(self):
        for m in self.modules():
            if isinstance(m, _DSQG_TYPES):
                yield m.phase_base
                yield m.phase_gain
                yield m.phase_gate
                yield m.query_probes
                yield m.key_probes

    def npci_theta_parameters(self):
        for m in self.modules():
            if isinstance(m, _DSQG_TYPES):
                yield m.npci_theta_k
                yield m.npci_theta_v

    def physics_summary(self):
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockTriadic) and block.interference:
                alpha = abs(block.ema_factor.item()) + EMA_FLOOR
                win = round(1.0 / max(alpha, EMA_FLOOR))
                entries.append(f'b{i}[{block.group_label}]: alpha={alpha:.4f}(w~{win}t)')
        return '  '.join(entries)

    def layer_summary(self):
        parts = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockTriadic):
                label = block.group_label
                j = block.attn.j_val
                iflag = '+IF' if block.interference else ''
                shift = getattr(block, 'plane_shift', 0)
                q6 = ',q6_g128' if isinstance(block.attn, DSQGAttentionV19Q6G128Smoke) else ''
                parts.append(f'L{i}:DSQG-{label}(J={j},shift={shift}{q6}){iflag}')
            elif isinstance(block, DSRBlock):
                parts.append(f'L{i}:DSR-{HISA_IMPL.upper()}HISA(C={block.attn.num_chunks},k={block.attn.top_k_chunks},HISA_m={block.attn.hisa_top_m_tokens})')
        if self.dsqg_w_enabled and self.dsqg_w_config is not None:
            path = _dsqg_w_candidate_path_label().lower().replace('_', '+')
            site_text = ','.join(self.dsqg_w_site_keys)
            parts.append(f'DSQG-W-sites={site_text}')
            parts.append(f'FINAL:DSQG-W(J<={self.dsqg_w_config.max_candidates},{path})')
        return '  '.join(parts)


# =============================================================================
# DATA UTILITIES
# =============================================================================

class BPETokenizerWrapper:
    def __init__(self, tok):
        self.tokenizer = tok

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


def _sha256_file(path, *, max_bytes=None):
    h = hashlib.sha256()
    remaining = max_bytes
    with open(path, 'rb') as f:
        while True:
            if remaining is not None and remaining <= 0:
                break
            chunk_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def _file_fingerprint(path, *, hash_file=False):
    if not path or not os.path.exists(path):
        return {'path': path, 'exists': False}
    st = os.stat(path)
    info = {
        'path': path,
        'exists': True,
        'size_bytes': st.st_size,
        'mtime_ns': st.st_mtime_ns,
    }
    if hash_file:
        info['sha256'] = _sha256_file(path)
    return info


def _git_status_short():
    try:
        return subprocess.check_output(
            ['git', 'status', '--short'], stderr=subprocess.DEVNULL
        ).decode().splitlines()
    except Exception:
        return []


def _env_snapshot():
    prefixes = ('DWARF_', 'HISA_', 'PYTORCH_', 'CUDA_VISIBLE_DEVICES')
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes)}


def _base_checkpoint_config(*, git_hash, tok_path, encoded_path, n_params):
    return {
        'script': os.path.relpath(__file__, _project_root),
        'script_fingerprint': _file_fingerprint(__file__, hash_file=True),
        'git_hash': git_hash,
        'git_status_short': _git_status_short(),
        'seed': SEED,
        'tokenizer': _file_fingerprint(tok_path, hash_file=True),
        'dataset': _file_fingerprint(encoded_path, hash_file=os.getenv('DWARF_HASH_DATASET', '0') == '1'),
        'env': _env_snapshot(),
        'model': {
            'embedding_dim': EMBEDDING_DIM,
            'num_heads': NUM_HEADS,
            'head_dim': EMBEDDING_DIM // NUM_HEADS,
            'ffn_dim': FFN_DIM,
            'seq_len': MAX_SEQ_LEN,
            'dsr_layer': DSR_LAYER,
            'num_layers': NUM_LAYERS,
            'num_chunks': NUM_CHUNKS,
            'top_k_chunks': TOP_K_CHUNKS,
            'hisa_top_m_tokens': HISA_TOP_M_TOKENS,
            'pure_dsqg_baseline': PURE_DSQG_BASELINE,
            'r_planes': R_PLANES,
            'stagger_movt_planes': STAGGER_MOVT_PLANES,
            'movt_plane_shifts': [
                _movt_plane_shift_for_dsqg_index(i, EMBEDDING_DIM // NUM_HEADS, R_PLANES)
                for i in range(sum(1 for label, *_ in LAYER_LAYOUT if label != 'DSR'))
            ] if STAGGER_MOVT_PLANES else [],
            'tied_lm_head': True,
            'dsqg_w': {
                'enabled': DSQG_W_ENABLED,
                'sourcewise': DSQG_W_SOURCEWISE,
                'triton_sourcewise': DSQG_W_TRITON_SOURCEWISE,
                'insertion': 'after_final_trunk_before_final_norm',
                'max_candidates': DSQG_W_MAX_CANDIDATES,
                'bottleneck': DSQG_W_BOTTLENECK,
                'gate_init': DSQG_W_GATE_INIT,
                'fuse_init_std': DSQG_W_FUSE_INIT_STD,
                'width_cell': DSQG_W_WIDTH_CELL,
                'width_bottleneck': DSQG_W_WIDTH_BOTTLENECK,
                'width_gate_init': DSQG_W_WIDTH_GATE_INIT,
                'force_width_gate': _env_float_or_none('DWARF_DSQG_W_FORCE_WIDTH_GATE'),
                'width_aux_weight': DSQG_W_WIDTH_AUX_WEIGHT,
                'width_entropy_floor': DSQG_W_WIDTH_ENTROPY_FLOOR,
                'width_entropy_weight': DSQG_W_WIDTH_ENTROPY_WEIGHT,
                'typed_mixer': DSQG_W_TYPED_MIXER,
                'typed_mixer_bottleneck': DSQG_W_TYPED_MIXER_BOTTLENECK,
                'typed_mixer_gate_init': DSQG_W_TYPED_MIXER_GATE_INIT,
                'force_typed_mixer_gate': _env_float_or_none('DWARF_DSQG_W_FORCE_TYPED_MIXER_GATE'),
                'typed_hisa_reps': DSQG_W_TYPED_HISA_REPS,
                'query_type_bias': DSQG_W_QUERY_TYPE_BIAS,
                'dsr_candidates': DSQG_W_DSR_CANDIDATES,
                'local_offsets': list(DSQG_W_LOCAL_OFFSETS),
                'long_offsets': list(DSQG_W_LONG_OFFSETS),
                'question_enabled': DSQG_W_QUESTION_ENABLED,
                'k_question': DSQG_W_K_QUESTION,
                'hisa_l3_enabled': DSQG_W_HISA_L3_ENABLED,
                'k_hisa_evidence': DSQG_W_K_HISA_EVIDENCE,
                'k_l3_skip': DSQG_W_K_L3_SKIP,
                'evidence_binding_hub': DSQG_W_EVIDENCE_BINDING_HUB,
                'ebh_bottleneck': DSQG_W_EBH_BOTTLENECK,
                'ebh_gate_init': DSQG_W_EBH_GATE_INIT,
                'force_ebh_gate': _env_float_or_none('DWARF_DSQG_W_FORCE_EBH_GATE'),
                'ebh_phase_bands': DSQG_W_EBH_PHASE_BANDS,
                'ebh_score_features': DSQG_W_EBH_SCORE_FEATURES,
                'ebh_sourcewise_packet': DSQG_W_EBH_SOURCEWISE_PACKET,
                'ebh_triton_lane_accum': DSQG_W_EBH_TRITON_LANE_ACCUM,
                'ebh_pair_mixer': DSQG_W_EBH_PAIR_MIXER,
                'ebh_pair_rank': DSQG_W_EBH_PAIR_RANK,
                'ebh_pair_gate_init': DSQG_W_EBH_PAIR_GATE_INIT,
                'force_ebh_pair_gate': _env_float_or_none('DWARF_DSQG_W_FORCE_EBH_PAIR_GATE'),
                'evidence_prior': DSQG_W_EVIDENCE_PRIOR,
                'evidence_prior_clip': DSQG_W_EVIDENCE_PRIOR_CLIP,
                'evidence_prior_init_scale': DSQG_W_EVIDENCE_PRIOR_INIT_SCALE,
                'candidate_quotas': DSQG_W_CANDIDATE_QUOTAS,
                'quota_hisa_max': DSQG_W_QUOTA_HISA_MAX,
                'candidate_workspace': DSQG_W_CANDIDATE_WORKSPACE,
                'candidate_workspace_dim': DSQG_W_CANDIDATE_WORKSPACE_DIM,
                'candidate_workspace_phase_bands': DSQG_W_CANDIDATE_WORKSPACE_PHASE_BANDS,
                'candidate_workspace_score_features': DSQG_W_CANDIDATE_WORKSPACE_SCORE_FEATURES,
                'candidate_workspace_query_scores': DSQG_W_CANDIDATE_WORKSPACE_QUERY_SCORES,
                'candidate_workspace_pair_transfer': DSQG_W_CANDIDATE_WORKSPACE_PAIR_TRANSFER,
                'candidate_workspace_pair_gate_init': DSQG_W_CANDIDATE_WORKSPACE_PAIR_GATE_INIT,
                'sourcewise_width_cell_fusion': os.getenv('DWARF_DSQG_W_SOURCEWISE_WIDTH_CELL_FUSION', '0') == '1',
                'projected_width_control': os.getenv('DWARF_DSQG_W_PROJECTED_WIDTH_CONTROL', '0') == '1',
                'triton_transformed_compact_read': os.getenv('DWARF_DSQG_W_TRITON_TRANSFORMED_COMPACT_READ', '0') == '1',
                'triton_compact_read_backward': os.getenv('DWARF_DSQG_W_TRITON_COMPACT_READ_BACKWARD', 'triton'),
                'triton_backward_organization': os.getenv('DWARF_DSQG_W_TRITON_BACKWARD_ORGANIZATION', 'monolithic'),
                'triton_backward_source_grads': os.getenv('DWARF_DSQG_W_TRITON_BACKWARD_SOURCE_GRADS', '1') != '0',
                'active_site_mode': 'multi_site' if len(DSQG_W_SITE_SPECS) > 1 else 'single_site',
                'site_scheduling_policy': 'fixed_env_order',
                'sites': [_dsqg_w_site_key(site) for site in DSQG_W_SITE_SPECS],
                'pre_hisa_ema_policy': 'enabled_required_for_promoted_lanes' if PRE_HISA_EMA_ENABLED else 'legacy_ablation_disabled',
                'layer_layout_marker': _layer_layout_marker(),
                'lane_label': _dsqg_w_lane_label(),
                'legacy_guarded_modes': _dsqg_w_legacy_mode_labels(),
                'candidate_path': _dsqg_w_candidate_path_label(),
            },
            'params': n_params,
            'pre_hisa_ema': PRE_HISA_EMA_ENABLED,
            'layer_layout': [(label, len(offsets) if offsets is not None else 0, has_if)
                             for label, offsets, _, _, has_if in LAYER_LAYOUT],
        },
        'training': {
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'scale_embed_lr_mult': SCALE_EMBED_LR_MULT,
            'phase_lr_mult': PHASE_LR_MULT,
            'npci_theta_lr_mult': NPCI_THETA_LR_MULT,
            'grad_clip_norm': GRAD_CLIP_NORM,
            'skip_nonfinite_step': SKIP_NONFINITE_STEP,
            'se_max_abort': SE_MAX_ABORT,
            'npci_theta_max': NPCI_THETA_MAX,
            'npci_theta_init': NPCI_THETA_INIT,
            'lr_warmup_steps': LR_WARMUP_STEPS,
            'min_lr_ratio': MIN_LR_RATIO,
            'scale_embed_constant_lr': SCALE_EMBED_CONSTANT_LR,
            'batch_size': BATCH_SIZE,
            'grad_accum': GRAD_ACCUM,
            'max_train_seqs': MAX_TRAIN_SEQS,
            'screen_epochs': SCREEN_EPOCHS,
            'checkpoint_strategy': CHECKPOINT_STRATEGY,
            'torch_compile_enabled': TORCH_COMPILE_ENABLED,
            'torch_compile_mode': TORCH_COMPILE_MODE,
            'compile_suppress_errors': COMPILE_SUPPRESS_ERRORS,
            'use_liger_ce': USE_LIGER_CE,
            'ce_chunk': CE_CHUNK,
            'optimizer_kind': OPTIMIZER_KIND,
            'muon_adjust_lr_fn': MUON_ADJUST_LR_FN,
            'muon_momentum': MUON_MOMENTUM,
            'muon_ns_steps': MUON_NS_STEPS,
        },
        'eval': {
            'passkey_distances': list(PASSKEY_DISTANCES),
            'passkey_trials': PASSKEY_TRIALS,
            'passkey_batch_size': PASSKEY_BATCH_SIZE,
            'require_prefix_clean': REQUIRE_PREFIX_CLEAN,
            'passkey_words': list(_PASSKEY_WORDS),
            'retrieval_cue': _RETRIEVAL_CUE,
        },
    }


def _adamw_cls():
    if _BNB_AVAILABLE and bnb is not None:
        return getattr(bnb.optim, 'PagedAdamW8bit', bnb.optim.AdamW8bit)
    return torch.optim.AdamW


def _is_dsqg_w_gate_param(name, p):
    return DSQG_W_ENABLED and p.ndim == 1 and 'dsqg_w' in name and name.endswith('.gate')


def _make_optimizer_param_groups(model_ref):
    scale_embed_params = list(model_ref.scale_embed_parameters())
    phase_params = list(model_ref.phase_parameters())
    npci_theta_params = list(model_ref.npci_theta_parameters())
    special_ids = {id(p) for p in scale_embed_params}
    special_ids.update(id(p) for p in phase_params)
    special_ids.update(id(p) for p in npci_theta_params)

    decay_params, no_decay_params, dsqg_w_gate_params = [], [], []
    for name, p in model_ref.named_parameters():
        if not p.requires_grad or id(p) in special_ids:
            continue
        lname = name.lower()
        if _is_dsqg_w_gate_param(name, p):
            dsqg_w_gate_params.append(p)
        elif p.ndim < 2 or name.endswith('.bias') or 'norm' in lname or name in ('embedding.weight', 'out.weight'):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    return [
        {'params': decay_params, 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'name': 'decay'},
        {'params': no_decay_params, 'lr': LR, 'weight_decay': 0.0, 'name': 'no_decay'},
        {'params': dsqg_w_gate_params, 'lr': LR * DSQG_W_GATE_LR_MULT, 'weight_decay': 0.0, 'name': 'dsqg_w_gate'},
        {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT, 'weight_decay': 0.0, 'name': 'scale_embed'},
        {'params': phase_params, 'lr': LR * PHASE_LR_MULT, 'weight_decay': 0.0, 'name': 'phase'},
        {'params': npci_theta_params, 'lr': LR * NPCI_THETA_LR_MULT, 'weight_decay': 0.0, 'name': 'npci_theta'},
    ]


def _is_muon_hidden_param(name, p, special_ids):
    if id(p) in special_ids or p.ndim != 2:
        return False
    lname = name.lower()
    if name in ('embedding.weight', 'out.weight'):
        return False
    if name.endswith('.bias') or 'norm' in lname:
        return False
    return True


def _make_hybrid_muon_param_groups(model_ref):
    scale_embed_params = list(model_ref.scale_embed_parameters())
    phase_params = list(model_ref.phase_parameters())
    npci_theta_params = list(model_ref.npci_theta_parameters())
    special_ids = {id(p) for p in scale_embed_params}
    special_ids.update(id(p) for p in phase_params)
    special_ids.update(id(p) for p in npci_theta_params)

    muon_hidden, adamw_decay, adamw_no_decay, adamw_dsqg_w_gate = [], [], [], []
    for name, p in model_ref.named_parameters():
        if not p.requires_grad or id(p) in special_ids:
            continue
        lname = name.lower()
        if _is_dsqg_w_gate_param(name, p):
            adamw_dsqg_w_gate.append(p)
        elif _is_muon_hidden_param(name, p, special_ids):
            muon_hidden.append(p)
        elif p.ndim >= 2 and not name.endswith('.bias') and 'norm' not in lname and name not in ('embedding.weight', 'out.weight'):
            adamw_decay.append(p)
        else:
            adamw_no_decay.append(p)

    if not muon_hidden:
        raise RuntimeError('DWARF_OPT=muon selected but no 2D hidden parameters were assigned to Muon')

    return {
        'muon': [
            {'params': muon_hidden, 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'name': 'muon_hidden'},
        ],
        'adamw': [
            {'params': adamw_decay, 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'name': 'adamw_decay'},
            {'params': adamw_no_decay, 'lr': LR, 'weight_decay': 0.0, 'name': 'adamw_no_decay'},
            {'params': adamw_dsqg_w_gate, 'lr': LR * DSQG_W_GATE_LR_MULT, 'weight_decay': 0.0, 'name': 'adamw_dsqg_w_gate'},
            {'params': scale_embed_params, 'lr': LR * SCALE_EMBED_LR_MULT, 'weight_decay': 0.0, 'name': 'adamw_scale_embed'},
            {'params': phase_params, 'lr': LR * PHASE_LR_MULT, 'weight_decay': 0.0, 'name': 'adamw_phase'},
            {'params': npci_theta_params, 'lr': LR * NPCI_THETA_LR_MULT, 'weight_decay': 0.0, 'name': 'adamw_npci_theta'},
        ],
        'scale_embed_params': scale_embed_params,
        'phase_params': phase_params,
    }


class _MultiOptimizer:
    """Small wrapper so the trainer can step Muon(hidden) + AdamW(rest) together."""

    def __init__(self, named_optimizers):
        self.named_optimizers = list(named_optimizers)
        self.param_groups = []
        for opt_name, opt in self.named_optimizers:
            for group in opt.param_groups:
                group.setdefault('optimizer', opt_name)
                self.param_groups.append(group)

    def zero_grad(self, set_to_none=True):
        for _, opt in self.named_optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for _, opt in self.named_optimizers:
            opt.step()

    def state_dict(self):
        return {
            'kind': 'multi',
            'optimizers': [
                {'name': name, 'state_dict': opt.state_dict()}
                for name, opt in self.named_optimizers
            ],
        }

    def load_state_dict(self, state_dict, *, skip_state_names=()):
        if state_dict.get('kind') != 'multi':
            raise ValueError('cannot load non-multi optimizer state into hybrid Muon optimizer')
        saved = {entry['name']: entry['state_dict'] for entry in state_dict['optimizers']}
        skip_state_names = set(skip_state_names)
        loaded, skipped = [], []
        for name, opt in self.named_optimizers:
            if name not in saved:
                raise ValueError(f'missing optimizer state for {name}')
            if name in skip_state_names:
                skipped.append(name)
                continue
            opt.load_state_dict(saved[name])
            loaded.append(name)
        return {'loaded': loaded, 'skipped': skipped}


class _LambdaLRScheduler:
    """LambdaLR-compatible scheduler for both real optimizers and _MultiOptimizer."""

    def __init__(self, optimizer, lr_lambda):
        self.optimizer = optimizer
        self.lr_lambdas = list(lr_lambda)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = 0
        self._last_lr = list(self.base_lrs)
        self._apply()

    def _apply(self):
        self._last_lr = []
        for i, group in enumerate(self.optimizer.param_groups):
            factor = self.lr_lambdas[i](self.last_epoch)
            lr = self.base_lrs[i] * factor
            group['lr'] = lr
            self._last_lr.append(lr)

    def step(self):
        self.last_epoch += 1
        self._apply()

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return {
            'base_lrs': list(self.base_lrs),
            'last_epoch': self.last_epoch,
            '_last_lr': list(self._last_lr),
        }

    def load_state_dict(self, state_dict):
        self.base_lrs = list(state_dict['base_lrs'])
        self.last_epoch = int(state_dict.get('last_epoch', 0))
        self._apply()


def _build_optimizer(model_ref):
    if OPTIMIZER_KIND == 'adamw':
        optimizer_groups = _make_optimizer_param_groups(model_ref)
        opt_cls = _adamw_cls()
        optimizer = opt_cls(optimizer_groups, betas=(0.9, 0.95), eps=1e-8)
        return optimizer, optimizer_groups[2]['params'], optimizer_groups[3]['params'], opt_cls.__name__

    if not hasattr(torch.optim, 'Muon'):
        raise RuntimeError('DWARF_OPT=muon requested, but torch.optim.Muon is not available')

    grouped = _make_hybrid_muon_param_groups(model_ref)
    muon = torch.optim.Muon(
        grouped['muon'],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        momentum=MUON_MOMENTUM,
        nesterov=True,
        ns_steps=MUON_NS_STEPS,
        adjust_lr_fn=MUON_ADJUST_LR_FN,
    )
    adamw_cls = _adamw_cls()
    adamw = adamw_cls(grouped['adamw'], betas=(0.9, 0.95), eps=1e-8)
    optimizer = _MultiOptimizer([('muon', muon), ('adamw', adamw)])
    label = f"HybridMuon({MUON_ADJUST_LR_FN}, ns={MUON_NS_STEPS})+{adamw_cls.__name__}"
    return optimizer, grouped['scale_embed_params'], grouped['phase_params'], label


def _streamed_linear_ce_loss(hidden: torch.Tensor,
                             targets: torch.Tensor,
                             weight: torch.Tensor,
                             *,
                             chunk_rows: int,
                             grad_denom: float | None = None,
                             loss_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Compute tied final-projection CE, optionally masked for answer-only continuation."""
    hidden_c = hidden.contiguous()
    h = hidden_c.view(-1, hidden_c.size(-1))
    y = targets.reshape(-1)
    flat_mask = None if loss_mask is None else loss_mask.reshape(-1).to(device=h.device, dtype=torch.bool)
    n_rows = h.size(0)
    n_valid = n_rows if flat_mask is None else int(flat_mask.sum().item())
    total_loss = torch.zeros((), device=h.device, dtype=torch.float32)
    chunk_rows = max(1, int(chunk_rows))
    grad_h = torch.empty_like(h) if grad_denom is not None else None

    if n_valid == 0:
        if grad_h is not None:
            hidden_c.backward(torch.zeros_like(hidden_c))
        return total_loss, 0

    for s in range(0, n_rows, chunk_rows):
        e = min(s + chunk_rows, n_rows)
        h_chunk = h[s:e]
        if grad_denom is not None:
            h_chunk = h_chunk.detach().requires_grad_(True)
        with _amp_context(h.device.type):
            logits = F.linear(h_chunk, weight)
        row_mask = None if flat_mask is None else flat_mask[s:e]
        if row_mask is not None:
            if not bool(row_mask.any()):
                if grad_h is not None:
                    grad_h[s:e].zero_()
                del logits, h_chunk
                continue
            logits_for_loss = logits[row_mask]
            targets_for_loss = y[s:e][row_mask]
        else:
            logits_for_loss = logits
            targets_for_loss = y[s:e]
        loss_sum = F.cross_entropy(logits_for_loss.float(), targets_for_loss, reduction='sum')
        total_loss = total_loss + loss_sum.detach()
        if grad_denom is not None:
            (loss_sum / float(grad_denom)).backward()
            grad_h[s:e].copy_(h_chunk.grad)
        del logits, logits_for_loss, targets_for_loss, loss_sum, h_chunk

    if grad_h is not None:
        hidden_c.backward(grad_h.view_as(hidden_c))

    return total_loss, n_valid


def _validate_dataset_token_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor):
        raise ValueError(f'{name} must be a tensor, got {type(tensor).__name__}')
    if tensor.ndim != 2:
        raise ValueError(f'{name} must have shape [rows, seq_len], got {tuple(tensor.shape)}')
    if tensor.size(1) < 2:
        raise ValueError(f'{name} seq_len must be >= 2 for next-token training, got {tensor.size(1)}')


def _coerce_dataset_loss_mask(name: str, mask: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(mask):
        raise ValueError(f'{name} must be a tensor, got {type(mask).__name__}')
    if tuple(mask.shape) != tuple(data.shape):
        raise ValueError(f'{name} shape {tuple(mask.shape)} must match {tuple(data.shape)}')
    return mask.to(dtype=torch.bool).contiguous()


def _dataset_loss_mask_stats(train_loss_mask: torch.Tensor,
                             val_loss_mask: torch.Tensor,
                             *,
                             source: str) -> dict:
    train_targets = train_loss_mask[:, 1:]
    val_targets = val_loss_mask[:, 1:]
    train_real = int(train_targets.sum().item())
    val_real = int(val_targets.sum().item())
    train_slots = int(train_targets.numel())
    val_slots = int(val_targets.numel())
    if train_real == 0:
        raise ValueError('loss masks select zero train target rows after next-token shift')
    sparse = not (bool(train_loss_mask.all()) and bool(val_loss_mask.all()))
    return {
        'source': source,
        'has_train_loss_mask': source == 'dataset',
        'has_val_loss_mask': source == 'dataset',
        'uses_sparse_loss_mask': bool(sparse),
        'train_real_tokens': train_real,
        'train_target_slots': train_slots,
        'train_real_fraction': float(train_real / max(train_slots, 1)),
        'val_real_tokens': val_real,
        'val_target_slots': val_slots,
        'val_real_fraction': float(val_real / max(val_slots, 1)),
    }


def _prepare_dataset_loss_masks(cache: dict,
                                train_data: torch.Tensor,
                                val_data: torch.Tensor,
                                *,
                                use_liger_ce: bool) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Validate optional DWARF-v2-shaped loss masks and compute shifted-row stats.

    Masks are stored token-aligned with the dataset. The trainer predicts
    ``batch[:, 1:]`` from ``batch[:, :-1]``, so the effective target mask is
    always ``batch_loss_mask[:, 1:]``: a true mark at token column ``c`` trains
    the prediction at input position ``c - 1`` for that token.
    """
    _validate_dataset_token_tensor('train', train_data)
    _validate_dataset_token_tensor('val', val_data)
    if not isinstance(cache, dict):
        raise ValueError('encoded dataset must be a dict with train/val tensors')

    has_train_mask = cache.get('train_loss_mask') is not None
    has_val_mask = cache.get('val_loss_mask') is not None
    if has_train_mask != has_val_mask:
        raise ValueError('dataset must provide both train_loss_mask and val_loss_mask, or neither')

    if not has_train_mask:
        print('  [dataset] no loss masks found; using all-token CE')
        train_loss_mask = torch.ones_like(train_data, dtype=torch.bool)
        val_loss_mask = torch.ones_like(val_data, dtype=torch.bool)
        return train_loss_mask, val_loss_mask, _dataset_loss_mask_stats(
            train_loss_mask, val_loss_mask, source='all_token'
        )

    train_loss_mask = _coerce_dataset_loss_mask('train_loss_mask', cache['train_loss_mask'], train_data)
    val_loss_mask = _coerce_dataset_loss_mask('val_loss_mask', cache['val_loss_mask'], val_data)
    if use_liger_ce and (not bool(train_loss_mask.all()) or not bool(val_loss_mask.all())):
        raise RuntimeError('Liger fused CE does not support sparse loss masks; run with DWARF_LIGER=0.')
    return train_loss_mask, val_loss_mask, _dataset_loss_mask_stats(
        train_loss_mask, val_loss_mask, source='dataset'
    )


def _attach_loss_mask_stats_to_checkpoint_config(config: dict, loss_mask_stats: dict) -> dict:
    config.setdefault('dataset', {})['loss_mask'] = dict(loss_mask_stats)
    return config


def _dsqg_w_width_aux_loss(*models):
    if (not DSQG_W_ENABLED) or (not DSQG_W_WIDTH_CELL) or DSQG_W_WIDTH_AUX_WEIGHT <= 0.0:
        return None
    for model_ref in models:
        if model_ref is None:
            continue
        telemetry = getattr(model_ref, 'dsqg_w_last_telemetry', {}) or {}
        aux = telemetry.get('dsqg_w_width_aux_loss')
        if torch.is_tensor(aux) and aux.requires_grad and aux.numel() == 1:
            return aux
    return None


def _dsqg_w_width_aux_value(*models):
    for model_ref in models:
        if model_ref is None:
            continue
        telemetry = getattr(model_ref, 'dsqg_w_last_telemetry', {}) or {}
        aux = telemetry.get('dsqg_w_width_aux_loss_value', telemetry.get('dsqg_w_width_aux_loss'))
        if torch.is_tensor(aux) and aux.numel() == 1:
            return float(aux.detach().item())
    return None


def _start_dsqg_w_profiler():
    if not PROFILE_DSQG_W:
        return None
    os.makedirs(PROFILE_DSQG_W_TRACE_DIR, exist_ok=True)
    print(
        f'  [DSQG-W profile] enabled trace_dir={PROFILE_DSQG_W_TRACE_DIR} '
        f'wait={PROFILE_DSQG_W_WAIT} warmup={PROFILE_DSQG_W_WARMUP} active={PROFILE_DSQG_W_ACTIVE}',
        flush=True,
    )
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=PROFILE_DSQG_W_WAIT,
            warmup=PROFILE_DSQG_W_WARMUP,
            active=PROFILE_DSQG_W_ACTIVE,
            repeat=1,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(PROFILE_DSQG_W_TRACE_DIR),
    )
    profiler.start()
    return profiler


def _finish_dsqg_w_profiler(profiler):
    if profiler is None:
        return None
    profiler.stop()
    sort_by = 'cuda_time_total' if torch.cuda.is_available() else 'self_cpu_time_total'
    table = profiler.key_averages().table(sort_by=sort_by, row_limit=80)
    table_path = PROFILE_DSQG_W_TABLE or os.path.join(PROFILE_DSQG_W_TRACE_DIR, 'key_averages.txt')
    os.makedirs(os.path.dirname(table_path) or '.', exist_ok=True)
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(table)
        f.write('\n')
    print(f'  [DSQG-W profile] key averages: {table_path}', flush=True)
    return table_path


@torch.inference_mode()
def evaluate(model, data, device, loss_mask=None):
    model.eval()
    model_ref = _unwrap_compiled_module(model)
    total_loss, total_tokens = 0.0, 0
    bs = max(1, BATCH_SIZE // 2)
    for i in range(0, len(data) - bs + 1, bs):
        x = data[i:i+bs, :-1].to(device, non_blocking=True)
        if x.dtype not in (torch.int32, torch.int64):
            x = x.long()
        y = data[i:i+bs, 1:].to(device, non_blocking=True).long()
        target_mask = None
        if loss_mask is not None:
            target_mask = loss_mask[i:i+bs, 1:].to(device, non_blocking=True)
        dsqg_w_question_indices, dsqg_w_hisa_evidence_indices, dsqg_w_l3_skip_indices = _dsqg_w_training_candidate_indices(x)
        with _amp_context(device):
            hidden = model.forward_hidden(
                x,
                dsqg_w_question_indices=dsqg_w_question_indices,
                dsqg_w_hisa_evidence_indices=dsqg_w_hisa_evidence_indices,
                dsqg_w_l3_skip_indices=dsqg_w_l3_skip_indices,
            )
            loss_sum, n_rows = _streamed_linear_ce_loss(
                hidden, y, model_ref.out.weight,
                chunk_rows=CE_CHUNK,
                grad_denom=None,
                loss_mask=target_mask,
            )
        total_loss += float(loss_sum.item())
        total_tokens += n_rows
        del hidden, x, y, target_mask
    return total_loss / max(total_tokens, 1)


def _passkey_config():
    return PasskeyConfig(
        max_seq_len=MAX_SEQ_LEN,
        distances=list(PASSKEY_DISTANCES),
        trials=PASSKEY_TRIALS,
        batch_size=PASSKEY_BATCH_SIZE,
        words=list(_PASSKEY_WORDS),
        filler_sentence=_FILLER_SENTENCE,
        intro_template=_INTRO_TEMPLATE,
        retrieval_cue=_RETRIEVAL_CUE,
        pad_id=0,
    )


@torch.inference_mode()
def passkey_accuracy(model, tokenizer, device):
    audit = passkey_prefix_consistency_audit(model, tokenizer, device, _passkey_config())
    print(
        f"  [passkey audit] clean={audit['prefix_consistent']} "
        f"max_pad_delta={audit['max_pad_logit_delta']:.3e} "
        f"max_suffix_delta={audit['max_suffix_logit_delta']:.3e}",
        flush=True,
    )
    if not audit['prefix_consistent']:
        print(
            '  [passkey audit] WARNING: prefix-only score is reported; legacy padded passkey is contaminated.',
            flush=True,
        )
        if REQUIRE_PREFIX_CLEAN:
            raise RuntimeError(
                'Passkey prefix-consistency gate failed: '
                f"max_pad_delta={audit['max_pad_logit_delta']:.3e}, "
                f"max_suffix_delta={audit['max_suffix_logit_delta']:.3e}"
            )
    return audit['prefix_accuracy']


# =============================================================================
# TRAINING
# =============================================================================

def _grad_norm_for_named_params(named_params, predicate):
    total_sq = 0.0
    seen = 0
    for name, p in named_params:
        if not predicate(name) or p.grad is None:
            continue
        grad = p.grad.detach()
        if grad.numel() == 0:
            continue
        norm = grad.float().norm().item()
        if math.isfinite(norm):
            total_sq += norm * norm
            seen += 1
    if seen == 0:
        return None
    return math.sqrt(total_sq)


def _dsqg_w_grad_diagnostics(model_ref):
    """Compact DSQG-W gradient norms before clipping/optimizer step.

    Width transfer aux only reaches the width-cell scoring path. The value/up/gate
    groups below are therefore the cheap diagnostic for whether CE is sending any
    signal through the indirect content path.
    """
    named_params = list(model_ref.named_parameters())
    score_terms = (
        '.width_cell.q_proj', '.width_cell.k_proj', '.width_cell.rel_diff_proj',
        '.width_cell.rel_prod_proj', '.width_cell.rel_diff_score', '.width_cell.rel_prod_score',
        '.width_cell.type_pair_bias', '.width_cell.source_pair_bias', '.width_cell.self_bias',
    )
    groups = {
        'w_width_score_gn': lambda n: any(term in n for term in score_terms),
        'w_width_v_gn': lambda n: '.width_cell.v_proj' in n,
        'w_width_up_gn': lambda n: '.width_cell.lateral_up' in n,
        'w_width_gate_gn': lambda n: '.width_cell.gate' in n,
        'w_mix_gate_gn': lambda n: '.typed_mixer.gate' in n,
        'w_all_gate_gn': lambda n: n.endswith('.gate') and (
            'dsqg_w' in n or '.width_cell.' in n or '.typed_mixer.' in n
        ),
    }
    out = {}
    for label, predicate in groups.items():
        val = _grad_norm_for_named_params(named_params, predicate)
        if val is not None:
            out[label] = val
    return out


def train():
    if not torch.cuda.is_available():
        raise RuntimeError('DWARF DSQG/HISA kernels require CUDA + Triton; CPU execution is not supported.')
    device = 'cuda'
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    print('=' * 70)
    if PURE_DSQG_BASELINE:
        print('  DWARF D512-L10 OLMo1Tok BaseV1 — PURE DSQG-D v1 control + R_PLANES=4, TIED LM_HEAD')
        print(f'  DSR/HISA: disabled; L{DSR_LAYER} uses DSQG-A control slot')
    else:
        print(f'  DWARF D512-L10 OLMo1Tok BaseV1 — {HISA_IMPL.upper()} HISA@L{DSR_LAYER} + R_PLANES=4, TIED LM_HEAD')
        print(f'  DSR@L{DSR_LAYER}: {HISA_IMPL_LABEL}(C={NUM_CHUNKS}, top_k={TOP_K_CHUNKS}, HISA_m={HISA_TOP_M_TOKENS})')
    print('  SCRATCH PRETRAINING: base_v1 OLMo-1 tokenizer 35 Mix6T / 20 Cosmo / 15 Code / 15 FinePDFs / 5 Wiki / 5 LongABC / 5 FW-Edu buffer.')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, hd={EMBEDDING_DIM//NUM_HEADS}, '
          f'L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  Groups: A(J={len(GROUP_A)}) B(J={len(GROUP_B)}) C(J={len(GROUP_C)})')
    print(f'  Per-layer bandwidth ratio: {(len(GROUP_A)*64)/EMBEDDING_DIM:.2f}x  (safe <= 3.0x)')
    print(f'  R_PLANES={R_PLANES}')
    if STAGGER_MOVT_PLANES:
        _shift_schedule = [
            _movt_plane_shift_for_dsqg_index(i, EMBEDDING_DIM // NUM_HEADS, R_PLANES)
            for i in range(sum(1 for label, *_ in LAYER_LAYOUT if label != 'DSR'))
        ]
        print(f'  MOVT plane staggering: enabled, DSQG-index shifts={_shift_schedule}')
    else:
        print('  MOVT plane staggering: disabled, all DSQG shifts=0')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
    print(f'  phase LR mult={PHASE_LR_MULT}  grad_clip={GRAD_CLIP_NORM}  skip_nonfinite={SKIP_NONFINITE_STEP}  se_abort={SE_MAX_ABORT}')
    print(f'  EMA alpha0={EMA_INIT} (window~{round(1/EMA_INIT)}t)')
    print(f'  NPCI theta init={NPCI_THETA_INIT} max={NPCI_THETA_MAX} LR mult={NPCI_THETA_LR_MULT}')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS:,}, Epochs={SCREEN_EPOCHS}, Seed={SEED}')
    print(f'  Batch: BS={BATCH_SIZE} x GA={GRAD_ACCUM} = eff_batch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'  checkpoint_strategy={CHECKPOINT_STRATEGY}')
    print('  DSQG: V20-compatible (R=4 sequential Givens, grouped sparse, SE gates)')
    if Q6_G128_ENABLED:
        if Q6_G128_FUSED_CONSUME:
            _banner_tile = _q6_g128_effective_stage_c_tile(MAX_SEQ_LEN, Q6_G128_STAGE_C_TILE)
            if Q6_G128_NUM_KV_HEADS != NUM_HEADS:
                _stage_label = ('Stage-F.3 paired-Q backward/scatter reuse + Stage-F.2 forward reuse experimental'
                                if Q6_G128_STAGE_F3_PAIR_BACKWARD else
                                'Stage-F.2 paired-Q forward reuse + Stage-F.1 KV-aware backward experimental'
                                if Q6_G128_STAGE_F2_PAIR_REUSE else
                                'Stage-F.1 KV-head-aware Stage-E fused-backward experimental')
            elif Q6_G128_STAGE_E_BACKWARD and Q6_G128_STAGE_D_BACKWARD:
                _stage_label = 'Stage-E fused-backward-core experimental'
            else:
                _stage_label = 'Stage-D Triton-scatter/vectorized-backward' if Q6_G128_STAGE_D_BACKWARD else 'Stage-C'
            print(f'  q6_g128 smoke path: enabled layers={sorted(Q6_G128_LAYERS)} seed={Q6_G128_SEED} '
                  f'Hq={NUM_HEADS} Hkv={Q6_G128_NUM_KV_HEADS} '
                  f'({_stage_label}, block_n={Q6_G128_FUSED_BLOCK_N}, backward_tile={_banner_tile})')
        else:
            print(f'  q6_g128 smoke path: enabled layers={sorted(Q6_G128_LAYERS)} seed={Q6_G128_SEED} '
                  f'Hq={NUM_HEADS} Hkv={Q6_G128_NUM_KV_HEADS} '
                  '(Triton direct-gather forward q6 K/V, STE BF16-gather gradients; not fused consume/recompute)')
    else:
        print('  q6_g128 smoke path: disabled')
    if DSQG_W_ENABLED:
        candidate_path = _dsqg_w_candidate_path_label()
        site_text = ','.join(_dsqg_w_site_key(site) for site in DSQG_W_SITE_SPECS)
        print(f'  DSQG-W recomposer sites={site_text}: enabled J<={DSQG_W_MAX_CANDIDATES} '
              f'bottleneck={DSQG_W_BOTTLENECK} gate_init={DSQG_W_GATE_INIT} '
              f'fuse_init_std={DSQG_W_FUSE_INIT_STD} sourcewise={DSQG_W_SOURCEWISE} '
              f'triton_sourcewise={DSQG_W_TRITON_SOURCEWISE} '
              f'width_cell={DSQG_W_WIDTH_CELL} width_bottleneck={DSQG_W_WIDTH_BOTTLENECK} '
              f'width_gate_init={DSQG_W_WIDTH_GATE_INIT} width_aux_weight={DSQG_W_WIDTH_AUX_WEIGHT} '
              f'width_entropy_floor={DSQG_W_WIDTH_ENTROPY_FLOOR} width_entropy_weight={DSQG_W_WIDTH_ENTROPY_WEIGHT} '
              f'typed_mixer={DSQG_W_TYPED_MIXER} typed_mixer_bottleneck={DSQG_W_TYPED_MIXER_BOTTLENECK} '
              f'typed_mixer_gate_init={DSQG_W_TYPED_MIXER_GATE_INIT} query_type_bias={DSQG_W_QUERY_TYPE_BIAS} '
              f'evidence_binding_hub={DSQG_W_EVIDENCE_BINDING_HUB} ebh_bottleneck={DSQG_W_EBH_BOTTLENECK} '
              f'ebh_gate_init={DSQG_W_EBH_GATE_INIT} ebh_phase_bands={DSQG_W_EBH_PHASE_BANDS} '
              f'ebh_score_features={DSQG_W_EBH_SCORE_FEATURES} ebh_sourcewise_packet={DSQG_W_EBH_SOURCEWISE_PACKET} '
              f'typed_hisa_reps={DSQG_W_TYPED_HISA_REPS} dsr_candidates={DSQG_W_DSR_CANDIDATES} '
              f'candidates={candidate_path}')
    else:
        print('  DSQG-W recomposer: disabled')
    if PURE_DSQG_BASELINE:
        print('  DSR:  disabled (pure DSQG-D v1 control)')
        print('  HISA Stage-2 selector: disabled (pure DSQG-D v1 control)')
    elif HISA_IMPL == 'v16':
        print('  DSR:  V16 strict-causal HISA')
        print(f"  HISA V16 selector: tile={os.getenv('DWARF_HISA_V16_SELECTOR_TILE', '16')} local_window={os.getenv('DWARF_HISA_V16_LOCAL_WINDOW', '64')}")
    else:
        print('  DSR:  V15HISA')
        print(f"  HISA Stage-2 selector: rep_r={os.getenv('HISA_STAGE2_REP_R', os.getenv('DWARF_HISA_STAGE2_REP_R', '0'))} (0=rowmax baseline)")
    if USE_LIGER_CE:
        print('  Using Liger fused CE')
    else:
        print('  Using streamed final-projection CE')
    print(f'  LayerNorm: {"LigerLayerNorm (fused)" if _LIGER_LN else "nn.LayerNorm"}')
    print(f'  SAC: {"enabled (PyTorch 2.4+)" if _SAC_AVAILABLE else "unavailable (requires PyTorch 2.4+)"}')
    print(f'  PASSKEY_TRIALS={PASSKEY_TRIALS}  CE_ROWS={CE_CHUNK}  LOG_INTERVAL={TRAIN_LOG_INTERVAL}  PIN_DATASET={PIN_DATASET}  REQUIRE_PREFIX_CLEAN={REQUIRE_PREFIX_CLEAN}')
    if MAX_ACC_STEPS:
        print(f'  MAX_ACC_STEPS={MAX_ACC_STEPS}  BENCH_ONLY={BENCH_ONLY}')
    print(f'  git={git_hash}')

    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('Tokenizer not found.')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    tok_vocab_size = tokenizer.vocab_size()
    if tok_vocab_size != VOCAB_SIZE:
        raise ValueError(
            f'Tokenizer vocab_size={tok_vocab_size} does not match model VOCAB_SIZE={VOCAB_SIZE}. '
            'Set DWARF_VOCAB_SIZE to match DWARF_TOKENIZER/DWARF_DATASET.'
        )
    print(f'Loaded tokenizer from {tok_path} (vocab={tok_vocab_size:,})')

    encoded_path = DATASET_PATH
    if not os.path.exists(encoded_path):
        raise FileNotFoundError(f'Dataset not found: {encoded_path}')
    _cache = torch.load(encoded_path, weights_only=True)
    cache_vocab_size = _cache.get('vocab_size') if isinstance(_cache, dict) else None
    if cache_vocab_size is not None and int(cache_vocab_size) != VOCAB_SIZE:
        raise ValueError(
            f'Dataset vocab_size={cache_vocab_size} does not match model VOCAB_SIZE={VOCAB_SIZE}. '
            'Set DWARF_VOCAB_SIZE to match the encoded dataset.'
        )
    # Keep cached sequences compact on host; cast targets to int64 per batch for CE.
    train_data = _cache['train'].to(dtype=torch.int32).contiguous()
    val_data = _cache['val'].to(dtype=torch.int32).contiguous()
    if train_data.size(1) != MAX_SEQ_LEN or val_data.size(1) != MAX_SEQ_LEN:
        raise ValueError(
            f'Dataset sequence length mismatch: train={train_data.size(1)} val={val_data.size(1)} '
            f'but DWARF_SEQ_LEN/MAX_SEQ_LEN={MAX_SEQ_LEN}'
        )
    train_loss_mask, val_loss_mask, loss_mask_stats = _prepare_dataset_loss_masks(
        _cache, train_data, val_data, use_liger_ce=USE_LIGER_CE
    )

    train_data, train_loss_mask, train_tranche = select_train_tranche(
        train_data=train_data,
        train_loss_mask=train_loss_mask,
        max_train_seqs=MAX_TRAIN_SEQS,
        offset_text=os.getenv('DWARF_TRAIN_SEQ_OFFSET'),
    )
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
        val_loss_mask = val_loss_mask[:MAX_VAL_SEQS]
    loss_mask_stats = _dataset_loss_mask_stats(
        train_loss_mask, val_loss_mask, source=loss_mask_stats['source']
    )
    if PIN_DATASET:
        train_data = train_data.pin_memory()
        val_data = val_data.pin_memory()
        train_loss_mask = train_loss_mask.pin_memory()
        val_loss_mask = val_loss_mask.pin_memory()
    train_real = loss_mask_stats['train_real_tokens']
    val_real = loss_mask_stats['val_real_tokens']
    train_slots = max(loss_mask_stats['train_target_slots'], 1)
    val_slots = max(loss_mask_stats['val_target_slots'], 1)
    print(
        f'  train: {len(train_data):,} seqs  val: {len(val_data):,} seqs  host_dtype={train_data.dtype} '
        f'train_real={train_real:,}/{train_slots:,} ({train_real/train_slots:.2%}) '
        f'val_real={val_real:,}/{val_slots:,} ({val_real/val_slots:.2%}) '
        f'train_tranche={train_tranche}'
    )

    model = TriadicJ96Dsr(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN,
        dsr_layer=DSR_LAYER,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
        dropout=DROPOUT,
        num_chunks=NUM_CHUNKS,
        top_k_chunks=TOP_K_CHUNKS,
    ).to(device)

    n_params = model.param_count()
    print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')
    print(f'  Layout: {model.layer_summary()}')
    if PURE_DSQG_BASELINE:
        print('  DSR effective stage2_rep_r=disabled')
    else:
        print(f'  DSR effective stage2_rep_r={getattr(model.blocks[DSR_LAYER].attn, "stage2_rep_r", "?")}')

    if TORCH_COMPILE_ENABLED:
        if COMPILE_CAPTURE_SCALARS:
            torch._dynamo.config.capture_scalar_outputs = True
        if COMPILE_CAPTURE_DYNAMIC:
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.suppress_errors = COMPILE_SUPPRESS_ERRORS
        if COMPILE_ACTIVATION_BUDGET is not None and hasattr(torch._dynamo.config, 'activation_memory_budget'):
            torch._dynamo.config.activation_memory_budget = COMPILE_ACTIVATION_BUDGET
        compile_wrap_t0 = time.time()
        model = torch.compile(
            model,
            mode=TORCH_COMPILE_MODE,
            dynamic=TORCH_COMPILE_DYNAMIC,
            fullgraph=TORCH_COMPILE_FULLGRAPH)
        compile_wrap_ms = (time.time() - compile_wrap_t0) * 1000.0
        print(
            f'  torch.compile=ON mode={TORCH_COMPILE_MODE} '
            f'dynamic={TORCH_COMPILE_DYNAMIC} fullgraph={TORCH_COMPILE_FULLGRAPH} '
            f'capture_scalars={COMPILE_CAPTURE_SCALARS} '
            f'capture_dynamic={COMPILE_CAPTURE_DYNAMIC} '
            f'suppress_errors={COMPILE_SUPPRESS_ERRORS} '
            f'budget={COMPILE_ACTIVATION_BUDGET} wrap_ms={compile_wrap_ms:.1f}')
    else:
        print('  torch.compile=OFF')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model_ref = _unwrap_compiled_module(model)

    optimizer, scale_embed_params, phase_params, optimizer_label = _build_optimizer(model_ref)
    print(f'  Optimizer: {optimizer_label} (DWARF_OPT={OPTIMIZER_KIND})')
    for group in optimizer.param_groups:
        n_group = sum(p.numel() for p in group['params'])
        print(f"  opt_group[{group.get('name', '?')} via {group.get('optimizer', OPTIMIZER_KIND)}]: "
              f"params={n_group:,} lr={group['lr']:.2e} wd={group.get('weight_decay', 0.0):.2g}")

    steps_per_epoch_nominal = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    if MAX_ACC_STEPS:
        steps_per_epoch_nominal = min(steps_per_epoch_nominal, MAX_ACC_STEPS)
    run_total_steps = SCREEN_EPOCHS * max(steps_per_epoch_nominal, 1)
    lr_schedule = build_lr_schedule_config(run_steps=run_total_steps)
    print(
        f'  LR schedule={lr_schedule.kind} total={lr_schedule.total_steps} '
        f'warmup={lr_schedule.warmup_steps} stable={lr_schedule.stable_steps} '
        f'decay={lr_schedule.decay_steps} offset={lr_schedule.step_offset}'
    )

    def _lr_lambda(step, group_idx):
        group_name = optimizer.param_groups[group_idx].get('name', '')
        if SCALE_EMBED_CONSTANT_LR and group_name.endswith('scale_embed'):
            return 1.0
        return lr_schedule_multiplier(step=step, config=lr_schedule, min_lr_ratio=MIN_LR_RATIO)

    scheduler = _LambdaLRScheduler(
        optimizer,
        lr_lambda=[lambda s, gi=gi: _lr_lambda(s, gi) for gi in range(len(optimizer.param_groups))])

    freeze_se = os.getenv('DWARF_FREEZE_SE', '0') == '1'
    if freeze_se:
        for p in scale_embed_params:
            p.requires_grad_(False)
        for gi, group in enumerate(optimizer.param_groups):
            if group.get('name', '').endswith('scale_embed'):
                group['lr'] = 0.0
                scheduler.base_lrs[gi] = 0.0
        print('  [FREEZE] scale_embed frozen (DWARF_FREEZE_SE=1)')

    resume_path = os.getenv('DWARF_RESUME', '')
    start_epoch = int(os.getenv('DWARF_START_EPOCH', '1'))
    skip_sched = os.getenv('DWARF_SKIP_SCHED', '0') == '1'
    reset_paged_adam = os.getenv('DWARF_RESUME_RESET_PAGED_ADAM', '0') == '1'
    resume_optimizer_state = {'mode': 'fresh', 'loaded': [], 'skipped': []}
    validate_schedule_resume(
        config=lr_schedule,
        resume_path=resume_path,
        skip_scheduler_state=skip_sched,
    )
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            incompatible = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(
                    f'  [resume] missing_keys={len(incompatible.missing_keys)} '
                    f'unexpected_keys={len(incompatible.unexpected_keys)}',
                    flush=True,
                )
            skip_opt = os.getenv('DWARF_SKIP_OPT', '0') == '1'
            if not skip_opt:
                try:
                    if isinstance(optimizer, _MultiOptimizer):
                        resume_optimizer_state = optimizer.load_state_dict(
                            ckpt['optimizer_state_dict'],
                            skip_state_names={'adamw'} if reset_paged_adam else (),
                        )
                    else:
                        if reset_paged_adam:
                            raise ValueError('DWARF_RESUME_RESET_PAGED_ADAM=1 requires the hybrid Muon optimizer')
                        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                        resume_optimizer_state = {'loaded': ['optimizer'], 'skipped': []}
                    resume_optimizer_state['mode'] = (
                        'muon_preserved_paged_adam_reset' if reset_paged_adam else 'full_restore'
                    )
                    if reset_paged_adam:
                        print(
                            '  [resume] preserving Muon state and resetting unsupported PagedAdam8bit state '
                            '(DWARF_RESUME_RESET_PAGED_ADAM=1)',
                            flush=True,
                        )
                except (ValueError, RuntimeError) as _oe:
                    print(f'  [resume] optimizer state mismatch ({_oe}); starting fresh optimizer')
                    resume_optimizer_state = {'mode': 'restore_failed_fresh', 'loaded': [], 'skipped': []}
            else:
                print('  [resume] skipping optimizer state (DWARF_SKIP_OPT=1)')
                resume_optimizer_state = {'mode': 'all_optimizer_state_skipped', 'loaded': [], 'skipped': ['muon', 'adamw']}
            if skip_sched:
                print('  [resume] skipping scheduler state (DWARF_SKIP_SCHED=1)')
            elif 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f'  Resumed from {resume_path} (starting epoch {start_epoch})')

    best_val_loss = float('inf')
    passkey_results = {}
    ppl_results = {}
    dsqg_w_profiler = _start_dsqg_w_profiler()

    if USE_LIGER_CE:
        liger_ce_fn = LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)

    model_ref = _unwrap_compiled_module(model)
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * (MAX_SEQ_LEN - 1)
    checkpoint_config = _base_checkpoint_config(
        git_hash=git_hash, tok_path=tok_path, encoded_path=encoded_path, n_params=n_params
    )
    checkpoint_config['training']['lr_schedule'] = {
        'kind': lr_schedule.kind,
        'total_steps': lr_schedule.total_steps,
        'warmup_steps': lr_schedule.warmup_steps,
        'stable_steps': lr_schedule.stable_steps,
        'decay_steps': lr_schedule.decay_steps,
        'step_offset': lr_schedule.step_offset,
    }
    checkpoint_config['training']['train_tranche'] = train_tranche
    checkpoint_config['training']['resume_optimizer_state'] = resume_optimizer_state
    _attach_loss_mask_stats_to_checkpoint_config(checkpoint_config, loss_mask_stats)

    for epoch in range(start_epoch, SCREEN_EPOCHS + 1):
        if device == 'cuda':
            torch.cuda.empty_cache()
        model.train()
        indices = torch.randperm(len(train_data))
        step = 0
        optimizer.zero_grad(set_to_none=True)
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
        if MAX_ACC_STEPS:
            steps_per_epoch = min(steps_per_epoch, MAX_ACC_STEPS)
        step_times = deque(maxlen=20)
        first_step_ms = None

        for acc_step in range(steps_per_epoch):
            t0 = time.time()

            loss_accum = 0.0
            width_aux_accum = 0.0
            width_aux_count = 0
            micro_starts = []
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start < len(train_data):
                    micro_starts.append(idx_start)
            total_rows_this_accum = 0
            for idx_start in micro_starts:
                mb = min(BATCH_SIZE, len(train_data) - idx_start)
                if mb > 0:
                    batch_indices = indices[idx_start:idx_start + BATCH_SIZE]
                    total_rows_this_accum += int(train_loss_mask[batch_indices, 1:].sum().item())
            total_rows_this_accum = max(total_rows_this_accum, 1)

            for idx_start in micro_starts:
                batch_indices = indices[idx_start:idx_start + BATCH_SIZE]
                batch = train_data[batch_indices]
                batch_mask = train_loss_mask[batch_indices]
                x = batch[:, :-1].to(device, non_blocking=True)
                if x.dtype not in (torch.int32, torch.int64):
                    x = x.long()
                y = batch[:, 1:].to(device, non_blocking=True).long()
                target_mask = batch_mask[:, 1:].to(device, non_blocking=True)
                dsqg_w_question_indices, dsqg_w_hisa_evidence_indices, dsqg_w_l3_skip_indices = _dsqg_w_training_candidate_indices(x)

                if USE_LIGER_CE:
                    with _amp_context(device):
                        hidden = model.forward_hidden(
                            x,
                            dsqg_w_question_indices=dsqg_w_question_indices,
                            dsqg_w_hisa_evidence_indices=dsqg_w_hisa_evidence_indices,
                            dsqg_w_l3_skip_indices=dsqg_w_l3_skip_indices,
                        )
                    width_aux = _dsqg_w_width_aux_loss(model_ref, model)
                    width_aux_value = _dsqg_w_width_aux_value(model_ref, model)
                    if DSQG_W_WIDTH_AUX_WEIGHT > 0.0 and width_aux_value is None:
                        raise RuntimeError('DWARF_DSQG_W_WIDTH_AUX_WEIGHT requested but DSQG-W width aux telemetry is unavailable')
                    if width_aux_value is not None:
                        width_aux_accum += width_aux_value
                        width_aux_count += 1
                    if width_aux is not None:
                        (width_aux * (DSQG_W_WIDTH_AUX_WEIGHT / float(max(len(micro_starts), 1)))).backward(retain_graph=True)
                    # LigerFusedLinearCrossEntropyLoss.forward(lin_weight, _input, target)
                    loss = liger_ce_fn(
                        model_ref.out.weight,
                        hidden.contiguous().reshape(-1, hidden.size(-1)),
                        y.reshape(-1))
                    # Liger returns a mean loss; scale by token rows so the
                    # accumulation window is weighted by total tokens, not microbatch count.
                    n_rows = y.numel()
                    (loss * (float(n_rows) / float(total_rows_this_accum))).backward()
                    loss_accum += float(loss.detach().item()) * n_rows
                    del hidden, loss
                else:
                    n_rows = y.numel()
                    with _amp_context(device):
                        hidden = model.forward_hidden(
                            x,
                            dsqg_w_question_indices=dsqg_w_question_indices,
                            dsqg_w_hisa_evidence_indices=dsqg_w_hisa_evidence_indices,
                            dsqg_w_l3_skip_indices=dsqg_w_l3_skip_indices,
                        )
                    width_aux = _dsqg_w_width_aux_loss(model_ref, model)
                    width_aux_value = _dsqg_w_width_aux_value(model_ref, model)
                    if DSQG_W_WIDTH_AUX_WEIGHT > 0.0 and width_aux_value is None:
                        raise RuntimeError('DWARF_DSQG_W_WIDTH_AUX_WEIGHT requested but DSQG-W width aux telemetry is unavailable')
                    if width_aux_value is not None:
                        width_aux_accum += width_aux_value
                        width_aux_count += 1
                    if width_aux is not None:
                        (width_aux * (DSQG_W_WIDTH_AUX_WEIGHT / float(max(len(micro_starts), 1)))).backward(retain_graph=True)
                    total_loss, _ = _streamed_linear_ce_loss(
                        hidden, y, model_ref.out.weight,
                        chunk_rows=CE_CHUNK,
                        grad_denom=total_rows_this_accum,
                        loss_mask=target_mask,
                    )
                    loss_accum += float(total_loss.item())
                    del hidden, total_loss

            loss_val = loss_accum / float(total_rows_this_accum)

            should_log = ((acc_step + 1) % TRAIN_LOG_INTERVAL == 0) or ((acc_step + 1) == steps_per_epoch)
            dsqg_w_grad_diag = _dsqg_w_grad_diagnostics(model_ref) if (DSQG_W_ENABLED and should_log) else {}
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            total_norm_for_guard = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            if (not math.isfinite(loss_val)) or (not math.isfinite(total_norm_for_guard)):
                msg = (f'  [guard] non-finite step at ep{epoch} step {acc_step+1}: '
                       f'loss={loss_val} grad_norm={total_norm_for_guard}')
                if SKIP_NONFINITE_STEP:
                    print(msg + ' — skipping optimizer step', flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise FloatingPointError(msg)

            if SE_MAX_ABORT > 0.0:
                se_guard = max((p.detach().abs().max().item() for p in model_ref.scale_embed_parameters()), default=0.0)
                if (not math.isfinite(se_guard)) or se_guard >= SE_MAX_ABORT:
                    raise FloatingPointError(
                        f'scale_embed guard tripped at ep{epoch} step {acc_step+1}: '
                        f'se_max={se_guard:.6f} >= {SE_MAX_ABORT:.6f}'
                    )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if dsqg_w_profiler is not None:
                dsqg_w_profiler.step()
            step += 1

            step_ms = (time.time() - t0) * 1000
            if first_step_ms is None:
                first_step_ms = step_ms
            step_times.append(step_ms)

            if should_log:
                avg_ms = sum(step_times) / len(step_times)
                tok_s = tokens_per_step / (avg_ms / 1000.0)
                se_max = max(
                    (p.abs().max().item() for p in model_ref.scale_embed_parameters()),
                    default=0.0)
                total_norm = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
                lr_now = scheduler.get_last_lr()[0]
                routing_entropy = getattr(
                    model_ref.blocks[DSR_LAYER].attn,
                    '_routing_entropy', None)
                entropy_str = ''
                if width_aux_count > 0:
                    entropy_str += f' w_aux={width_aux_accum / float(width_aux_count):.4f}'
                if routing_entropy is not None:
                    if torch.is_tensor(routing_entropy):
                        routing_entropy = float(routing_entropy.detach().item())
                    if isinstance(routing_entropy, float) and math.isfinite(routing_entropy):
                        entropy_str += f' routing_ent={routing_entropy:.3f}'
                stage2_frac = getattr(
                    model_ref.blocks[DSR_LAYER].attn,
                    '_stage2_selected_fraction', None)
                if stage2_frac is not None:
                    if torch.is_tensor(stage2_frac):
                        stage2_frac = float(stage2_frac.detach().item())
                    if isinstance(stage2_frac, float) and math.isfinite(stage2_frac):
                        entropy_str += f' stage2_frac={stage2_frac:.3f}'
                dsqg_w_tel = getattr(model_ref, 'dsqg_w_last_telemetry', {}) or {}
                if dsqg_w_tel:
                    def _tel_float(name):
                        val = dsqg_w_tel.get(name)
                        if torch.is_tensor(val) and val.numel() == 1:
                            return float(val.detach().item())
                        if isinstance(val, (int, float)):
                            return float(val)
                        return None
                    for _name, _label in [
                        ('dsqg_w_gate_mean', 'w_gate'),
                        ('dsqg_w_delta_to_x_ratio', 'w_dx'),
                        ('dsqg_w_hisa_source_mass', 'w_hisa'),
                        ('dsqg_w_candidate_score_bias_norm', 'w_score'),
                        ('dsqg_w_candidate_score_mean', 'w_smean'),
                        ('dsqg_w_typed_mixer_gate_mean', 'w_mix_gate'),
                        ('dsqg_w_typed_mixer_forced_gate', 'w_mix_forced'),
                        ('dsqg_w_width_gate_mean', 'w_width_gate'),
                        ('dsqg_w_width_forced_gate', 'w_width_forced'),
                        ('dsqg_w_width_delta_norm', 'w_width_delta'),
                        ('dsqg_w_width_entropy', 'w_width_ent'),
                        ('dsqg_w_width_self_mass', 'w_width_self'),
                        ('dsqg_w_width_question_to_hisa_evidence_mass', 'w_width_qh'),
                        ('dsqg_w_width_hisa_evidence_to_question_mass', 'w_width_hq'),
                        ('dsqg_w_width_transfer_aux_loss', 'w_width_xfer'),
                        ('dsqg_w_width_entropy_penalty', 'w_width_ep'),
                        ('dsqg_w_width_rel_diff_score_norm', 'w_rel_diff'),
                        ('dsqg_w_width_rel_prod_score_norm', 'w_rel_prod'),
                        ('dsqg_w_ebh_enabled', 'w_ebh'),
                        ('dsqg_w_ebh_bind_gate_mean', 'w_ebh_gate'),
                        ('dsqg_w_ebh_forced_gate', 'w_ebh_forced'),
                        ('dsqg_w_ebh_delta_to_x_ratio', 'w_ebh_dx'),
                        ('dsqg_w_ebh_gated_delta_to_x_ratio', 'w_ebh_gdx'),
                        ('dsqg_w_ebh_bound_packet_norm', 'w_ebh_pkt'),
                        ('dsqg_w_ebh_active_row_fraction', 'w_ebh_active'),
                        ('dsqg_w_ebh_pair_mixer_enabled', 'w_ebh_pair'),
                        ('dsqg_w_ebh_pair_gate_mean', 'w_ebh_pair_gate'),
                        ('dsqg_w_ebh_pair_forced_gate', 'w_ebh_pair_forced'),
                        ('dsqg_w_ebh_pair_entropy', 'w_ebh_pair_ent'),
                        ('dsqg_w_ebh_pair_self_mass', 'w_ebh_pair_self'),
                        ('dsqg_w_ebh_pair_delta_norm', 'w_ebh_pair_delta'),
                        ('dsqg_w_ebh_pair_question_to_hisa_mass', 'w_ebh_pair_qh'),
                        ('dsqg_w_ebh_pair_hisa_to_question_mass', 'w_ebh_pair_hq'),
                        ('dsqg_w_sourcewise_ebh_materialized', 'w_ebh_mat'),
                        ('dsqg_w_ebh_packet_sourcewise', 'w_ebh_packet'),
                        ('dsqg_w_ebh_packet_triton', 'w_ebh_triton'),
                        ('dsqg_w_ebh_packet_semantic_approx', 'w_ebh_sem_approx'),
                        ('dsqg_w_candidate_workspace_enabled', 'w_ws'),
                        ('dsqg_w_candidate_workspace_dim', 'w_ws_dim'),
                        ('dsqg_w_candidate_workspace_score_bias_norm', 'w_ws_score'),
                        ('dsqg_w_candidate_workspace_query_conditioned', 'w_ws_q'),
                        ('dsqg_w_candidate_workspace_query_score_norm', 'w_ws_qscore'),
                        ('dsqg_w_candidate_workspace_norm', 'w_ws_norm'),
                        ('dsqg_w_candidate_workspace_pair_transfer', 'w_ws_pair'),
                        ('dsqg_w_candidate_workspace_pair_gate', 'w_ws_pair_gate'),
                        ('dsqg_w_candidate_workspace_materialized_d_candidates', 'w_ws_matd'),
                        ('dsqg_w_metadata_cache_hit', 'w_cache'),
                        ('dsqg_w_static_source_count', 'w_srcs'),
                        ('dsqg_w_candidate_slot_count', 'w_j'),
                        ('dsqg_w_detached_recomposer', 'w_det'),
                        ('dsqg_w_active_site_cycle', 'w_site_cycle'),
                        ('dsqg_w_fast_evidence_mean', 'w_fast'),
                        ('dsqg_w_fast_evidence_mean_bypass', 'w_fast_bypass'),
                        ('dsqg_w_force_trainable_candidate_path', 'w_trainable'),
                        ('dsqg_w_sourcewise_semantic_materialized', 'w_mat'),
                        ('dsqg_w_triton_sourcewise_semantic_bypass', 'w_sem_bypass'),
                        ('dsqg_w_geometry_fixed_slots', 'w_fixed'),
                        ('dsqg_w_geometry_slab_candidate_slots', 'w_slab'),
                        ('dsqg_w_triton_backward_source_grad_every', 'w_src_every'),
                    ]:
                        _val = _tel_float(_name)
                        if _val is not None and math.isfinite(_val):
                            entropy_str += f' {_label}={_val:.3f}'
                    for _name, _label in [
                        ('dsqg_w_gate_logit_mean', 'w_gate_logit'),
                        ('dsqg_w_typed_mixer_gate_logit_mean', 'w_mix_gate_logit'),
                        ('dsqg_w_width_gate_logit_mean', 'w_width_gate_logit'),
                        ('dsqg_w_ebh_bind_gate_logit_mean', 'w_ebh_gate_logit'),
                        ('dsqg_w_ebh_pair_gate_logit_mean', 'w_ebh_pair_gate_logit'),
                    ]:
                        _val = _tel_float(_name)
                        if _val is not None and math.isfinite(_val):
                            entropy_str += f' {_label}={_val:.6f}'
                for _label, _val in dsqg_w_grad_diag.items():
                    if isinstance(_val, (int, float)) and math.isfinite(float(_val)):
                        entropy_str += f' {_label}={float(_val):.3e}'
                print(f'  [ep{epoch} step {acc_step+1}/{steps_per_epoch}] '
                      f'ce={loss_val:.4f} se_max={se_max:.3f} '
                      f'grad_norm={total_norm:.4f} lr={lr_now:.2e} '
                      f'{tok_s:.0f} tok/s{entropy_str}', flush=True)

        if BENCH_ONLY:
            avg_ms = sum(step_times) / len(step_times)
            tok_s = tokens_per_step / (avg_ms / 1000.0)
            compile_overhead_ms = max(first_step_ms - avg_ms, 0.0) if first_step_ms is not None else 0.0
            memory_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
            print(f'\n[BENCH] first_step_ms={first_step_ms:.1f} trailing_avg_ms={avg_ms:.1f} '
                  f'steady_tok_s={tok_s:.0f} approx_compile_overhead_ms={compile_overhead_ms:.1f}')
            print(f'[BENCH] peak_vram={memory_mb:.0f}MB compile={TORCH_COMPILE_ENABLED} '
                  f'mode={TORCH_COMPILE_MODE if TORCH_COMPILE_ENABLED else "eager"} '
                  f'window={len(step_times)} steps={step}')
            q6_reports = []
            for i, block in enumerate(model_ref.blocks):
                if isinstance(block, DSQGBlockTriadic):
                    attn = _unwrap_compiled_module(block.attn)
                    report = getattr(attn, '_q6_last_report', None)
                    if report:
                        q6_reports.append((i, report))
            for i, report in q6_reports:
                print(f'[BENCH_Q6] layer={i} read={report.get("read_implementation", "unknown")} '
                      f'scratch_mode={report.get("scratch_mode", "unknown")} '
                      f'resident_q6={report.get("resident_q6_bytes", 0) / 1e6:.3f}MB '
                      f'gather_out={report.get("gather_output_bytes", 0) / 1e6:.3f}MB '
                      f'materialized_gather={report.get("materialized_gather_bytes", report.get("gather_output_bytes", 0)) / 1e6:.3f}MB '
                      f'peak_scratch={report.get("peak_scratch_bytes", 0) / 1e6:.3f}MB '
                      f'Hq={report.get("num_query_heads", 0)} Hkv={report.get("num_kv_heads", 0)} kv_group={report.get("kv_group_size", 0)} '
                      f'stage_c_tile={report.get("stage_c_backward_tile_tokens", 0)} '
                      f'stage_c_replay_tiles={report.get("stage_c_backward_replay_tiles", 0)} '
                      f'stage_c_tile_gather={report.get("stage_c_backward_tile_gather_bytes", 0) / 1e6:.3f}MB '
                      f'stage_c_tile_vs_full={report.get("stage_c_backward_tile_gather_vs_full", 0.0):.3f} '
                      f'stage_d_decode={report.get("stage_d_tile_decode", "none")} '
                      f'stage_d_scatter={report.get("stage_d_sequence_scatter", "none")} '
                      f'stage_e_core={report.get("stage_e_backward_core", "disabled")} '
                      f'stage_f2_pair_forward={report.get("stage_f2_pair_forward_core", "disabled")} '
                      f'stage_f3_pair_backward={report.get("stage_f3_pair_backward_core", "disabled")} '
                      f'prob_scratch={int(bool(report.get("stage_f3_split_movt_prob_scratch_enabled", False)))} '
                      f'pair_direct={int(bool(report.get("stage_f3_sidecar_pair_direct_enabled", False)))} '
                      f'compression_vs_bf16={report.get("compression_vs_bf16", 0.0):.3f}')
            _finish_dsqg_w_profiler(dsqg_w_profiler)
            return

        val_loss = evaluate(model, val_data, device, val_loss_mask)
        val_ppl = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        _is_best = val_loss < best_val_loss
        if _is_best:
            best_val_loss = val_loss
            clean_state = model_ref.state_dict()
            torch.save({
                'model_state_dict': clean_state,
                'config': checkpoint_config,
            }, os.path.join(CHECKPOINT_DIR, f'{CKPT_BASE_NAME}_best.pt'),
               pickle_protocol=5)
            marker = ' *'

        _save_full = (_is_best or epoch % 3 == 0 or epoch == SCREEN_EPOCHS)
        _ep_state = model_ref.state_dict()
        _ep_ckpt = {'model_state_dict': _ep_state, 'epoch': epoch,
                    'global_step': step, 'config': checkpoint_config}
        if _save_full:
            _ep_ckpt['optimizer_state_dict'] = optimizer.state_dict()
            _ep_ckpt['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(_ep_ckpt,
                   os.path.join(CHECKPOINT_DIR, f'{CKPT_BASE_NAME}_ep{epoch}.pt'),
                   pickle_protocol=5)

        se_vals = [m.scale_embed.detach().abs()
                   for m in model_ref.modules() if isinstance(m, _DSQG_TYPES)]
        if se_vals:
            se_all = torch.cat(se_vals)
            se_mean = se_all.mean().item()
            se_max = se_all.max().item()
            total_se = se_all.numel()
            print(f'\nEp {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')
            print(f'  scale_embed |mean|={se_mean:.4f} |max|={se_max:.4f}')

            for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                count = (se_all >= threshold).sum().item()
                pct = count / total_se * 100
                print(f'  SE>={threshold}: {pct:.1f}% ({count}/{total_se})')

            if se_max >= 1.8:
                if se_max >= 2.0:
                    print(f'  PERCOLATION ZONE: |scale_embed|max={se_max:.4f} >= 1.8 (near threshold ~2.0)')
                    print(f'  PHASE TRANSITION: |scale_embed|max={se_max:.4f} CROSSED 2.0!')
                else:
                    print(f'  PERCOLATION ZONE: |scale_embed|max={se_max:.4f} >= 1.8 (near threshold ~2.0)')

        for i, block in enumerate(model_ref.blocks):
            if isinstance(block, DSQGBlockTriadic):
                attn = _unwrap_compiled_module(block.attn)
                phase_base = attn.phase_base.detach().abs()
                phase_gain = attn.phase_gain.detach().abs()

                for plane in range(R_PLANES):
                    pb_plane = phase_base[:, :, plane]
                    pg_plane = phase_gain[:, :, plane]
                    combined = torch.cat([pb_plane.flatten(), pg_plane.flatten()])
                    print(f'  MOVT[L{i}] plane={plane}: |mean|={combined.mean():.4f} |max|={combined.max():.4f} |min|={combined.min():.4f}')

                head_vals = torch.cat([torch.cat([phase_base[:, :, p], phase_gain[:, :, p]]) for p in range(R_PLANES)])
                head_means = head_vals.mean(dim=0)
                print(f'  MOVT[L{i}] head means: min={head_means.min():.4f} max={head_means.max():.4f}')

                all_phase = torch.cat([phase_base.flatten(), phase_gain.flatten()])
                print(f'  MOVT[L{i}]: |mean|={all_phase.mean():.4f} |max|={all_phase.max():.4f}')

        routing_entropy = getattr(
            model_ref.blocks[DSR_LAYER].attn, '_routing_entropy', None)
        if routing_entropy is not None:
            if torch.is_tensor(routing_entropy):
                routing_entropy = float(routing_entropy.detach().item())
            if isinstance(routing_entropy, float) and math.isfinite(routing_entropy):
                print(f'  DSR routing entropy: {routing_entropy:.4f} '
                      f'(max={math.log(NUM_CHUNKS):.2f}, min=0.00)')

        print(f'  Physics: {model_ref.physics_summary()}')

        pk = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        print('  ' + format_passkey_results(pk))

    _finish_dsqg_w_profiler(dsqg_w_profiler)
    elapsed_s = time.time() - t_start
    memory_mb = torch.cuda.max_memory_allocated() / 1e6
    passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)

    print('\n' + '=' * 70)
    print(f'  DSR + R_PLANES={R_PLANES} Bible-Muon OLMo1Tok BaseV1 Summary (D{EMBEDDING_DIM}-L{NUM_LAYERS}, tied lm_head)')
    print('=' * 70)
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'  ep{ep}: ppl={ppl_results.get(ep, 999.0):.2f}  '
              f'passkey={passkey_results.get(ep, 0.0):.1f}%')
    print(f'  peak_vram={memory_mb:.0f}MB  elapsed={elapsed_s:.0f}s')
    print(f'  params={n_params / 1e6:.1f}M  R_PLANES={R_PLANES}')
    print(f'  num_chunks={NUM_CHUNKS}  top_k_chunks={TOP_K_CHUNKS}  HISA_m={HISA_TOP_M_TOKENS}')

    if passkey_final >= 80:
        print('\n  CONTENT-ADDRESSED ROUTING ACHIEVED — passkey >= 80%')
    elif passkey_final >= 60:
        print('\n  PARTIAL — routing emerging but not fully content-addressed')
    else:
        print('\n  BELOW THRESHOLD — FA signal may be required for routing bootstrap')


if __name__ == '__main__':
    import traceback
    try:
        train()
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)
