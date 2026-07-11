from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train" / "train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"
OVERLAY = ROOT / "kernel_overlays" / "bwd_tile_tuning"
CANONICAL = ROOT / "kernels"


def _imported_kernel_path(kernel_dir: Path | None) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    if kernel_dir is None:
        env.pop("DWARF_DSQG_KERNEL_DIR", None)
    else:
        env["DWARF_DSQG_KERNEL_DIR"] = str(kernel_dir)
    code = """
import importlib.util
import json
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('trainer_import_probe', Path(sys.argv[1]))
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(json.dumps({'kernel': module.DSQG_KERNEL_MODULE_PATH}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(TRAINER)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(json.loads(completed.stdout.splitlines()[-1])["kernel"]).resolve()


def test_trainer_imports_canonical_kernel_without_override() -> None:
    assert _imported_kernel_path(None) == (CANONICAL / "dsqg_attention_v20_bf16_se.py").resolve()


def test_trainer_imports_overlay_kernel_with_explicit_override() -> None:
    assert _imported_kernel_path(OVERLAY) == (OVERLAY / "dsqg_attention_v20_bf16_se.py").resolve()
