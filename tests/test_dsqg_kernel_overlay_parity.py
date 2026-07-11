from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_dsqg_kernel_overlay.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_dsqg_kernel_overlay", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parity_launcher_accepts_explicit_kernel_paths_and_tolerances(tmp_path: Path) -> None:
    module = _load_module()

    args = module.parse_args(
        [
            "--baseline-kernel", "/tmp/baseline.py",
            "--candidate-kernel", "/tmp/candidate.py",
            "--json-out", str(tmp_path / "result.json"),
            "--batch-size", "2",
            "--seq-len", "2048",
            "--atol", "0.003",
            "--rtol", "0.02",
        ]
    )

    assert args.batch_size == 2
    assert args.seq_len == 2048
    assert args.atol == 0.003
    assert args.rtol == 0.02
