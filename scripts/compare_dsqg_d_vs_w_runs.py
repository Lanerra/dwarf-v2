#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean


def _match(text: str, pattern: str, default=None):
    m = re.search(pattern, text)
    if not m:
        return default
    if m.lastindex == 1:
        return m.group(1)
    return m.groups()


def parse_run(name: str, run_dir: Path) -> dict:
    stdout = run_dir / "trainer.stdout.log"
    stderr = run_dir / "trainer.stderr.log"
    text = stdout.read_text(errors="replace")
    steps = []
    for m in re.finditer(
        r"\[ep(?P<epoch>\d+) step (?P<step>\d+)/(?P<total>\d+)\] "
        r"ce=(?P<ce>[0-9.]+) se_max=(?P<se>[0-9.]+) grad_norm=(?P<grad>[0-9.]+) "
        r"lr=(?P<lr>[0-9.e+-]+) (?P<toks>[0-9]+) tok/s routing_ent=(?P<routing>[0-9.]+)",
        text,
    ):
        steps.append({
            "step": int(m.group("step")),
            "total": int(m.group("total")),
            "ce": float(m.group("ce")),
            "se_max": float(m.group("se")),
            "grad_norm": float(m.group("grad")),
            "lr": float(m.group("lr")),
            "tok_s": int(m.group("toks")),
            "routing_ent": float(m.group("routing")),
        })
    params = _match(text, r"Parameters: ([0-9,]+) \(([0-9.]+M)\)", (None, None))
    peak = _match(text, r"peak_vram=([0-9]+)MB  elapsed=([0-9]+)s", (None, None))
    return {
        "name": name,
        "run_dir": str(run_dir),
        "stdout": str(stdout),
        "stderr_size": stderr.stat().st_size if stderr.exists() else None,
        "gpu": _match(text, r"GPU: (.+)", "unknown").strip(),
        "params": params[0],
        "params_m": params[1],
        "layout": _match(text, r"  Layout: ([^\n]+)", "unknown"),
        "dsqg_w": _match(text, r"(DSQG-W recomposer[^\n]+)", "DSQG-W line missing"),
        "step_count": len(steps),
        "avg_logged_tok_s": round(mean(s["tok_s"] for s in steps), 1) if steps else None,
        "first_step": steps[0] if steps else None,
        "mid_step": next((s for s in steps if s["step"] == 1000), None),
        "final_step": steps[-1] if steps else None,
        "val_ppl": float(_match(text, r"Ep 1/1 \| Val PPL ([0-9.]+)", "nan")),
        "passkey_pct": float(_match(text, r"Passkey mean=([0-9.]+)%", "nan")),
        "peak_vram_mb": int(peak[0]) if peak[0] is not None else None,
        "trainer_elapsed_s": int(peak[1]) if peak[1] is not None else None,
        "checkpoints": sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsqg-d", type=Path, required=True)
    parser.add_argument("--dsqg-w", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    d = parse_run("DSQG-D baseline", args.dsqg_d)
    w = parse_run("DSQG-W 2,6,final", args.dsqg_w)
    comparison = {
        "runs": [d, w],
        "d_over_w_tok_s": round(d["avg_logged_tok_s"] / w["avg_logged_tok_s"], 3),
        "w_minus_d_ppl": round(w["val_ppl"] - d["val_ppl"], 3),
        "w_over_d_params_m_ratio": round(float(w["params_m"].rstrip("M")) / float(d["params_m"].rstrip("M")), 3),
        "w_minus_d_vram_mb": w["peak_vram_mb"] - d["peak_vram_mb"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
