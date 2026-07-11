#!/usr/bin/env python3
"""Seal a zero-match exact-token audit into an immutable trainable FWE manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_SCHEMA = "dsqg.fineweb_edu_dedup.v1"
SEALED_SCHEMA = "dsqg.fineweb_edu_dedup.repaired.v1"
SEALABLE_SOURCE_SCHEMAS = {SOURCE_SCHEMA, SEALED_SCHEMA}
DECONTAM_SCHEMA = "dwarf.dataset_decontam.v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def seal_zero_match_audit(
    *,
    artifact_path: Path | str,
    source_manifest_path: Path | str,
    decontam_path: Path | str,
    output_manifest_path: Path | str,
) -> dict[str, str]:
    artifact = Path(artifact_path).resolve()
    source_manifest_file = Path(source_manifest_path).resolve()
    decontam_file = Path(decontam_path).resolve()
    output_manifest = Path(output_manifest_path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"artifact is missing: {artifact}")
    if output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output_manifest}")

    source_manifest = load_json(source_manifest_file, "source manifest")
    if source_manifest.get("schema_version") not in SEALABLE_SOURCE_SCHEMAS:
        raise ValueError(
            "source manifest must use one of "
            f"{sorted(SEALABLE_SOURCE_SCHEMAS)}"
        )
    decontam = load_json(decontam_file, "decontamination report")
    if (
        decontam.get("schema_version") != DECONTAM_SCHEMA
        or decontam.get("summary", {}).get("match_count") != 0
        or decontam.get("matches") != []
    ):
        raise ValueError("zero-match exact-token decontamination report is required for sealing")

    sealed = dict(source_manifest)
    sealed.update(
        {
            "schema_version": SEALED_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "output": str(artifact),
            "output_size_bytes": artifact.stat().st_size,
            "output_sha256": sha256_file(artifact),
            "decontamination": {
                "status": "passed_zero_match_exact_token_audit",
                "report": str(decontam_file),
                "report_sha256": sha256_file(decontam_file),
                "source_manifest": str(source_manifest_file),
                "source_manifest_sha256": sha256_file(source_manifest_file),
            },
        }
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"artifact": str(artifact), "manifest": str(output_manifest), "decontam": str(decontam_file)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--decontam", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(seal_zero_match_audit(
        artifact_path=args.artifact,
        source_manifest_path=args.source_manifest,
        decontam_path=args.decontam,
        output_manifest_path=args.output_manifest,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
