"""Private subprocess entry point for diagnostic-pilot workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from kvcot.discovery.attempt_artifacts import atomic_write_json
from kvcot.discovery.diagnostic_pilot_contract import verify_canonical_hash
from kvcot.discovery.discovery_config import load_discovery_config
from kvcot.discovery.manifest import B2AOneExampleManifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kvcot.discovery.diagnostic_pilot_worker_entry")
    parser.add_argument("--role", required=True, choices=("fullkv", "diagnostic-rkv"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--candidate-ordinal", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fullkv-result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output).resolve()
    try:
        prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
        verify_canonical_hash(prompts)
        manifests = prompts["manifests"]
        if not 0 <= args.candidate_ordinal < len(manifests):
            raise ValueError("candidate ordinal out of range")
        manifest = B2AOneExampleManifest.model_validate(manifests[args.candidate_ordinal])
        config = load_discovery_config(args.config)
        if args.role == "fullkv":
            from kvcot.discovery.diagnostic_pilot_workers import run_fullkv_diagnostic_worker

            result = run_fullkv_diagnostic_worker(config, manifest)
        else:
            if not args.fullkv_result:
                raise ValueError("diagnostic-rkv requires --fullkv-result")
            fullkv = json.loads(Path(args.fullkv_result).read_text(encoding="utf-8"))
            from kvcot.discovery.diagnostic_pilot_workers import run_rkv_diagnostic_worker

            result = run_rkv_diagnostic_worker(config, manifest, fullkv)
        result["candidate_ordinal"] = args.candidate_ordinal
        atomic_write_json(output, result)
        return 0
    except BaseException as exc:
        failure = {
            "role": args.role,
            "candidate_ordinal": args.candidate_ordinal,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "traceback": traceback.format_exc(),
            "retry_allowed": False,
        }
        try:
            atomic_write_json(output.with_suffix(output.suffix + ".failure.json"), failure)
        except Exception:
            pass
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
