"""Command-line entry point for reproducible web-data pipeline operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_curation_lab.pipeline.stages.raw_cc.audit import load_audit_config, run_audit
from web_curation_lab.pipeline.stages.raw_cc.census import load_census_config, run_census
from web_curation_lab.pipeline.stages.raw_cc.pilot import load_pilot_config, run_pilot
from web_curation_lab.pipeline.stages.raw_cc.probe import load_probe_config, run_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="web-curation-data", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a bounded prefix of a 0_raw_cc WARC input.",
    )
    inspect_parser.add_argument("--config", type=Path, required=True)
    inspect_parser.add_argument(
        "--limit",
        type=int,
        help="Override the config's raw WARC-record limit.",
    )

    census_parser = subparsers.add_parser(
        "census",
        help="Count one complete 0_raw_cc WARC without writing training text.",
    )
    census_parser.add_argument("--config", type=Path, required=True)

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit ambiguous 0_raw_cc MIME and decoding decisions.",
    )
    audit_parser.add_argument("--config", type=Path, required=True)

    pilot_parser = subparsers.add_parser(
        "pilot",
        help="Download and census a deterministic multi-WARC 0_raw_cc pilot.",
    )
    pilot_parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        config = load_probe_config(args.config, limit_override=args.limit)
        summary = run_probe(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "census":
        config = load_census_config(args.config)
        summary = run_census(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "audit":
        config = load_audit_config(args.config)
        summary = run_audit(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "pilot":
        config, census_config = load_pilot_config(args.config)
        summary = run_pilot(config, census_config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
