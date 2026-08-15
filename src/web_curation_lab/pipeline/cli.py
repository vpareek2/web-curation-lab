"""Command-line entry point for reproducible web-data pipeline operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_curation_lab.pipeline.stages.raw_cc.audit import load_audit_config, run_audit
from web_curation_lab.pipeline.stages.raw_cc.census import load_census_config, run_census
from web_curation_lab.pipeline.stages.raw_cc.materialize import (
    load_materialize_config,
    run_materialization,
)
from web_curation_lab.pipeline.stages.raw_cc.pilot import load_pilot_config, run_pilot
from web_curation_lab.pipeline.stages.raw_cc.probe import load_probe_config, run_probe
from web_curation_lab.training.datatrove_loader import create_training_view


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

    materialize_parser = subparsers.add_parser(
        "materialize",
        help="Materialize a resumable 0_raw_cc inventory into DataTrove token shards.",
    )
    materialize_parser.add_argument("--config", type=Path, required=True)
    materialize_parser.add_argument(
        "--limit",
        type=int,
        help="Process only this deterministic prefix of the inventory.",
    )

    view_parser = subparsers.add_parser(
        "create-training-view",
        help="Freeze an exact batch-aligned view over DataTrove .ds shards.",
    )
    view_parser.add_argument("--dataset-dir", type=Path, required=True)
    view_parser.add_argument("--output", type=Path, required=True)
    view_parser.add_argument("--sequence-length", type=int, required=True)
    view_parser.add_argument("--global-batch-size", type=int, required=True)
    view_parser.add_argument("--training-steps", type=int, required=True)
    view_parser.add_argument("--seed", type=int, default=42)
    view_parser.add_argument("--token-size-bytes", type=int, choices=(2, 4), default=2)
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
    if args.command == "materialize":
        config = load_materialize_config(args.config)
        summary = run_materialization(config, limit=args.limit)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "create-training-view":
        summary = create_training_view(
            dataset_dir=args.dataset_dir,
            output_path=args.output,
            sequence_length=args.sequence_length,
            global_batch_size=args.global_batch_size,
            training_steps=args.training_steps,
            seed=args.seed,
            token_size_bytes=args.token_size_bytes,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
