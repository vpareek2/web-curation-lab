"""Command-line interface for Web Curation Lab evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .assets import prepare_assets
from .config import load_config
from .general import run_general
from .orchestrator import run_all
from .statistics import run_statistics


def _gpus(value: int | None) -> int:
    detected = torch.cuda.device_count()
    result = value if value is not None else max(1, detected)
    if result < 1:
        raise ValueError("num-gpus must be at least one")
    if detected and result > detected:
        raise ValueError(f"Requested {result} GPUs, but only {detected} are visible")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="web-curation-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-assets")
    prepare.add_argument("--config", type=Path, required=True)

    for name in ("statistics", "general", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name in {"general", "run"}:
            command.add_argument("--num-gpus", type=int)

    args = parser.parse_args()
    if args.command == "prepare-assets":
        print(prepare_assets(args.config))
        return
    config = load_config(args.config)
    if args.command == "statistics":
        print(run_statistics(model_path=args.model, config=config, output_path=args.output))
    elif args.command == "general":
        print(
            run_general(
                model_path=args.model,
                config=config,
                output_dir=args.output,
                num_gpus=_gpus(args.num_gpus),
            )
        )
    else:
        print(
            run_all(
                model_path=args.model,
                config_path=args.config,
                config=config,
                output_dir=args.output,
                num_gpus=_gpus(args.num_gpus),
            )
        )


if __name__ == "__main__":
    main()
