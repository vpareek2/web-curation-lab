"""Materialize every dataset used by configured OLMo Eval suites."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import HfApi
from olmo_eval.common.configs import expand_tasks
from olmo_eval.evals.tasks.common import get_task


def hub_repo_id(source) -> str | None:
    """Resolve the Hub repository behind a DataSource, including file loaders."""

    path = source.path.removeprefix("hf://")
    if path not in {"csv", "json", "parquet", "text"}:
        return path.removeprefix("datasets/")
    if not isinstance(source.data_files, str):
        return None
    match = re.match(r"^hf://datasets/([^/]+/[^/]+)(?:/|$)", source.data_files)
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested = json.loads(args.tasks_json)
    expanded = expand_tasks(requested)
    api = HfApi()
    revisions: dict[str, str] = {}
    tasks: list[dict] = []
    for spec in expanded:
        task = get_task(spec)
        config = task.config
        count = sum(1 for _ in task.instances)
        fewshot = task.get_fewshot()
        if config.num_fewshot and len(fewshot) < config.num_fewshot:
            raise RuntimeError(
                f"{spec} requested {config.num_fewshot} few-shot examples, "
                f"but only {len(fewshot)} loaded"
            )
        sources = []
        main_source = config.get_data_source()
        fewshot_source = config.get_fewshot_source()
        for source in (main_source, fewshot_source):
            if source is None:
                continue
            serialized = source.to_dict()
            # OLMo Eval represents fixed examples embedded in task source code
            # with names such as `olmes_arc_challenge_fixed`. DataSource's
            # generic type inference calls any bare name a Hub dataset even
            # though these aliases are never loaded from the Hub.
            if source is fewshot_source and (
                isinstance(config.fewshot_source, str)
                and config.fewshot_source.endswith("_fixed")
            ):
                serialized["source_type"] = "embedded"
            if serialized["source_type"] == "hf":
                repo_id = hub_repo_id(source)
                if repo_id is not None:
                    serialized["resolved_repo_id"] = repo_id
                    if repo_id not in revisions:
                        revisions[repo_id] = api.dataset_info(
                            repo_id,
                            revision=source.revision,
                        ).sha
            if serialized not in sources:
                sources.append(serialized)
        tasks.append(
            {
                "task": spec,
                "instances": count,
                "fewshot_instances": len(fewshot),
                "config": config.to_dict(),
                "sources": sources,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "requested_suites": requested,
                "expanded_tasks": expanded,
                "dataset_revisions": revisions,
                "tasks": tasks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
