#!/usr/bin/env python3
"""Run EEG events concurrently against one continuous-batching Qwen3.6 vLLM."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.auto_annotate.llm_soz_annotator import (  # noqa: E402
    DEFAULT_PRIVATE_MANIFEST,
    DEFAULT_PRIVATE_ROOT,
    DEFAULT_TUSZ_EVENT_INDEX,
    DEFAULT_TUSZ_MANIFEST,
    DEFAULT_TUSZ_ROOT,
    LLMEventTask,
    build_generic_tasks,
    build_private_tasks,
    build_tusz_tasks,
)
from scripts.index_qwen35_results import build_index  # noqa: E402


FINAL_SUCCESS = {
    "llm_candidate_ready",
    "llm_abstained",
    "llm_context_exhausted",
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _safe_event_dir(event_id: str) -> str:
    stem = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in event_id
    ).strip("._-")[:72]
    digest = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:10]
    return f"{stem or 'event'}_{digest}"


def _build_tasks(
    args: argparse.Namespace,
) -> tuple[list[LLMEventTask], dict[str, Any], Path, Path, tuple[str, ...]]:
    anchor_fields: tuple[str, ...] = ()
    if args.dataset == "private":
        manifest = Path(
            args.manifest or DEFAULT_PRIVATE_MANIFEST
        ).expanduser().resolve()
        eeg_root = Path(
            args.eeg_root or DEFAULT_PRIVATE_ROOT
        ).expanduser().resolve()
        tasks, audit = build_private_tasks(
            manifest,
            eeg_root,
            context_pre_s=args.context_pre_s or 60.0,
            context_post_s=args.context_post_s or 90.0,
            search_pre_s=args.search_pre_s or 45.0,
            search_post_s=args.search_post_s or 30.0,
        )
    elif args.dataset == "tusz":
        manifest = Path(
            args.manifest or DEFAULT_TUSZ_MANIFEST
        ).expanduser().resolve()
        eeg_root = Path(
            args.eeg_root or DEFAULT_TUSZ_ROOT
        ).expanduser().resolve()
        event_index = (
            None
            if args.all_tusz_events
            else Path(args.tusz_event_index).expanduser().resolve()
        )
        tasks, audit = build_tusz_tasks(
            manifest,
            eeg_root,
            event_index=event_index,
            context_pre_s=args.context_pre_s or 20.0,
            context_post_s=args.context_post_s or 45.0,
            search_pre_s=args.search_pre_s or 5.0,
            search_post_s=args.search_post_s or 10.0,
        )
    else:
        if not args.manifest or not args.eeg_root:
            raise ValueError(
                "--manifest and --eeg-root are required for generic datasets"
            )
        manifest = Path(args.manifest).expanduser().resolve()
        eeg_root = Path(args.eeg_root).expanduser().resolve()
        anchor_fields = tuple(
            field.strip()
            for field in args.generic_anchor_fields.split(",")
            if field.strip()
        )
        if not anchor_fields:
            raise ValueError("--generic-anchor-fields selects no fields")
        tasks, audit = build_generic_tasks(
            manifest,
            eeg_root,
            dataset_name=args.dataset_name,
            anchor_fields=anchor_fields,
            context_pre_s=args.context_pre_s or 30.0,
            context_post_s=args.context_post_s or 60.0,
            search_pre_s=args.search_pre_s or 10.0,
            search_post_s=args.search_post_s or 20.0,
        )
    requested = {
        value.strip()
        for value in args.event_ids.split(",")
        if value.strip()
    }
    if requested:
        tasks = [task for task in tasks if task.event_id in requested]
    tasks = tasks[args.start_row :]
    if args.max_events >= 0:
        tasks = tasks[: args.max_events]
    if not tasks:
        raise ValueError("no events selected for the vLLM queue")
    return tasks, audit, manifest, eeg_root, anchor_fields


def _prior_successes(
    paths: Sequence[str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], set[str]]:
    successes: dict[tuple[str, str], dict[str, Any]] = {}
    roots: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        for row in _read_jsonl(path):
            if row.get("record_status") not in FINAL_SUCCESS:
                continue
            key = (
                str(row.get("dataset") or ""),
                str(row.get("event_id") or ""),
            )
            if not all(key):
                continue
            successes[key] = row
            if row.get("run_root"):
                roots.add(str(Path(str(row["run_root"])).expanduser().resolve()))
    return successes, roots


def _shard_success(shard: Path) -> dict[str, Any] | None:
    records = shard / "records"
    if not records.is_dir():
        return None
    for path in records.glob("*.json"):
        if path.name.endswith(".core_checkpoint.json"):
            continue
        payload = _read_json(path)
        if payload.get("record_status") in FINAL_SUCCESS:
            return payload
    return None


def _health_check(base_url: str, expected_model: str, timeout_s: float) -> None:
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint += "/models"
    else:
        endpoint += "/v1/models"
    with urllib.request.urlopen(endpoint, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    model_ids = {
        str(item.get("id") or "")
        for item in payload.get("data") or []
        if isinstance(item, Mapping)
    }
    if expected_model not in model_ids:
        raise RuntimeError(
            f"vLLM is reachable but {expected_model!r} is not served; "
            f"available={sorted(model_ids)}"
        )


def _event_command(
    args: argparse.Namespace,
    task: LLMEventTask,
    *,
    shard: Path,
    manifest: Path,
    eeg_root: Path,
    anchor_fields: Sequence[str],
) -> list[str]:
    rtk = shutil.which("rtk")
    if not rtk:
        raise FileNotFoundError("rtk is required for Qwen3.6 event workers")
    command = [
        rtk,
        str(ROOT / ".venv-qwen35" / "bin" / "python"),
        "-u",
        str(ROOT / "scripts" / "run_qwen36_vllm_annotation.py"),
        "--dataset",
        args.dataset,
        "--manifest",
        str(manifest),
        "--eeg-root",
        str(eeg_root),
        "--event-id",
        task.event_id,
        "--max-events",
        "1",
        "--output-dir",
        str(shard),
        "--resume",
        "--no-fail-fast",
    ]
    if args.dataset == "tusz":
        if args.all_tusz_events:
            command.append("--all-tusz-events")
        else:
            command.extend(
                ["--tusz-event-index", str(Path(args.tusz_event_index).resolve())]
            )
    elif args.dataset == "generic":
        command.extend(
            [
                "--dataset-name",
                args.dataset_name,
                "--generic-anchor-fields",
                ",".join(anchor_fields),
            ]
        )
    for option, value in (
        ("--context-pre-s", args.context_pre_s),
        ("--context-post-s", args.context_post_s),
        ("--search-pre-s", args.search_pre_s),
        ("--search-post-s", args.search_post_s),
    ):
        if value is not None:
            command.extend([option, str(value)])
    command.extend(args.annotator_arg)
    return command


def _queue_progress(
    output_dir: Path,
    plans: Sequence[Mapping[str, Any]],
    active: Sequence[str],
) -> None:
    statuses = [str(plan.get("status") or "") for plan in plans]
    _atomic_json(
        output_dir / "progress.json",
        {
            "schema_version": "qwen36_vllm_queue_progress_v1",
            "selected_events": len(plans),
            "completed_events": sum(
                status in {"completed", "reused_current", "reused_prior"}
                for status in statuses
            ),
            "successful_events": sum(
                status in {"completed", "reused_current", "reused_prior"}
                for status in statuses
            ),
            "failed_events": sum(
                status in {"failed", "launch_failed"} for status in statuses
            ),
            "queued_events": sum(
                status in {"planned", "running"} for status in statuses
            ),
            "active_event_ids": sorted(active),
            "worker_concurrency": sum(status == "running" for status in statuses),
            "updated_at_unix_s": time.time(),
            "training_export_allowed": False,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("private", "tusz", "generic"), required=True
    )
    parser.add_argument("--dataset-name", default="generic")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--eeg-root", default="")
    parser.add_argument(
        "--tusz-event-index", default=str(DEFAULT_TUSZ_EVENT_INDEX)
    )
    parser.add_argument("--all-tusz-events", action="store_true")
    parser.add_argument(
        "--generic-anchor-fields", default="coarse_candidate_s"
    )
    parser.add_argument("--event-ids", default="")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--context-pre-s", type=float)
    parser.add_argument("--context-post-s", type=float)
    parser.add_argument("--search-pre-s", type=float)
    parser.add_argument("--search-post-s", type=float)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prior-success-index",
        action="append",
        default=[],
        help=(
            "Existing successes.jsonl whose completed events should be preserved "
            "and skipped. Repeat to combine earlier runs."
        ),
    )
    parser.add_argument(
        "--rerun-prior-successes",
        action="store_true",
        help="Ignore --prior-success-index for a homogeneous Qwen3.6 rerun.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "QWEN36_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"
        ),
    )
    parser.add_argument(
        "--served-model",
        default=os.environ.get(
            "QWEN36_VLLM_MODEL_NAME", "qwen36-soz"
        ),
    )
    parser.add_argument("--health-timeout-s", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--annotator-arg",
        action="append",
        default=[],
        help="One additional raw argument forwarded to every event worker.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_row < 0 or args.workers < 1:
        raise SystemExit("--start-row must be >=0 and --workers must be >=1")
    output_dir = Path(args.output_dir).expanduser().resolve()
    shards_dir = output_dir / "shards"
    logs_dir = output_dir / "logs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tasks, audit, manifest, eeg_root, anchor_fields = _build_tasks(args)
    prior, prior_roots = _prior_successes(args.prior_success_index)
    plans: list[dict[str, Any]] = []
    for position, task in enumerate(tasks, start=1):
        shard = shards_dir / _safe_event_dir(task.event_id)
        prior_row = prior.get((task.dataset, task.event_id))
        current = _shard_success(shard)
        if prior_row is not None and not args.rerun_prior_successes:
            status = "reused_prior"
        elif current is not None:
            status = "reused_current"
        else:
            status = "planned"
        plans.append(
            {
                "position": position,
                "dataset": task.dataset,
                "event_id": task.event_id,
                "source_file": task.edf_path,
                "navigation_anchor_s": task.anchor_s,
                "status": status,
                "returncode": 0 if status.startswith("reused_") else None,
                "shard_output_dir": str(shard),
                "log_path": str(logs_dir / f"{_safe_event_dir(task.event_id)}.log"),
                "prior_record_path": (
                    str(prior_row.get("record_path") or "")
                    if prior_row is not None
                    else ""
                ),
                "prior_model_release": (
                    str(prior_row.get("model_release") or "")
                    if prior_row is not None
                    else ""
                ),
                "error": "",
            }
        )
    _atomic_json(
        output_dir / "queue_config.json",
        {
            "schema_version": "qwen36_vllm_queue_config_v1",
            "dataset": args.dataset,
            "dataset_name": args.dataset_name,
            "manifest_audit": audit,
            "workers": args.workers,
            "base_url": args.base_url,
            "served_model": args.served_model,
            "prior_success_indexes": [
                str(Path(path).expanduser().resolve())
                for path in args.prior_success_index
            ],
            "prior_run_roots": sorted(prior_roots),
            "preserve_prior_successes": not args.rerun_prior_successes,
            "training_export_allowed": False,
        },
    )
    _atomic_jsonl(output_dir / "queue_plan.jsonl", plans)
    _queue_progress(output_dir, plans, [])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "events": len(plans),
                    "reused_prior": sum(
                        plan["status"] == "reused_prior" for plan in plans
                    ),
                    "reused_current": sum(
                        plan["status"] == "reused_current" for plan in plans
                    ),
                    "planned": sum(
                        plan["status"] == "planned" for plan in plans
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    _health_check(
        args.base_url, args.served_model, float(args.health_timeout_s)
    )
    task_by_event = {task.event_id: task for task in tasks}
    active: set[str] = set()

    def run_one(plan: dict[str, Any]) -> tuple[int, str]:
        task = task_by_event[str(plan["event_id"])]
        command = _event_command(
            args,
            task,
            shard=Path(str(plan["shard_output_dir"])),
            manifest=manifest,
            eeg_root=eeg_root,
            anchor_fields=anchor_fields,
        )
        environment = dict(os.environ)
        environment["QWEN36_VLLM_BASE_URL"] = args.base_url
        environment["QWEN36_VLLM_MODEL_NAME"] = args.served_model
        log_path = Path(str(plan["log_path"]))
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n=== queue attempt {time.time():.3f} ===\n"
            )
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return int(completed.returncode), ""

    pending = [plan for plan in plans if plan["status"] == "planned"]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures: dict[
            concurrent.futures.Future[tuple[int, str]], dict[str, Any]
        ] = {}
        pending_iterator = iter(pending)

        def submit_next() -> bool:
            try:
                plan = next(pending_iterator)
            except StopIteration:
                return False
            plan["status"] = "running"
            active.add(str(plan["event_id"]))
            future = executor.submit(run_one, plan)
            futures[future] = plan
            return True

        for _ in range(min(args.workers, len(pending))):
            submit_next()
        _atomic_jsonl(output_dir / "queue_plan.jsonl", plans)
        _queue_progress(output_dir, plans, active)
        while futures:
            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                plan = futures.pop(future)
                try:
                    returncode, error = future.result()
                    plan["returncode"] = returncode
                    plan["error"] = error
                    plan["status"] = (
                        "completed" if returncode == 0 else "failed"
                    )
                except Exception as exc:
                    plan["returncode"] = 127
                    plan["status"] = "launch_failed"
                    plan["error"] = f"{type(exc).__name__}: {exc}"
                active.discard(str(plan["event_id"]))
                submit_next()
                _atomic_jsonl(output_dir / "queue_plan.jsonl", plans)
                _queue_progress(output_dir, plans, active)
                print(
                    f"[{plan['position']}/{len(plans)}] {plan['event_id']} "
                    f"{plan['status']}",
                    flush=True,
                )

    index_roots = [*sorted(prior_roots), str(output_dir)]
    summary = build_index(index_roots, output_dir / "indexes")
    _atomic_json(output_dir / "queue_summary.json", summary)
    failures = [
        plan
        for plan in plans
        if plan["status"] in {"failed", "launch_failed"}
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
