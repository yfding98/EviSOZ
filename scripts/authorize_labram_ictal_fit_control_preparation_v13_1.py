#!/usr/bin/env python3
"""Issue the closed v13.1-v3 authorization for fit-only LaBraM controls.

This command does not materialize targets, train a model, open the I-gate, or
run an evaluator.  It verifies every frozen input byte, executes the exact
non-training test suite, builds the six fixed output bindings, asks the formal
orchestrator to validate the resulting closed payload, and publishes one
canonical JSON file with kernel-enforced ``O_EXCL`` semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_labram_ictal_fit_control_preparation_v13 as orchestrator  # noqa: E402


_SELECTION_ORDER = ("fold0", "fold1", "fold2", "fold3", "fold4", "final")
_PROTOCOL = Path(
    "research/02_method/"
    "labram_k31_source_native_confirmation_protocol_v13_20260811_zh.md"
)
_PROTOCOL_SHA256 = "d109c9ba8841ec7277260138f3e6d4111caf5ec9016e5cd451265cf87fa8759b"
_AMENDMENT = Path(
    "research/02_method/"
    "labram_k31_source_native_confirmation_protocol_v13_1_amendment_20260811_zh.md"
)
_AMENDMENT_SHA256 = "1747cdf701ad13be849a1e12a0fed511d3cf17d6dbd9ac1e7e3c6066cdf3c968"
_PREPROCESSING = Path(
    "outputs/preprocessing_parity_formal_v2_1_20260809/selection-capability"
)
_PREPROCESSING_SELECTION_SHA256 = (
    "b4aa73bff2800f12186085976a5655db6882a38232d775d11234efa387171485"
)
_PREPROCESSING_PROTOCOL_RECEIPT_SHA256 = (
    "9a75dd2f3293d4d944380c0d82dcfca6a95e332f3b999e32e52b15d89622a196"
)
_TARGET_SNAPSHOT = Path(
    "outputs/tusz_ictal_prediction_artifacts_formal_v4_20260809/final/native"
)
_TARGET_SNAPSHOT_MANIFEST_SHA256 = (
    "bc22681928e596ef6564af51f54215e96a9560a21cdeaedef043ccd324596cba"
)
_TARGET_SNAPSHOT_RECEIPT_SHA256 = (
    "e216338d5112a67d20fcba5d545834af2b84c8896a8713b9919866e839c7953a"
)
_SUPERSESSION_SIDECAR = Path(orchestrator._V2_SUPERSESSION_SIDECAR_PATH)

_PINNED_SELECTIONS = {
    "fold0": {
        "k31_manifest_sha256": "c183acd41ea91eb0164180e80e61fe67820c84d0cd72311493327b462741064d",
        "k31_checkpoint_sha256": "e6f318d413ff0a5ca7bf0150fd2906e26bb54dee9116e4564de499dc1216bf28",
        "training_manifest_bundle_sha256": "88857946163e6583795079810d65d078d0d3c325a7dfe1bca86fe29c68e78200",
        "training_manifest_source_sha256": "0cd023bcee1d58dd7e254427837dc0a942b2310a31882134b80d61469ec6510a",
        "training_token_corpus_index_sha256": "fae930dbbaed5e80af12723909110ab05b3ab7ce86562aa357aa78e8bbbd59b1",
    },
    "fold1": {
        "k31_manifest_sha256": "1ffd565f5a80259e202f62878ac7585ef28e6fccff44bb1fdaea4aae79a7e1eb",
        "k31_checkpoint_sha256": "a351d7766175617906939ca44fffbec801957670d6bdbb5b89e90dd46ad40d2f",
        "training_manifest_bundle_sha256": "f03cb001fcfe32ccb05a646813a6941014212f71f8bc6e11a887c852e3f585c7",
        "training_manifest_source_sha256": "b86a7d982fc1e060cd0dea9086f6adac6a4f3ea9874ca4d1c6206da960e9ac7f",
        "training_token_corpus_index_sha256": "bb130ae50e857798a80be2cd551565251db85d6fb3ba95a8ba5c0eb49033de72",
    },
    "fold2": {
        "k31_manifest_sha256": "c3147f4542a02fdb255e3e33ae396674b744b0fe18a60e15340927c28685c468",
        "k31_checkpoint_sha256": "748f1e479b2b256f5718c21f424662e0f707dafa898ccc70ba66494c32483ca5",
        "training_manifest_bundle_sha256": "15365ee222f2ef041e8ba54266a82208d7798f425a860a04e61b560cf01678ca",
        "training_manifest_source_sha256": "a0712976693e64dba17b09538f064f2b7197d3464447c90b5ef450a800775b6b",
        "training_token_corpus_index_sha256": "f52fc4d9d5f88ca6cd94d358a1ad385d955ac1d3c2b83b27f9ba677a890aab52",
    },
    "fold3": {
        "k31_manifest_sha256": "d7474077616be3aad24c843f8a39df2b9c8e6e5e16133aff22576fb5e8cc0efc",
        "k31_checkpoint_sha256": "209b49563d292b1b69ade53ef1d91984976aafd0355b421aee5e9e526885c890",
        "training_manifest_bundle_sha256": "d0b158c80c76f5302ce47e8074b5901bf81e646855493be7760657c70f49bbb4",
        "training_manifest_source_sha256": "06af3bec1714c93a110ca7bcdcd83aeac6cb2940492ed442e58f82e836704094",
        "training_token_corpus_index_sha256": "39ae5bd86a21bec577e01ebd8d68bf6d31b783b1d3ed1ac1a0f98b26601c701f",
    },
    "fold4": {
        "k31_manifest_sha256": "6236fad9ad53951a39976f41093f55f8a85be5c51d2fcd788a3333adb8b03cb9",
        "k31_checkpoint_sha256": "c4e2195e34fea82ae3cb61ee0513df7950e684d90712db9b485b89f1ceb9b3a6",
        "training_manifest_bundle_sha256": "81eb77a10479778a149fbf02546c039b4141a242a24e93eaf0d46af7ec5e7381",
        "training_manifest_source_sha256": "d4de03763212278463da6a67373b0af6f084e9abc1c67b91e41c7ab7787923d9",
        "training_token_corpus_index_sha256": "431b13e959384da011ac7b2353836c74d8b1a3c3e41556871fa3bb4983c40fcf",
    },
    "final": {
        "k31_manifest_sha256": "906f415c89d5e6ec0eb8059b00dada7d2ae50ed0d5fa3a020fe45954521ce2c8",
        "k31_checkpoint_sha256": "d8b5b494a30431ab2ecfecc2ea61af27c3f7703352ee2c9fa7676242d9957d24",
        "training_manifest_bundle_sha256": "73e821d08805c3a7e8ae75011dd98fe10c388d7291c74881286438e91cacc35f",
        "training_manifest_source_sha256": "d5329b9231ecea7aaae6e126f5cd7a17a51f21b950025b32369592379acf8cb8",
        "training_token_corpus_index_sha256": "a7d672e3228cdc71fafb46e910033f6a5302a9e2e0a5f5716f7f4c8292ecfc26",
    },
}

_CODE_FILES = (
    "src/__init__.py",
    "scripts/authorize_labram_ictal_fit_control_preparation_v13_1.py",
    "scripts/run_labram_ictal_fit_control_preparation_v13.py",
    "scripts/materialize_labram_ictal_fit_only_targets_v13.py",
    "scripts/materialize_labram_ictal_fit_token_view_v13.py",
    "scripts/materialize_tusz_ictal_token_cache.py",
    "scripts/train_labram_ictal_capacity_matched_channel_control_v13.py",
    "scripts/train_labram_ictal_matched_independent_control_v13.py",
    "scripts/_v13_minimal_import.py",
    "src/soz/cached_concept_training.py",
    "src/soz/concept_losses.py",
    "src/soz/concept_metrics.py",
    "src/soz/concept_token_io.py",
    "src/soz/concept_training.py",
    "src/soz/data/tusz_training.py",
    "src/soz/formal_token_corpus.py",
    "src/soz/geometry.py",
    "src/soz/ictal_fit_primitives_v13.py",
    "src/soz/ictal_fit_only_targets_v13.py",
    "src/soz/ictal_fit_only_consumer_v13.py",
    "src/soz/ictal_fit_token_view_v13.py",
    "src/soz/ictal_fit_token_view_consumer_v13.py",
    "src/soz/ictal_matched_control_v13.py",
    "src/soz/ictal_recovery_oof_v1_2.py",
    "src/soz/preprocessing_parity.py",
    "src/soz/models/concept_heads.py",
    "src/soz/models/foundation.py",
    "src/soz/models/labram.py",
    "tests/test_ictal_matched_control_v13.py",
    "tests/test_ictal_fit_control_orchestrator_v13.py",
)

_PASSED_RE = re.compile(r"(?:^|\s)([0-9]+) passed(?:[ ,]|$)")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, expected_sha256: str, *, field: str) -> Path:
    source = (ROOT / path).resolve(strict=True)
    if ROOT not in source.parents or source.is_symlink() or not source.is_file():
        raise ValueError(f"{field} must be a regular workspace file")
    if _file_sha256(source) != expected_sha256:
        raise ValueError(f"{field} SHA differs from its frozen value")
    return source


def _load_json(path: Path, *, field: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _selection_source_paths(selection: str) -> tuple[Path, Path, Path]:
    if selection == "final":
        training = Path(
            "outputs/tusz_ictal_master_manifest_v4_1_20260809_current_preflight"
        )
        corpus = Path("outputs/tusz_ictal_token_corpus_formal_v4_20260809/master")
    else:
        index = int(selection.removeprefix("fold"))
        training = Path(
            "outputs/tusz_ictal_oof_fold_manifests_v4_1_20260809_current_preflight"
        ) / f"fold_{index}"
        corpus = Path("outputs/tusz_ictal_token_corpus_formal_v4_20260809") / f"fold_{index}"
    k31 = Path("outputs/labram_ictal_k31_oof_recovery_v1_2_20260810") / selection
    return k31, training, corpus


def _verify_tests() -> dict[str, object]:
    command_text = str(orchestrator._TEST_COMMAND)
    expected_passed = int(orchestrator._EXPECTED_TEST_PASSED)
    words = shlex.split(command_text)
    if not words or words[0] != "pytest":
        raise RuntimeError("Orchestrator test command must be a fixed pytest command")
    command = [sys.executable, "-m", "pytest", *words[1:]]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise RuntimeError("v13.1 authorization tests failed")
    matches = _PASSED_RE.findall(completed.stdout)
    if not matches or int(matches[-1]) != expected_passed:
        raise RuntimeError("v13.1 authorization test count differs from policy")
    return {
        "command": command_text,
        "expected_passed": expected_passed,
        "passed": True,
    }


def _selection_rows(output_root: Path) -> list[dict[str, object]]:
    rows = []
    for selection in _SELECTION_ORDER:
        pins = _PINNED_SELECTIONS[selection]
        k31, training, corpus = _selection_source_paths(selection)
        k31_manifest = _require_file(
            k31 / "recovery_run.json",
            pins["k31_manifest_sha256"],
            field=f"{selection}.k31_manifest",
        )
        k31_payload = _load_json(k31_manifest, field=f"{selection}.k31_manifest")
        checkpoint_name = k31_payload.get("checkpoint_filename")
        if checkpoint_name != "model.safetensors":
            raise ValueError(f"{selection} checkpoint filename changed")
        checkpoint = _require_file(
            k31 / str(checkpoint_name),
            pins["k31_checkpoint_sha256"],
            field=f"{selection}.k31_checkpoint",
        )
        manifest_file = _require_file(
            training / "manifest.json",
            pins["training_manifest_bundle_sha256"],
            field=f"{selection}.training_manifest_bundle",
        )
        receipt_file = _require_file(
            training / "receipt.json",
            pins["training_manifest_source_sha256"],
            field=f"{selection}.training_manifest_source",
        )
        index_file = _require_file(
            corpus / "index.json",
            pins["training_token_corpus_index_sha256"],
            field=f"{selection}.training_token_index",
        )
        expected_manifest_values = {
            "selection": selection,
            "checkpoint_sha256": pins["k31_checkpoint_sha256"],
            "training_manifest_sha256": pins["training_manifest_source_sha256"],
            "training_corpus_index_sha256": pins["training_token_corpus_index_sha256"],
            "target_snapshot_manifest_sha256": _TARGET_SNAPSHOT_MANIFEST_SHA256,
            "target_snapshot_receipt_sha256": _TARGET_SNAPSHOT_RECEIPT_SHA256,
            "target_semantics": "tusz_bipolar_edge_time_involvement_not_soz",
        }
        changed = tuple(
            name
            for name, expected in expected_manifest_values.items()
            if k31_payload.get(name) != expected
        )
        if changed:
            raise ValueError(f"{selection} k31 lineage changed: {changed}")
        relative_root = output_root.relative_to(ROOT)
        rows.append(
            {
                "selection": selection,
                "k31_bundle": str(k31),
                "k31_manifest_sha256": _file_sha256(k31_manifest),
                "k31_checkpoint": str(checkpoint.relative_to(ROOT)),
                "k31_checkpoint_sha256": _file_sha256(checkpoint),
                "training_manifest_bundle": str(training),
                "training_manifest_bundle_sha256": _file_sha256(manifest_file),
                "training_manifest_source_sha256": _file_sha256(receipt_file),
                "training_token_corpus": str(corpus),
                "training_token_corpus_index_sha256": _file_sha256(index_file),
                "fit_target_output": str(relative_root / "fit_targets" / selection),
                "fit_token_output": str(relative_root / "fit_token_views" / selection),
                "capacity_control_output": str(
                    relative_root / "capacity_controls" / selection / "bundle"
                ),
                "independent_control_output": str(
                    relative_root / "naked_controls" / selection / "bundle"
                ),
            }
        )
    return rows


def _build_authorization(output_root: Path) -> dict[str, object]:
    _require_file(_PROTOCOL, _PROTOCOL_SHA256, field="protocol")
    _require_file(_AMENDMENT, _AMENDMENT_SHA256, field="amendment")
    _require_file(
        Path(orchestrator._V2_AUTHORIZATION_PATH),
        orchestrator._V2_AUTHORIZATION_SHA256,
        field="superseded_v2_authorization",
    )
    orchestrator._verify_superseded_v2_state()
    preprocessing = (ROOT / _PREPROCESSING).resolve(strict=True)
    snapshot = (ROOT / _TARGET_SNAPSHOT).resolve(strict=True)
    if not preprocessing.is_dir() or not snapshot.is_dir():
        raise ValueError("A frozen source bundle is missing")
    selection_sha = _file_sha256(preprocessing / "selection.json")
    if selection_sha != _PREPROCESSING_SELECTION_SHA256:
        raise ValueError("Preprocessing selection artifact changed")
    preprocessing_receipt = _load_json(
        preprocessing / "receipt.json", field="preprocessing.receipt"
    )
    if (
        preprocessing_receipt.get("protocol_receipt_sha256")
        != _PREPROCESSING_PROTOCOL_RECEIPT_SHA256
    ):
        raise ValueError("Preprocessing protocol receipt changed")
    for name, expected in (
        ("manifest", _TARGET_SNAPSHOT_MANIFEST_SHA256),
        ("receipt", _TARGET_SNAPSHOT_RECEIPT_SHA256),
    ):
        if _file_sha256(snapshot / f"{name}.json") != expected:
            raise ValueError(f"Target snapshot {name} changed")

    tests = _verify_tests()
    code_files = []
    for relative in _CODE_FILES:
        source = (ROOT / relative).resolve(strict=True)
        if ROOT not in source.parents or source.is_symlink() or not source.is_file():
            raise ValueError(f"Authorized code file is unsafe: {relative}")
        code_files.append({"path": relative, "sha256": _file_sha256(source)})
    payload = {
        "schema_version": orchestrator._AUTHORIZATION_SCHEMA,
        "authorization_status": (
            "AUTHORIZED_FIT_ONLY_PREPARATION_NO_GATE_NO_EVALUATION"
        ),
        "protocol": {"path": str(_PROTOCOL), "sha256": _PROTOCOL_SHA256},
        "amendment": {"path": str(_AMENDMENT), "sha256": _AMENDMENT_SHA256},
        "execution_policy": dict(orchestrator._FROZEN_EXECUTION_POLICY),
        "sandbox_policy": dict(orchestrator._expected_sandbox_policy()),
        "target_broker_trace_policy": dict(
            orchestrator._expected_target_broker_trace_policy()
        ),
        "supersession": dict(orchestrator._expected_supersession()),
        "preprocessing": {
            "bundle": str(_PREPROCESSING),
            "selection_artifact_sha256": _PREPROCESSING_SELECTION_SHA256,
            "protocol_receipt_sha256": _PREPROCESSING_PROTOCOL_RECEIPT_SHA256,
        },
        "source_target_snapshot": {
            "bundle": str(_TARGET_SNAPSHOT),
            "manifest_sha256": _TARGET_SNAPSHOT_MANIFEST_SHA256,
            "receipt_sha256": _TARGET_SNAPSHOT_RECEIPT_SHA256,
        },
        "selections": _selection_rows(output_root),
        "code_files": code_files,
        "test_receipt": tests,
        "output_root": str(output_root.relative_to(ROOT)),
    }
    orchestrator._validate_authorization(payload)
    return payload


def _safe_missing_workspace_path(value: Path, *, field: str) -> Path:
    lexical = Path(os.path.abspath(value))
    if lexical == ROOT or ROOT not in lexical.parents or lexical.name in {"", ".", ".."}:
        raise ValueError(f"{field} must be a concrete workspace child")
    if os.path.lexists(lexical):
        raise FileExistsError(f"{field} already exists: {lexical}")
    return lexical


def _write_exclusive(path: Path, raw: bytes) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("Authorization output parent must be a regular directory")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--authorized-output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_file = _safe_missing_workspace_path(args.output_file, field="output_file")
    output_root = _safe_missing_workspace_path(
        args.authorized_output_root, field="authorized_output_root"
    )
    orchestrator._assert_direct_formal_output_child(
        output_file, field="output_file"
    )
    orchestrator._assert_direct_formal_output_child(
        output_root, field="authorized_output_root"
    )
    if output_file == output_root or output_file in output_root.parents:
        raise ValueError("Authorization file and future output root overlap")
    sidecar = _safe_missing_workspace_path(
        _SUPERSESSION_SIDECAR, field="supersession_sidecar"
    )
    orchestrator._assert_direct_formal_output_child(
        sidecar, field="supersession_sidecar"
    )
    if sidecar == output_root or sidecar in output_root.parents or sidecar == output_file:
        raise ValueError("Supersession sidecar overlaps another formal output")
    payload = _build_authorization(output_root)
    raw = _canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    if not args.preflight_only:
        _write_exclusive(output_file, raw)
        sidecar_payload = {
            "schema_version": "soz_ictal_fit_control_authorization_supersession_v13_1_v3",
            "supersession_record": orchestrator._expected_supersession(),
            "superseded_by": {
                "path": str(output_file.relative_to(ROOT)),
                "sha256": digest,
            },
        }
        _write_exclusive(sidecar, _canonical_json_bytes(sidecar_payload))
    print(
        json.dumps(
            {
                "schema_version": "soz_ictal_fit_control_authorization_issue_receipt_v13_1_v3",
                "authorization_file": str(output_file.relative_to(ROOT)),
                "authorization_sha256": digest,
                "authorization_published": not args.preflight_only,
                "supersession_sidecar": str(sidecar.relative_to(ROOT)),
                "superseded_v2_authorization_sha256": (
                    orchestrator._V2_AUTHORIZATION_SHA256
                ),
                "authorized_output_root": str(output_root.relative_to(ROOT)),
                "selection_order": list(_SELECTION_ORDER),
                "authorized_checkpoint_count": 12,
                "tests_passed": payload["test_receipt"]["expected_passed"],
                "training_started": False,
                "gate_opened": False,
                "i_gate_target_values_materialized": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
