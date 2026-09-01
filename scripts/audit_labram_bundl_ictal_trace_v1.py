#!/usr/bin/env python3
"""Fail-closed post-run trace audit for the LaBraM BUNDL-style candidate.

The training runner owns only an internal metric decision.  This independent
script is the sole authority allowed to emit the final source-native
qualification status, and only after binding the immutable result to a
complete successful-file-access trace with zero protected-data hits.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

AUDIT_SCHEMA = "soz_labram_bundl_source_only_trace_audit_v1"
RESULT_SCHEMA = "soz_labram_bundl_edge_time_source_only_oof_v1"
PROVISIONAL_PASS_STATUS = "provisional_internal_gates_passed_pending_trace_audit"
INTERNAL_FAIL_STATUS = "stop_source_native_probability_gate_failed"
FINAL_QUALIFIED_STATUS = "source_native_candidate_qualified"
BLOCKED_STATUS = "blocked_trace_or_result_integrity_failed"

RUNNER = ROOT / "scripts/run_labram_bundl_ictal_source_only_v1.py"
AUDITOR_RELATIVE_PATH = "scripts/audit_labram_bundl_ictal_trace_v1.py"
PROTOCOL = ROOT / "research/02_method/labram_next_single_candidate_protocol_20260812_zh.md"
PROTOCOL_SHA256 = "7560721b54bfd5c5c5247d53426c717391f8c766a1c677e3b3a77037e14bf537"
IDENTITY_FIREWALL = (
    ROOT / "outputs/labram_bundl_identity_firewall_v1_20260812/identity_firewall.json"
)
IDENTITY_FIREWALL_SHA256 = (
    "b1322d1bdae1608fe5040e2f1c40dadbfd04f6608d352be1421e694889b56942"
)
PUBLIC_UNION = ROOT / "outputs/public_development_union_v11_20260811/manifest.json"
PUBLIC_UNION_SHA256 = (
    "89a9ca456c724c2dee4d14a2c0da5a1190e58f97ad602060f6dda5f619b97232"
)
TARGET_SNAPSHOT = (
    ROOT / "outputs/tusz_ictal_prediction_artifacts_formal_v4_20260809/final/native"
)
TOKEN_CORPUS = ROOT / "outputs/tusz_ictal_token_corpus_formal_v4_20260809/master"
PRELAUNCH_LEDGER = (
    ROOT / "outputs/labram_bundl_ictal_source_only_v1_prelaunch_ledger_20260812.json"
)
PRELAUNCH_LEDGER_SCHEMA = "soz_labram_bundl_source_only_prelaunch_ledger_v1"

LEDGER_CODE_PATHS = {
    "runner": RUNNER,
    "bundl_ictal": ROOT / "src/soz/bundl_ictal.py",
    "concept_heads": ROOT / "src/soz/models/concept_heads.py",
    "concept_metrics": ROOT / "src/soz/concept_metrics.py",
    "cached_dataset": ROOT / "src/soz/cached_concept_training.py",
    "target_loader": ROOT / "src/soz/ictal_target_snapshot.py",
    "auditor": Path(__file__).resolve(),
}
LEDGER_DATA_SHA256 = {
    "master_bundle": "73e821d08805c3a7e8ae75011dd98fe10c388d7291c74881286438e91cacc35f",
    "master_source": "d5329b9231ecea7aaae6e126f5cd7a17a51f21b950025b32369592379acf8cb8",
    "token_corpus_index": "a7d672e3228cdc71fafb46e910033f6a5302a9e2e0a5f5716f7f4c8292ecfc26",
    "target_snapshot_manifest": "bc22681928e596ef6564af51f54215e96a9560a21cdeaedef043ccd324596cba",
    "target_snapshot_receipt": "e216338d5112a67d20fcba5d545834af2b84c8896a8713b9919866e839c7953a",
    "training_targets_file": "99bc6250dfadb407fe890a39ef9fa00743968d7d1c6ce3e710d917c555294722",
    "training_target_mask_file": "e4a372ef95dc85be57077389d83ca1fbdd18c99489af090fe2755e4b0cc5da60",
    "preprocessing_selection": "b4aa73bff2800f12186085976a5655db6882a38232d775d11234efa387171485",
    "preprocessing_protocol": "9a75dd2f3293d4d944380c0d82dcfca6a95e332f3b999e32e52b15d89622a196",
    "identity_firewall": IDENTITY_FIREWALL_SHA256,
    "public_union": PUBLIC_UNION_SHA256,
}

DEFAULT_OUTPUT = ROOT / "outputs/labram_bundl_ictal_source_only_v1_20260812"
DEFAULT_RESULT = DEFAULT_OUTPUT / "result.json"
DEFAULT_TRACE = ROOT / "outputs/labram_bundl_ictal_source_only_v1_20260812.strace.raw"
DEFAULT_RECEIPT = DEFAULT_OUTPUT / "post_run_trace_audit.json"
FORMAL_RUNNER_ARGV = (
    "--device",
    "cuda",
    "--output-directory",
    str(DEFAULT_OUTPUT),
)
FORMAL_STRACE_ARGV = (
    "strace",
    "-f",
    "-s",
    "4096",
    "-yy",
    "-e",
    "trace=open,openat,openat2,statx,readlink",
    "-o",
    str(DEFAULT_TRACE),
    "python3",
    str(RUNNER),
    *FORMAL_RUNNER_ARGV,
)

EXPECTED_GATE_KEYS = frozenset(
    {
        "bce_point_improves",
        "bce_ci_upper_nonpositive",
        "brier_point_improves",
        "brier_ci_upper_nonpositive",
        "auroc_noninferiority",
        "average_precision_noninferiority",
        "sensitivity_noninferiority",
        "false_positive_rate_noninferiority",
        "bce_and_brier_jointly_noninferior_at_least_four_of_five_folds",
        "identity_firewall",
        "missing_mask_preserved",
        "foundation_optimizer_parameter_count_zero",
    }
)
PRIVATE_RAW_ROOTS = (Path("/mnt/hd1/dyf/dataset/EEG"),)
CHINESE_PRIVATE_MARKERS = (
    "头皮扩散",
    "私有数据",
    "显著通道",
    "扩散通道",
    "颞叶癫痫",
)
SOURCE_SUFFIXES = frozenset({".py", ".pyc", ".pyi", ".so", ".pyd"})

_CALL_RE = re.compile(r"\b(open|openat|openat2|statx|readlink)\(")
_RESUMED_RE = re.compile(r"<\.\.\.\s+(open|openat|openat2|statx|readlink)\s+resumed>")
_QUOTED_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_RESULT_RE = re.compile(r"=\s*(-?\d+)(?:<[^>]*>)?(?:\s+.*)?$")
_PID_RE = re.compile(r"^\s*(?:\[pid\s+)?(\d+)\]?\s+")
_EXIT_ZERO_RE = re.compile(r"\+\+\+ exited with 0 \+\+\+\s*$")
_DIRFD_PATH_RE = re.compile(r"(?:AT_FDCWD|-?\d+)<((?:\\.|[^>])*)>")


@dataclass(frozen=True)
class FileAccess:
    line_number: int
    pid: str
    syscall: str
    raw_path: str
    normalized_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _expected_formal_launch() -> dict[str, object]:
    return {
        "working_directory": str(ROOT),
        "python_executable": str(Path(sys.executable).resolve()),
        "runner_argv": list(FORMAL_RUNNER_ARGV),
        "strace_argv": list(FORMAL_STRACE_ARGV),
        "trace_path": str(DEFAULT_TRACE),
        "output_directory": str(DEFAULT_OUTPUT),
        "result_path": str(DEFAULT_RESULT),
        "audit_receipt_path": str(DEFAULT_RECEIPT),
    }


def expected_prelaunch_ledger_payload() -> dict[str, object]:
    """Return the single closed-schema ledger payload to seal pre-launch."""

    return {
        "schema_version": PRELAUNCH_LEDGER_SCHEMA,
        "code_sha256": {
            name: _sha256(path) for name, path in LEDGER_CODE_PATHS.items()
        },
        "protocol_sha256": PROTOCOL_SHA256,
        "data_sha256": dict(LEDGER_DATA_SHA256),
        "formal_launch": _expected_formal_launch(),
    }


def _split_pid(line: str) -> tuple[str, str]:
    match = _PID_RE.match(line)
    if match is None:
        return "main", line
    return match.group(1), line[match.end() :]


def _decode_c_string(token: str) -> str:
    try:
        decoded = ast.literal_eval(token)
    except (SyntaxError, ValueError):
        return token[1:-1]
    if not isinstance(decoded, str):
        return token[1:-1]
    # strace renders non-ASCII pathname bytes as octal escapes.  ast decodes
    # those escapes to Latin-1 code points; this round-trip recovers UTF-8.
    try:
        return decoded.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return decoded


def _first_quoted_path(fragment: str) -> str | None:
    match = _QUOTED_RE.search(fragment)
    return None if match is None else _decode_c_string(match.group(0))


def _successful_result(fragment: str) -> bool:
    match = _RESULT_RE.search(fragment)
    return match is not None and int(match.group(1)) >= 0


def _normalize_path(raw_path: str, call_fragment: str = "") -> str:
    value = raw_path.removesuffix(" (deleted)")
    path = Path(value)
    if not path.is_absolute():
        base = ROOT
        quoted = _QUOTED_RE.search(call_fragment)
        prefix = call_fragment if quoted is None else call_fragment[: quoted.start()]
        dirfd_paths = _DIRFD_PATH_RE.findall(prefix)
        if dirfd_paths:
            decoded_base = _decode_c_string(f'"{dirfd_paths[-1]}"')
            if Path(decoded_base).is_absolute():
                base = Path(decoded_base)
        path = base / path
    return os.path.normpath(str(path))


def _parse_successful_accesses(
    text: str,
) -> tuple[list[FileAccess], int, set[str]]:
    accesses: list[FileAccess] = []
    pending: dict[tuple[str, str], tuple[int, str, str]] = {}
    exit_zero_pids: set[str] = set()

    for line_number, line in enumerate(text.splitlines(), start=1):
        pid, body = _split_pid(line)
        if _EXIT_ZERO_RE.search(body):
            exit_zero_pids.add(pid)

        resumed = _RESUMED_RE.search(body)
        if resumed is not None:
            syscall = resumed.group(1)
            pending_row = pending.pop((pid, syscall), None)
            if pending_row is not None and _successful_result(body):
                start_line, raw_path, start_body = pending_row
                accesses.append(
                    FileAccess(
                        line_number=start_line,
                        pid=pid,
                        syscall=syscall,
                        raw_path=raw_path,
                        normalized_path=_normalize_path(raw_path, start_body),
                    )
                )
            continue

        call = _CALL_RE.search(body)
        if call is None:
            continue
        syscall = call.group(1)
        raw_path = _first_quoted_path(body[call.end() :])
        if raw_path is None:
            continue
        if "<unfinished ...>" in body:
            pending[(pid, syscall)] = (line_number, raw_path, body)
            continue
        if _successful_result(body):
            accesses.append(
                FileAccess(
                    line_number=line_number,
                    pid=pid,
                    syscall=syscall,
                    raw_path=raw_path,
                    normalized_path=_normalize_path(raw_path, body),
                )
            )
    return accesses, len(pending), exit_zero_pids


def _is_within(path: str, root: Path) -> bool:
    try:
        return os.path.commonpath((path, os.path.normpath(str(root)))) == os.path.normpath(
            str(root)
        )
    except ValueError:
        return False


def _is_allowed_repo_source(path: str) -> bool:
    if not _is_within(path, ROOT):
        return False
    relative = Path(os.path.relpath(path, ROOT))
    if not relative.parts or relative.parts[0] not in {"src", "scripts"}:
        return False
    return relative.suffix.casefold() in SOURCE_SUFFIXES


def _deny_rule(path: str) -> str | None:
    normalized = os.path.normpath(path)
    lower = normalized.casefold()
    name = Path(normalized).name.casefold()

    # Normal imports such as src/soz/data/deepsoz.py and their pyc files are
    # code, not label access.  The identity-only sidecar is an explicit input.
    if _is_allowed_repo_source(normalized):
        return None
    if normalized == os.path.normpath(str(IDENTITY_FIREWALL)):
        return None

    if name == "apikey.txt":
        return "credential_file"
    if lower.endswith((".xls", ".xlsx")):
        return "spreadsheet_annotation"
    compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", lower)
    if "eegfmri" in compact:
        return "named_private_source"
    if any(marker in normalized for marker in CHINESE_PRIVATE_MARKERS):
        return "named_private_source"
    if any(_is_within(normalized, root) for root in PRIVATE_RAW_ROOTS):
        return "private_raw_eeg_root"

    parts = tuple(part.casefold() for part in Path(normalized).parts)
    has_data_anchor = any(
        part in {"outputs", "output", "data", "dataset", "datasets"}
        or part.startswith("dataset_")
        for part in parts
    )
    if has_data_anchor and any("deepsoz" in part for part in parts):
        return "deepsoz_data_or_target"
    if (has_data_anchor or _is_within(normalized, ROOT)) and any(
        "private" in part for part in parts
    ):
        return "private_data_or_artifact"
    return None


def _safe_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _load_result(path: Path) -> tuple[Mapping[str, object], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {}, type(error).__name__
    if not isinstance(payload, Mapping):
        return {}, "result_not_object"
    return payload, None


def _path_was_accessed(accesses: Sequence[FileAccess], path: Path) -> bool:
    expected = os.path.normpath(str(path))
    return any(row.normalized_path == expected for row in accesses)


def _token_payload_was_accessed(accesses: Sequence[FileAccess]) -> bool:
    return any(
        _is_within(row.normalized_path, TOKEN_CORPUS / "events")
        and Path(row.normalized_path).name == "concept_tokens.safetensors"
        for row in accesses
    )


def audit(
    result_path: Path,
    trace_path: Path,
    ledger_path: Path = PRELAUNCH_LEDGER,
) -> dict[str, object]:
    result_path = Path(os.path.abspath(result_path))
    trace_path = Path(os.path.abspath(trace_path))
    ledger_path = Path(os.path.abspath(ledger_path))
    result, result_load_error = _load_result(result_path)
    ledger, ledger_load_error = _load_result(ledger_path)
    ledger_sha = _sha256(ledger_path) if ledger_path.is_file() else None
    expected_ledger = expected_prelaunch_ledger_payload()

    ledger_checks = {
        "ledger_loaded_as_object": ledger_load_error is None,
        "ledger_regular_not_symlink": (
            ledger_path.is_file() and not ledger_path.is_symlink()
        ),
        "ledger_closed_schema": set(ledger) == set(expected_ledger),
        "ledger_schema": ledger.get("schema_version") == PRELAUNCH_LEDGER_SCHEMA,
        "ledger_code_hashes_current": ledger.get("code_sha256")
        == expected_ledger["code_sha256"],
        "ledger_protocol_hash": ledger.get("protocol_sha256") == PROTOCOL_SHA256,
        "ledger_data_hash_contract": ledger.get("data_sha256")
        == LEDGER_DATA_SHA256,
        "ledger_formal_launch_contract": ledger.get("formal_launch")
        == _expected_formal_launch(),
        "training_targets_file_current": (
            _sha256(TARGET_SNAPSHOT / "training_targets.npy")
            == LEDGER_DATA_SHA256["training_targets_file"]
        ),
        "training_target_mask_file_current": (
            _sha256(TARGET_SNAPSHOT / "training_target_mask.npy")
            == LEDGER_DATA_SHA256["training_target_mask_file"]
        ),
    }

    result_checks: dict[str, bool] = {}

    def result_check(name: str, condition: object) -> None:
        result_checks[name] = condition is True

    result_check("result_loaded_as_object", result_load_error is None)
    result_check(
        "result_regular_not_symlink",
        result_path.is_file() and not result_path.is_symlink(),
    )
    result_check("result_schema", result.get("schema_version") == RESULT_SCHEMA)
    result_check("protocol_sha", result.get("protocol_sha256") == PROTOCOL_SHA256)
    result_check(
        "result_ledger_hash",
        ledger_sha is not None and result.get("prelaunch_ledger_sha256") == ledger_sha,
    )
    result_check("final_qualification_false", result.get("final_qualification") is False)
    result_check("trace_audit_required", result.get("trace_audit_required") is True)
    result_check("trace_audit_not_completed", result.get("trace_audit_completed") is False)
    result_check(
        "final_authority_is_this_auditor",
        result.get("final_qualification_authority") == AUDITOR_RELATIVE_PATH,
    )

    gates = _safe_mapping(result.get("gates"))
    gates_are_boolean = bool(gates) and all(type(value) is bool for value in gates.values())
    result_check("gate_schema_exact", set(gates) == EXPECTED_GATE_KEYS)
    result_check("gate_values_boolean", gates_are_boolean)
    recomputed_internal_pass = (
        gates_are_boolean
        and set(gates) == EXPECTED_GATE_KEYS
        and all(value is True for value in gates.values())
    )
    result_check(
        "all_gates_recomputes",
        type(result.get("all_gates_passed")) is bool
        and result.get("all_gates_passed") is recomputed_internal_pass,
    )
    expected_internal_status = (
        PROVISIONAL_PASS_STATUS if recomputed_internal_pass else INTERNAL_FAIL_STATUS
    )
    result_check("internal_status_consistent", result.get("status") == expected_internal_status)

    for field in (
        "deepsoz_target_values_loaded",
        "deepsoz_target_values_used",
        "private_inputs_loaded",
        "private_inputs_used",
        "historical_i_dev_or_gate_outcomes_loaded",
        "historical_i_dev_or_gate_outcomes_used",
        "formal_soz_promotion",
        "checkpoint_authorized_for_soz_reasoner",
    ):
        result_check(f"result_{field}_false", result.get(field) is False)

    preflight = _safe_mapping(result.get("preflight"))
    source_hashes = _safe_mapping(preflight.get("source_file_sha256"))
    result_check(
        "all_code_hashes_match_result_and_ledger",
        source_hashes == _safe_mapping(ledger.get("code_sha256"))
        and source_hashes == expected_ledger["code_sha256"],
    )
    result_check(
        "preflight_ledger_hash",
        ledger_sha is not None and preflight.get("prelaunch_ledger_sha256") == ledger_sha,
    )
    result_check(
        "preflight_identity_sidecar_hash",
        preflight.get("identity_firewall_sidecar_sha256") == IDENTITY_FIREWALL_SHA256,
    )
    result_check(
        "preflight_public_union_hash",
        preflight.get("public_union_sha256") == PUBLIC_UNION_SHA256,
    )
    result_check(
        "preflight_protected_values_false",
        all(
            preflight.get(field) is False
            for field in (
                "deepsoz_target_values_loaded",
                "deepsoz_target_values_used",
                "private_inputs_loaded",
                "private_inputs_used",
                "historical_i_dev_or_gate_outcomes_loaded",
                "historical_i_dev_or_gate_outcomes_used",
            )
        ),
    )
    execution = _safe_mapping(result.get("execution_receipt"))
    result_check(
        "execution_python_matches_ledger",
        execution.get("python_executable")
        == _safe_mapping(ledger.get("formal_launch")).get("python_executable"),
    )

    artifact_paths: dict[str, Path] = {}
    for key, expected_filename in (
        ("prediction_file", "oof_predictions.safetensors"),
        ("state_file", "outer_fold_states.safetensors"),
    ):
        metadata = _safe_mapping(result.get(key))
        filename = metadata.get("filename")
        artifact = result_path.parent / expected_filename
        artifact_paths[key] = artifact
        result_check(f"{key}_filename", filename == expected_filename)
        result_check(f"{key}_exists", artifact.is_file())
        result_check(
            f"{key}_hash",
            artifact.is_file()
            and isinstance(metadata.get("sha256"), str)
            and _sha256(artifact) == metadata.get("sha256"),
        )

    trace_bytes = b""
    trace_read_error: str | None = None
    try:
        trace_bytes = trace_path.read_bytes()
    except OSError as error:
        trace_read_error = type(error).__name__
    trace_text = trace_bytes.decode("utf-8", errors="replace")
    accesses, unfinished_count, exit_zero_pids = _parse_successful_accesses(trace_text)

    required_evidence = {
        "runner": _path_was_accessed(accesses, RUNNER),
        "prelaunch_ledger": _path_was_accessed(accesses, ledger_path),
        "identity_sidecar": _path_was_accessed(accesses, IDENTITY_FIREWALL),
        "public_union": _path_was_accessed(accesses, PUBLIC_UNION),
        "target_snapshot_manifest": _path_was_accessed(
            accesses, TARGET_SNAPSHOT / "manifest.json"
        ),
        "target_snapshot_receipt": _path_was_accessed(
            accesses, TARGET_SNAPSHOT / "receipt.json"
        ),
        "training_targets": _path_was_accessed(
            accesses, TARGET_SNAPSHOT / "training_targets.npy"
        ),
        "training_target_mask": _path_was_accessed(
            accesses, TARGET_SNAPSHOT / "training_target_mask.npy"
        ),
        "token_corpus_index": _path_was_accessed(accesses, TOKEN_CORPUS / "index.json"),
        "token_corpus_payload": _token_payload_was_accessed(accesses),
        "result_file": _path_was_accessed(accesses, result_path),
        "prediction_file": _path_was_accessed(accesses, artifact_paths["prediction_file"]),
        "state_file": _path_was_accessed(accesses, artifact_paths["state_file"]),
    }
    runner_pids = {
        row.pid
        for row in accesses
        if row.normalized_path == os.path.normpath(str(RUNNER))
    }

    denied_hits: list[dict[str, object]] = []
    for row in accesses:
        rule = _deny_rule(row.normalized_path)
        if rule is None:
            continue
        denied_hits.append(
            {
                "rule": rule,
                "path_sha256": hashlib.sha256(
                    row.normalized_path.encode("utf-8", errors="surrogatepass")
                ).hexdigest(),
                "line_number": row.line_number,
                "syscall": row.syscall,
            }
        )

    trace_checks = {
        "trace_readable": trace_read_error is None,
        "trace_regular_not_symlink": trace_path.is_file() and not trace_path.is_symlink(),
        "trace_nonempty": len(trace_bytes) > 0,
        "trace_contains_no_nul": b"\x00" not in trace_bytes,
        "successful_file_accesses_present": len(accesses) > 0,
        "no_unresolved_file_syscalls": unfinished_count == 0,
        "runner_process_exited_zero": bool(runner_pids & exit_zero_pids),
        "all_required_access_evidence_present": all(required_evidence.values()),
        "protected_data_denylist_zero_hit": len(denied_hits) == 0,
    }

    result_integrity_passed = all(result_checks.values()) and all(ledger_checks.values())
    trace_audit_passed = all(trace_checks.values())
    if not result_integrity_passed or not trace_audit_passed:
        final_status = BLOCKED_STATUS
        final_qualification = False
    elif recomputed_internal_pass:
        final_status = FINAL_QUALIFIED_STATUS
        final_qualification = True
    else:
        final_status = INTERNAL_FAIL_STATUS
        final_qualification = False

    result_sha = _sha256(result_path) if result_path.is_file() else None
    trace_sha = hashlib.sha256(trace_bytes).hexdigest() if trace_read_error is None else None
    receipt = {
        "schema_version": AUDIT_SCHEMA,
        "status": final_status,
        "final_qualification": final_qualification,
        "internal_status": result.get("status"),
        "internal_gates_passed": recomputed_internal_pass,
        "result_integrity_passed": result_integrity_passed,
        "trace_audit_passed": trace_audit_passed,
        "ledger_checks": ledger_checks,
        "result_checks": result_checks,
        "trace_checks": trace_checks,
        "required_access_evidence": required_evidence,
        "successful_file_access_count": len(accesses),
        "unfinished_file_syscall_count": unfinished_count,
        "denylist_hit_count": len(denied_hits),
        "denylist_hits": denied_hits,
        "sealed_inputs": {
            "prelaunch_ledger_path": str(ledger_path),
            "prelaunch_ledger_sha256": ledger_sha,
            "prelaunch_ledger_size_bytes": (
                ledger_path.stat().st_size if ledger_path.is_file() else None
            ),
            "result_path": str(result_path),
            "result_sha256": result_sha,
            "result_size_bytes": result_path.stat().st_size if result_path.is_file() else None,
            "trace_path": str(trace_path),
            "trace_sha256": trace_sha,
            "trace_size_bytes": len(trace_bytes) if trace_read_error is None else None,
        },
        "pinned_inputs": {
            "prelaunch_ledger_sha256": ledger_sha,
            "protocol_sha256": _sha256(PROTOCOL) if PROTOCOL.is_file() else None,
            "identity_sidecar_sha256": (
                _sha256(IDENTITY_FIREWALL) if IDENTITY_FIREWALL.is_file() else None
            ),
            "public_union_sha256": _sha256(PUBLIC_UNION) if PUBLIC_UNION.is_file() else None,
        },
        "source_file_sha256": {
            name: _sha256(path) for name, path in LEDGER_CODE_PATHS.items()
        },
        "result_load_error": result_load_error,
        "ledger_load_error": ledger_load_error,
        "trace_read_error": trace_read_error,
        "audited_syscalls": ["open", "openat", "openat2", "statx", "readlink"],
        "formal_soz_promotion": False,
        "checkpoint_authorized_for_soz_reasoner": False,
        "claim_boundary": (
            "A pass qualifies only the source-native TUSZ bipolar edge-time "
            "visible-involvement producer under the declared syscall-set audit; "
            "it is not evidence of SOZ improvement or OS-level non-reachability."
        ),
    }

    # Pin checks are audit gates too; calculate them after recording observed
    # digests so a changed public artifact cannot silently qualify.
    pinned_ok = (
        ledger_sha is not None
        and result.get("prelaunch_ledger_sha256") == ledger_sha
        and receipt["pinned_inputs"]["protocol_sha256"] == PROTOCOL_SHA256
        and receipt["pinned_inputs"]["identity_sidecar_sha256"]
        == IDENTITY_FIREWALL_SHA256
        and receipt["pinned_inputs"]["public_union_sha256"] == PUBLIC_UNION_SHA256
    )
    receipt["pinned_inputs_match"] = pinned_ok
    if not pinned_ok:
        receipt["status"] = BLOCKED_STATUS
        receipt["final_qualification"] = False
        receipt["result_integrity_passed"] = False
    return receipt


def _validate_formal_paths(result: Path, trace: Path, receipt: Path) -> None:
    observed = tuple(Path(os.path.abspath(path)) for path in (result, trace, receipt))
    expected = tuple(
        Path(os.path.abspath(path))
        for path in (DEFAULT_RESULT, DEFAULT_TRACE, DEFAULT_RECEIPT)
    )
    if observed != expected:
        raise ValueError("Formal trace audit requires the frozen default paths")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_formal_paths(args.result, args.trace, args.receipt)
    receipt_path = Path(os.path.abspath(args.receipt))
    if os.path.lexists(receipt_path):
        raise FileExistsError(f"Audit receipt already exists: {receipt_path}")
    if not receipt_path.parent.is_dir():
        raise FileNotFoundError("Formal training output directory does not exist")
    receipt = audit(args.result, args.trace)
    with receipt_path.open("xb") as handle:
        handle.write(_canonical_json(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "final_qualification": receipt["final_qualification"],
                "trace_audit_passed": receipt["trace_audit_passed"],
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if receipt["final_qualification"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
