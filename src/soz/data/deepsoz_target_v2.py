"""Versioned, stand-alone DeepSOZ patient-target artifact.

This module materializes the benchmark-complement v2 target policy without
modifying the conservative v1 DeepSOZ--TUSZ split package.  The artifact is
patient-level and contains no EEG paths, event rows, or private-cohort labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from ..geometry import STANDARD_19
from .deepsoz import (
    BINARY_STATE_EXPLICIT_0,
    BINARY_STATE_EXPLICIT_1,
    BINARY_STATE_MISSING,
    BINARY_STATE_PATIENT_VARIABLE,
    EXPECTED_CONCEPT_OOF_FOLDS,
    PZ_PRIMARY_STATE,
    DeepSOZReferenceRegistry,
    build_deepsoz_reference_registry,
    normalize_patient_id,
    registry_to_frame,
)


TARGET_V2_POLICY_VERSION = "deepsoz-benchmark-target-v2.0.0"
TARGET_V2_SCHEMA_VERSION = "deepsoz-patient-target-artifact-v2.0.0"
VERIFIED_TARGET_V2_RECEIPT_SCHEMA = "soz_verified_deepsoz_target_v2_receipt_v1"
TARGETS_FILENAME = "patient_targets_v2.csv"
SUMMARY_FILENAME = "summary.json"
README_FILENAME = "README.md"
V1_PACKAGE_DIRNAME = "deepsoz_tusz_patient_splits_v1"
_V1_MARKERS = frozenset(
    {
        "record_crosswalk.csv",
        "patient_targets.csv",
        "event_inputs.csv",
        "split_manifest.csv",
    }
)
_V2_OUTPUT_FILES = (TARGETS_FILENAME, SUMMARY_FILENAME, README_FILENAME)
_V2_OUTPUT_FILE_SET = frozenset(_V2_OUTPUT_FILES)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_TARGET_BYTES = 64 * 1024 * 1024
_MAX_SUMMARY_BYTES = 4 * 1024 * 1024
_MAX_README_BYTES = 1024 * 1024
_MAX_INPUT_CSV_BYTES = 256 * 1024 * 1024
_ELIGIBLE_SPLITS = ("source_train", "source_dev", "source_eval")
_STATE_VOCABULARY = (
    BINARY_STATE_EXPLICIT_1,
    BINARY_STATE_EXPLICIT_0,
    BINARY_STATE_MISSING,
    BINARY_STATE_PATIENT_VARIABLE,
    PZ_PRIMARY_STATE,
)


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Target-v2 provenance is not canonical JSON data") from exc


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return text


def _target_policy() -> dict[str, Any]:
    return {
        "explicit_1": "reference_positive",
        "explicit_0": "dataset_complement_negative_not_biological_negative",
        "missing": "unknown_loss_mask_zero",
        "patient_variable_in_head": "masked_and_patient_ineligible",
        "pz_primary": PZ_PRIMARY_STATE,
        "outside_head": "audit_only_never_projected_into_standard19",
        "target_granularity": "patient",
        "concept_oof_n_folds": EXPECTED_CONCEPT_OOF_FOLDS,
        "private_labels_included": False,
    }


TARGET_V2_POLICY_SHA256 = _canonical_sha256(
    {
        "schema_version": TARGET_V2_SCHEMA_VERSION,
        "policy_version": TARGET_V2_POLICY_VERSION,
        "standard_19": STANDARD_19,
        "policy": _target_policy(),
        "state_vocabulary": _STATE_VOCABULARY,
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _guard_output_directory(output_dir: Path, split_csv: Path) -> None:
    resolved_output = _resolved(output_dir)
    resolved_split_parent = _resolved(split_csv).parent
    if (
        resolved_output.name == V1_PACKAGE_DIRNAME
        or resolved_output == resolved_split_parent
    ):
        raise ValueError(
            "Refusing to write target-v2 into the frozen v1 split package: "
            f"{resolved_output}"
        )
    if resolved_output.is_dir():
        legacy_files = sorted(
            name for name in _V1_MARKERS if (resolved_output / name).exists()
        )
        if legacy_files:
            raise ValueError(
                "Refusing to mix target-v2 with a v1-style package containing "
                f"{legacy_files}"
            )


def _counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _build_target_frame(
    registry: DeepSOZReferenceRegistry,
    *,
    source_sha256: str,
    split_sha256: str,
) -> pd.DataFrame:
    frame = registry_to_frame(registry)
    frame.insert(0, "policy_version", TARGET_V2_POLICY_VERSION)
    frame.insert(1, "schema_version", TARGET_V2_SCHEMA_VERSION)
    frame.insert(2, "source_input_sha256", source_sha256)
    frame.insert(3, "split_input_sha256", split_sha256)
    return frame


def _target_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Return the frozen writer representation used for exact rebuild checks."""

    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _build_summary(
    *,
    registry: DeepSOZReferenceRegistry,
    frame: pd.DataFrame,
    source: pd.DataFrame,
    split: pd.DataFrame,
    source_csv: Path,
    split_csv: Path,
    source_sha256: str,
    split_sha256: str,
) -> dict[str, Any]:
    eligible = [reference for reference in registry if reference.eligible_for_localization]
    return {
        "schema_version": TARGET_V2_SCHEMA_VERSION,
        "policy_version": TARGET_V2_POLICY_VERSION,
        "generated_by": "src.soz.data.deepsoz_target_v2",
        "inputs": {
            "deepsoz_source": {
                "path": str(_resolved(source_csv)),
                "sha256": source_sha256,
                "rows": int(len(source)),
                "patients": int(source["pt_id"].map(normalize_patient_id).nunique()),
            },
            "split_manifest": {
                "path": str(_resolved(split_csv)),
                "sha256": split_sha256,
                "rows": int(len(split)),
                "patients": int(
                    split["deepsoz_patient_id"].map(normalize_patient_id).nunique()
                ),
            },
        },
        "policy": _target_policy(),
        "counts": {
            "patients_total": len(registry),
            "patients_eligible": len(eligible),
            "patients_ineligible": len(registry) - len(eligible),
            "all_model_splits": _counts(
                [reference.model_split for reference in registry]
            ),
            "eligible_model_splits": _counts(
                [reference.model_split for reference in eligible]
            ),
            "exclusion_reasons": _counts(
                [
                    reference.exclusion_reason
                    for reference in registry
                    if not reference.eligible_for_localization
                ]
            ),
        },
        "target_state_counts": {
            channel: _counts(frame[f"benchmark_state_{channel}"].astype(str).tolist())
            for channel in STANDARD_19
        },
        "pz_raw_state_counts": {
            view: _counts(frame[f"pz_{view}_state"].astype(str).tolist())
            for view in ("first", "second", "or")
        },
        "outside_head_state_counts": {
            name: _counts(frame[f"outside_{name}_state"].astype(str).tolist())
            for name in ("OZ", "A1", "A2")
        },
        "state_vocabulary": list(_STATE_VOCABULARY),
    }


def _render_readme(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    return f"""# DeepSOZ patient target artifact v2

Policy: `{summary['policy_version']}`  
Schema: `{summary['schema_version']}`

This is a patient-level target overlay. It is independent of, and does not
modify, `outputs/{V1_PACKAGE_DIRNAME}`.

- explicit `1`: benchmark reference positive;
- explicit `0`: dataset-complement negative, not a clinically confirmed
  biological negative;
- missing: unknown with loss mask zero;
- patient-variable in-head field: masked and ineligible;
- PZ: primary mask zero, with raw `pz`, `pz.1`, and row-wise OR audits retained;
- OZ/A1/A2: outside-head audit only, never projected into the standard-19 head;
- private labels: absent by construction.

Patients: {counts['patients_total']} total, {counts['patients_eligible']} eligible,
{counts['patients_ineligible']} ineligible.

Input SHA256 values and all frozen policy semantics are stored in
`{SUMMARY_FILENAME}` and repeated in `{TARGETS_FILENAME}`.
"""


def _reject_symlink_components(path: Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot contain symlink components")
    return absolute


def _read_stable_regular_file(
    path: str | Path,
    *,
    field: str,
    max_bytes: int,
) -> tuple[bytes, str]:
    source = _reject_symlink_components(Path(path), field=field)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"{field} must be a regular file")
    before = source.stat()
    if before.st_size < 1 or before.st_size > max_bytes:
        raise ValueError(f"{field} has an invalid size")
    payload = source.read_bytes()
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{field} changed while it was read")
    return payload, hashlib.sha256(payload).hexdigest()


def _check_expected_sha(actual: str, expected: object, *, field: str) -> None:
    if actual != _require_sha256(expected, field=field):
        raise ValueError(f"{field} does not match the current exact bytes")


def _parse_registry_csv(
    payload: bytes,
    *,
    field: str,
    allow_legacy_pz_pair: bool = False,
) -> pd.DataFrame:
    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} must be strict UTF-8 CSV") from exc
    try:
        header = next(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (csv.Error, StopIteration) as exc:
        raise ValueError(f"{field} has no valid CSV header") from exc
    if not header or any(not name.strip() for name in header):
        raise ValueError(f"{field} contains an empty CSV column name")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    allowed_pz_pair = (
        allow_legacy_pz_pair
        and duplicates == ["pz"]
        and header.count("pz") == 2
    )
    if duplicates and not allowed_pz_pair:
        raise ValueError(f"{field} contains duplicate CSV columns: {duplicates}")
    try:
        frame = pd.read_csv(io.StringIO(text, newline=""))
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{field} is not a valid CSV table") from exc
    if frame.columns.duplicated().any():
        raise ValueError(f"{field} contains duplicate parsed columns")
    if allowed_pz_pair and not {"pz", "pz.1"} <= set(frame.columns):
        raise ValueError(
            f"{field} did not preserve the two legacy PZ columns as pz/pz.1"
        )
    return frame


def _validate_target_csv_header(
    payload: bytes,
    *,
    expected_columns: Sequence[object],
) -> None:
    try:
        text = payload.decode("utf-8", errors="strict")
        header = next(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise ValueError("Target-v2 CSV has no strict UTF-8 header") from exc
    if len(header) != len(set(header)):
        duplicates = sorted({name for name in header if header.count(name) > 1})
        raise ValueError(f"Target-v2 CSV contains duplicate columns: {duplicates}")
    expected = tuple(str(column) for column in expected_columns)
    actual = tuple(header)
    if actual != expected:
        raise ValueError(
            "Target-v2 CSV violates its closed column schema; "
            f"missing={sorted(set(expected)-set(actual))}, "
            f"unknown={sorted(set(actual)-set(expected))}"
        )


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _parse_summary_json(payload: bytes) -> dict[str, object]:
    try:
        decoded = payload.decode("utf-8", errors="strict")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Target-v2 summary must be strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Target-v2 summary must be a JSON object")
    return parsed


def _summary_for_comparison(
    summary: Mapping[str, object],
) -> dict[str, object]:
    """Copy a summary while neutralizing untrusted legacy path strings."""

    normalized = copy.deepcopy(dict(summary))
    try:
        inputs = normalized["inputs"]
        if not isinstance(inputs, dict):
            raise TypeError
        for key in ("deepsoz_source", "split_manifest"):
            item = inputs[key]
            if not isinstance(item, dict):
                raise TypeError
            path = item["path"]
            if not isinstance(path, str) or not path.strip():
                raise TypeError
            item["path"] = "<untrusted_legacy_path_not_provenance>"
    except (KeyError, TypeError) as exc:
        raise ValueError("Target-v2 summary has an invalid input path field") from exc
    return normalized


def _patient_roster_sha256(patient_ids: Sequence[object]) -> str:
    normalized = tuple(sorted(normalize_patient_id(value) for value in patient_ids))
    if len(set(normalized)) != len(normalized):
        raise ValueError("Target-v2 patient roster cannot contain duplicates")
    return _canonical_sha256(normalized)


@dataclass(frozen=True)
class DeepSOZTargetV2VerifiedReceipt:
    """Path-free binding used when joining verified targets to evidence."""

    target_artifact_sha256: str
    summary_artifact_sha256: str
    readme_artifact_sha256: str
    source_input_sha256: str
    split_input_sha256: str
    policy_sha256: str
    patient_ids: tuple[str, ...]
    eligible_patient_ids: tuple[str, ...]
    eligible_split_patient_ids: tuple[tuple[str, tuple[str, ...]], ...]
    patient_roster_sha256: str
    eligible_patient_roster_sha256: str
    patient_count: int
    eligible_patient_count: int
    policy_version: str = TARGET_V2_POLICY_VERSION
    target_schema_version: str = TARGET_V2_SCHEMA_VERSION
    schema_version: str = VERIFIED_TARGET_V2_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "target_artifact_sha256",
            "summary_artifact_sha256",
            "readme_artifact_sha256",
            "source_input_sha256",
            "split_input_sha256",
            "policy_sha256",
            "patient_roster_sha256",
            "eligible_patient_roster_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.policy_sha256 != TARGET_V2_POLICY_SHA256:
            raise ValueError("Target-v2 receipt policy SHA is not the frozen policy")
        if self.policy_version != TARGET_V2_POLICY_VERSION:
            raise ValueError("Target-v2 receipt policy version is not frozen")
        if self.target_schema_version != TARGET_V2_SCHEMA_VERSION:
            raise ValueError("Target-v2 receipt artifact schema is not frozen")
        if self.schema_version != VERIFIED_TARGET_V2_RECEIPT_SCHEMA:
            raise ValueError("Unsupported verified target-v2 receipt schema")

        patient_ids = tuple(normalize_patient_id(value) for value in self.patient_ids)
        eligible_ids = tuple(
            normalize_patient_id(value) for value in self.eligible_patient_ids
        )
        if (
            not patient_ids
            or tuple(sorted(patient_ids)) != patient_ids
            or len(set(patient_ids)) != len(patient_ids)
        ):
            raise ValueError("Target-v2 receipt patient roster must be sorted and unique")
        if (
            tuple(sorted(eligible_ids)) != eligible_ids
            or len(set(eligible_ids)) != len(eligible_ids)
            or not set(eligible_ids) <= set(patient_ids)
        ):
            raise ValueError("Target-v2 eligible roster is invalid")
        object.__setattr__(self, "patient_ids", patient_ids)
        object.__setattr__(self, "eligible_patient_ids", eligible_ids)
        if self.patient_count != len(patient_ids):
            raise ValueError("Target-v2 patient_count disagrees with its roster")
        if self.eligible_patient_count != len(eligible_ids):
            raise ValueError("Target-v2 eligible_patient_count disagrees with its roster")
        if self.patient_roster_sha256 != _patient_roster_sha256(patient_ids):
            raise ValueError("Target-v2 patient roster SHA mismatch")
        if self.eligible_patient_roster_sha256 != _patient_roster_sha256(
            eligible_ids
        ):
            raise ValueError("Target-v2 eligible patient roster SHA mismatch")

        split_keys = tuple(key for key, _ in self.eligible_split_patient_ids)
        if split_keys != _ELIGIBLE_SPLITS:
            raise ValueError("Target-v2 eligible split rosters use the wrong order")
        flattened: list[str] = []
        normalized_splits: list[tuple[str, tuple[str, ...]]] = []
        for key, values in self.eligible_split_patient_ids:
            roster = tuple(normalize_patient_id(value) for value in values)
            if tuple(sorted(roster)) != roster or len(set(roster)) != len(roster):
                raise ValueError(f"Target-v2 {key} roster must be sorted and unique")
            flattened.extend(roster)
            normalized_splits.append((key, roster))
        if tuple(sorted(flattened)) != eligible_ids:
            raise ValueError("Target-v2 eligible split rosters are not an exact partition")
        object.__setattr__(
            self, "eligible_split_patient_ids", tuple(normalized_splits)
        )

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class VerifiedDeepSOZTargetV2Artifact:
    """Strictly rebuilt target registry and its path-free provenance receipt."""

    registry: DeepSOZReferenceRegistry
    receipt: DeepSOZTargetV2VerifiedReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.registry, DeepSOZReferenceRegistry):
            raise TypeError("Verified target-v2 registry has the wrong type")
        if not isinstance(self.receipt, DeepSOZTargetV2VerifiedReceipt):
            raise TypeError("Verified target-v2 receipt has the wrong type")
        patient_ids = tuple(reference.patient_id for reference in self.registry)
        eligible_ids = tuple(
            reference.patient_id
            for reference in self.registry
            if reference.eligible_for_localization
        )
        split_rosters = tuple(
            (
                split,
                tuple(
                    reference.patient_id
                    for reference in self.registry
                    if reference.eligible_for_localization
                    and reference.model_split == split
                ),
            )
            for split in _ELIGIBLE_SPLITS
        )
        if patient_ids != self.receipt.patient_ids:
            raise ValueError("Verified target-v2 registry patient roster mismatch")
        if eligible_ids != self.receipt.eligible_patient_ids:
            raise ValueError("Verified target-v2 registry eligible roster mismatch")
        if split_rosters != self.receipt.eligible_split_patient_ids:
            raise ValueError("Verified target-v2 registry split roster mismatch")


def load_verified_deepsoz_target_v2_artifact(
    artifact_directory: str | Path,
    source_csv: str | Path,
    split_csv: str | Path,
    *,
    expected_target_artifact_sha256: str,
    expected_summary_artifact_sha256: str,
    expected_readme_artifact_sha256: str,
    expected_source_input_sha256: str,
    expected_split_input_sha256: str,
) -> VerifiedDeepSOZTargetV2Artifact:
    """Strictly rebuild and verify a legacy target-v2 artifact.

    Every persisted file and both registry inputs require an independently
    supplied exact-byte SHA.  The absolute input paths recorded by the legacy
    summary are validated only as non-empty strings and are never resolved,
    opened, returned, or treated as provenance.  The caller-supplied source
    and split files are the sole inputs to the rebuilt registry.
    """

    bundle = _reject_symlink_components(
        Path(artifact_directory), field="Target-v2 artifact directory"
    )
    if not bundle.is_dir():
        raise FileNotFoundError("Target-v2 artifact directory does not exist")
    entries = tuple(sorted(bundle.iterdir(), key=lambda path: path.name))
    names = {entry.name for entry in entries}
    if names != _V2_OUTPUT_FILE_SET or len(entries) != len(_V2_OUTPUT_FILES):
        raise ValueError(
            "Target-v2 artifact violates its closed file schema; "
            f"missing={sorted(_V2_OUTPUT_FILE_SET-names)}, "
            f"unknown={sorted(names-_V2_OUTPUT_FILE_SET)}"
        )
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("Target-v2 artifact files must be regular and non-symlinked")

    target_bytes, target_sha = _read_stable_regular_file(
        bundle / TARGETS_FILENAME,
        field="Target-v2 patient target CSV",
        max_bytes=_MAX_TARGET_BYTES,
    )
    summary_bytes, summary_sha = _read_stable_regular_file(
        bundle / SUMMARY_FILENAME,
        field="Target-v2 summary JSON",
        max_bytes=_MAX_SUMMARY_BYTES,
    )
    readme_bytes, readme_sha = _read_stable_regular_file(
        bundle / README_FILENAME,
        field="Target-v2 README",
        max_bytes=_MAX_README_BYTES,
    )
    source_bytes, source_sha = _read_stable_regular_file(
        source_csv,
        field="DeepSOZ source CSV",
        max_bytes=_MAX_INPUT_CSV_BYTES,
    )
    split_bytes, split_sha = _read_stable_regular_file(
        split_csv,
        field="DeepSOZ split manifest CSV",
        max_bytes=_MAX_INPUT_CSV_BYTES,
    )
    _check_expected_sha(
        target_sha,
        expected_target_artifact_sha256,
        field="expected_target_artifact_sha256",
    )
    _check_expected_sha(
        summary_sha,
        expected_summary_artifact_sha256,
        field="expected_summary_artifact_sha256",
    )
    _check_expected_sha(
        readme_sha,
        expected_readme_artifact_sha256,
        field="expected_readme_artifact_sha256",
    )
    _check_expected_sha(
        source_sha,
        expected_source_input_sha256,
        field="expected_source_input_sha256",
    )
    _check_expected_sha(
        split_sha,
        expected_split_input_sha256,
        field="expected_split_input_sha256",
    )

    source = _parse_registry_csv(
        source_bytes,
        field="DeepSOZ source CSV",
        allow_legacy_pz_pair=True,
    )
    split = _parse_registry_csv(
        split_bytes,
        field="DeepSOZ split manifest CSV",
    )
    registry = build_deepsoz_reference_registry(source, split)
    expected_frame = _build_target_frame(
        registry,
        source_sha256=source_sha,
        split_sha256=split_sha,
    )
    _validate_target_csv_header(
        target_bytes,
        expected_columns=expected_frame.columns,
    )
    expected_target_bytes = _target_csv_bytes(expected_frame)
    if target_bytes != expected_target_bytes:
        raise ValueError(
            "Target-v2 CSV does not exactly match the registry rebuilt from "
            "the pinned source and split inputs"
        )

    summary = _parse_summary_json(summary_bytes)
    expected_summary = _build_summary(
        registry=registry,
        frame=expected_frame,
        source=source,
        split=split,
        source_csv=Path(source_csv),
        split_csv=Path(split_csv),
        source_sha256=source_sha,
        split_sha256=split_sha,
    )
    expected_readme_bytes = _render_readme(expected_summary).encode("utf-8")
    if readme_bytes != expected_readme_bytes:
        raise ValueError("Target-v2 README does not match the frozen policy rebuild")
    expected_summary["artifact_sha256"] = {
        TARGETS_FILENAME: target_sha,
        README_FILENAME: readme_sha,
    }
    if _canonical_json_bytes(_summary_for_comparison(summary)) != (
        _canonical_json_bytes(_summary_for_comparison(expected_summary))
    ):
        raise ValueError(
            "Target-v2 summary counts, hashes, schema, or policy do not exactly "
            "match the rebuilt registry"
        )

    patient_ids = tuple(reference.patient_id for reference in registry)
    eligible_ids = tuple(
        reference.patient_id
        for reference in registry
        if reference.eligible_for_localization
    )
    eligible_split_ids = tuple(
        (
            model_split,
            tuple(
                reference.patient_id
                for reference in registry
                if reference.eligible_for_localization
                and reference.model_split == model_split
            ),
        )
        for model_split in _ELIGIBLE_SPLITS
    )
    receipt = DeepSOZTargetV2VerifiedReceipt(
        target_artifact_sha256=target_sha,
        summary_artifact_sha256=summary_sha,
        readme_artifact_sha256=readme_sha,
        source_input_sha256=source_sha,
        split_input_sha256=split_sha,
        policy_sha256=TARGET_V2_POLICY_SHA256,
        patient_ids=patient_ids,
        eligible_patient_ids=eligible_ids,
        eligible_split_patient_ids=eligible_split_ids,
        patient_roster_sha256=_patient_roster_sha256(patient_ids),
        eligible_patient_roster_sha256=_patient_roster_sha256(eligible_ids),
        patient_count=len(patient_ids),
        eligible_patient_count=len(eligible_ids),
    )
    return VerifiedDeepSOZTargetV2Artifact(registry=registry, receipt=receipt)


def build_deepsoz_target_v2_artifact(
    source_csv: str | Path,
    split_csv: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a self-contained v2 target artifact without touching v1 files."""

    source_path = Path(source_csv)
    split_path = Path(split_csv)
    output_path = Path(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    _guard_output_directory(output_path, split_path)

    source_sha256 = sha256_file(source_path)
    split_sha256 = sha256_file(split_path)
    source = pd.read_csv(source_path)
    split = pd.read_csv(split_path)
    registry = build_deepsoz_reference_registry(source, split)
    frame = _build_target_frame(
        registry,
        source_sha256=source_sha256,
        split_sha256=split_sha256,
    )
    summary = _build_summary(
        registry=registry,
        frame=frame,
        source=source,
        split=split,
        source_csv=source_path,
        split_csv=split_path,
        source_sha256=source_sha256,
        split_sha256=split_sha256,
    )
    readme = _render_readme(summary)

    existing = [name for name in _V2_OUTPUT_FILES if (output_path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Target-v2 output already contains {existing}; pass overwrite=True "
            "to replace only v2 files"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path / TARGETS_FILENAME, index=False, encoding="utf-8")
    (output_path / README_FILENAME).write_text(readme, encoding="utf-8")
    summary["artifact_sha256"] = {
        TARGETS_FILENAME: sha256_file(output_path / TARGETS_FILENAME),
        README_FILENAME: sha256_file(output_path / README_FILENAME),
    }
    (output_path / SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "DeepSOZTargetV2VerifiedReceipt",
    "README_FILENAME",
    "SUMMARY_FILENAME",
    "TARGETS_FILENAME",
    "TARGET_V2_POLICY_VERSION",
    "TARGET_V2_POLICY_SHA256",
    "TARGET_V2_SCHEMA_VERSION",
    "VERIFIED_TARGET_V2_RECEIPT_SCHEMA",
    "VerifiedDeepSOZTargetV2Artifact",
    "build_deepsoz_target_v2_artifact",
    "load_verified_deepsoz_target_v2_artifact",
    "sha256_file",
]
