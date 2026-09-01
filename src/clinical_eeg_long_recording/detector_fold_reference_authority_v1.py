"""Fold- and phase-scoped TUSZ detector-reference authority.

This additive module exposes public TUSZ global ``TERM,seiz`` intervals to
exactly one outer fold and one training phase at a time.  It is intentionally
incapable of materialising an outer-held-out, source-development,
source-evaluation, or private reference.  The three authorities are:

``selection_fit``
    The three source-train groups excluding the outer-held-out group and the
    next-fold inner-validation group.
``inner_validation``
    The next-fold validation group, opened only after a content-addressed
    selection-fit prediction/checkpoint freeze is supplied.
``final_refit``
    All four outer-training groups, including the former inner-validation
    group, opened only after the selected epoch is frozen.  The outer-held-out
    group remains excluded.

Every authority binds the canonical five-fold plan and exact phase roster,
the acquisition-header registry, the exact ``TERM,seiz`` parser source, and
the bytes plus projected event inventory of every opened sidecar.  No EDF
annotation, channel label target, SOZ label, spreadsheet, doctor text, source
eval reference, or private reference has an input slot.

The output is a training-target authority/lineage artifact, not performance
evidence and not an authorization to open an outer-held-out reference.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping, Sequence

from .continuous_detection_source_dev_join import (
    SOURCE_DEV_REFERENCE_PARSER_ID,
    parse_tusz_term_seiz_reference_bytes,
)
from .detector_reference_phase_gate_v1 import (
    DETECTOR_PREDICTION_CONTRACT_ID_V1,
    DETECTOR_REFERENCE_PHASE_GATE_EXECUTION_STATUS_V1,
    DETECTOR_REFERENCE_PHASE_GATE_VALIDATOR_ID_V1,
    DETECTOR_SELECTION_SCORER_ID_V1,
    DETECTOR_SELECTION_SCORER_VERSION_V1,
    DetectorReferenceGateReplayV1,
    build_detector_selection_metric_receipt_v1,
    detector_reference_phase_gate_source_sha256_v1,
    replay_detector_reference_phase_gate_v1,
    validate_detector_controller_signature_authority_v1,
    validate_detector_reference_phase_gate_proof_v1,
)
from .eeg_acquisition_header_allowlist_v1 import (
    EEG_ACQUISITION_HEADER_PARSER_ID_V1,
    acquisition_header_parser_source_sha256_v1,
    build_eeg_acquisition_header_allowlist_policy_v1,
)
from .tusz_detector_cleanroom_fold_plan_v1 import (
    TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1,
    validate_tusz_detector_cleanroom_fold_plan_v1,
)


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH: Final[Path] = (
    ROOT / "configs" / "clinical_eeg_detector_fold_reference_authority_registry_v1.json"
)
DETECTOR_FOLD_REFERENCE_AUTHORITY_REGISTRY_SCHEMA_V1: Final[
    str
] = "clinical_eeg_detector_fold_reference_authority_registry_v1"
DETECTOR_FOLD_REFERENCE_PHASE_RECEIPT_SCHEMA_V1: Final[
    str
] = "clinical_eeg_detector_fold_reference_phase_receipt_v1"
DETECTOR_FOLD_REFERENCE_AUTHORITY_METHOD_ID_V1: Final[
    str
] = "next_outer_fold_inner_validation_phase_isolated_exact_TERM_seiz_v1"
REFERENCE_SIDECAR_MAPPING_ID_V1: Final[
    str
] = "source_train_edf_suffix_to_same_recording_csv_bi_v1"
REFERENCE_PHASES_V1: Final[tuple[str, ...]] = (
    "selection_fit",
    "inner_validation",
    "final_refit",
)
DEFAULT_CONTROLLER_KEY_ID_V1: Final[str] = (
    "CLINICAL-EEG-OFFLINE-PHASE-CONTROLLER-ED25519-20260824"
)
# Generated offline for this authority and immediately discarded.  Only the
# raw Ed25519 public key is checked in; no private/signing material exists in
# the repository or runtime API.
DEFAULT_CONTROLLER_PUBLIC_KEY_HEX_V1: Final[str] = (
    "6927c5346dd579723f52e1c29d8f6aeeda403a24a67e68cbd2905784ab9840af"
)
CHECKED_TUSZ_FOLD_PLAN_RECEIPT_SHA256_V1: Final[str] = (
    "ddd668de44a5566a329ba126e3d60d4415f20f5c856e6c847f396d515f7e4a1b"
)

_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")
_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "analysis_identity_id",
        "source_edf_relative_path",
        "reference_relative_path",
        "recording_duration_seconds_fraction",
        "reference_file_sha256",
        "reference_file_bytes",
        "selected_term_seiz_event_count",
        "ignored_non_term_seiz_row_count",
        "seizure_intervals",
        "event_inventory_sha256",
    }
)
_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "authority_id",
        "method_id",
        "registry_id",
        "registry_receipt_sha256",
        "outer_fold_id",
        "phase",
        "phase_gate",
        "fold_plan_binding",
        "parser_bindings",
        "authorized_fold_ids",
        "authorized_roster",
        "forbidden_outer_heldout_roster",
        "records",
        "selection_metric_receipt",
        "reference_file_sha256_roster_sha256",
        "reference_event_inventory_sha256",
        "reference_open_log",
        "scope_receipt",
        "receipt_sha256",
    }
)

_OPAQUE_PHASE_AUTHORITY_ISSUER_SEAL_V1 = object()


class ValidatedDetectorFoldReferencePhaseAuthorityV1(Mapping[str, Any]):
    """Process-local opaque authority issued only after all required replay.

    A JSON/self-hashed receipt is evidence to replay, not authority by itself.
    This wrapper cannot be reconstructed from serialized fields; callers must
    obtain it from the materializer or the explicit actual-byte replay issuer.
    Nested values are deep-copied on access so the sealed receipt cannot be
    mutated after issuance.
    """

    __slots__ = ("__receipt", "__issuer_seal")

    def __init__(self, receipt: Mapping[str, Any], *, _issuer_seal: object) -> None:
        if _issuer_seal is not _OPAQUE_PHASE_AUTHORITY_ISSUER_SEAL_V1:
            raise PermissionError("opaque detector phase authority has no valid issuer seal")
        self.__receipt = deepcopy(dict(receipt))
        self.__issuer_seal = _issuer_seal

    def __getitem__(self, key: str) -> Any:
        return deepcopy(self.__receipt[key])

    def __iter__(self):
        return iter(self.__receipt)

    def __len__(self) -> int:
        return len(self.__receipt)

    def __deepcopy__(self, memo):
        del memo
        return self

    def to_receipt(self) -> dict[str, Any]:
        return deepcopy(self.__receipt)

    def _has_valid_issuer_seal(self) -> bool:
        return self.__issuer_seal is _OPAQUE_PHASE_AUTHORITY_ISSUER_SEAL_V1


def _issue_opaque_phase_authority_v1(
    receipt: Mapping[str, Any],
) -> ValidatedDetectorFoldReferencePhaseAuthorityV1:
    return ValidatedDetectorFoldReferencePhaseAuthorityV1(
        receipt, _issuer_seal=_OPAQUE_PHASE_AUTHORITY_ISSUER_SEAL_V1
    )


def require_validated_detector_fold_reference_phase_authority_v1(
    value: object,
) -> ValidatedDetectorFoldReferencePhaseAuthorityV1:
    """Require a process-local authority; raw receipt mappings fail closed."""

    if (
        not isinstance(value, ValidatedDetectorFoldReferencePhaseAuthorityV1)
        or not value._has_valid_issuer_seal()
    ):
        raise PermissionError(
            "formal detector reference consumption requires an opaque actual-byte-replayed authority"
        )
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detector_reference_parser_source_sha256_v1() -> str:
    return _file_sha256(
        Path(__file__)
        .with_name("continuous_detection_source_dev_join.py")
        .resolve(strict=True)
    )


def detector_reference_authority_source_sha256_v1() -> str:
    return _file_sha256(Path(__file__).resolve(strict=True))


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a normalized identifier")
    return value


def _fraction(value: object, context: str, *, positive: bool) -> Fraction:
    if (
        type(value) is not list
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
    ):
        raise ValueError(f"{context} must be a reduced fraction")
    result = Fraction(value[0], value[1])
    if [result.numerator, result.denominator] != value:
        raise ValueError(f"{context} fraction must be reduced")
    if (positive and result <= 0) or (not positive and result < 0):
        raise ValueError(f"{context} has invalid sign")
    return result


def _fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _phase_fold_ids(outer_fold_id: int, phase: str) -> list[int]:
    fold_ids = list(range(TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1))
    if type(outer_fold_id) is not int or outer_fold_id not in fold_ids:
        raise ValueError("outer fold ID is invalid")
    if phase not in REFERENCE_PHASES_V1:
        raise ValueError("detector reference phase is invalid")
    inner = (outer_fold_id + 1) % TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1
    if phase == "selection_fit":
        return [value for value in fold_ids if value not in {outer_fold_id, inner}]
    if phase == "inner_validation":
        return [inner]
    return [value for value in fold_ids if value != outer_fold_id]


def _source_train_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        deepcopy(row)
        for row in plan["source_record_duration_rows"]
        if row["model_split"] == "source_train"
    ]
    expected = plan["source_split_rosters"]["source_train"]
    if len(rows) != expected["recording_count"]:
        raise ValueError("source-train duration-row denominator drifted")
    return rows


def _phase_rows(
    plan: Mapping[str, Any], *, outer_fold_id: int, phase: str
) -> list[dict[str, Any]]:
    authorized_folds = set(_phase_fold_ids(outer_fold_id, phase))
    assignments = {
        row["local_patient_id"]: row["held_out_fold_id"]
        for row in plan["patient_fold_assignments"]
    }
    rows = []
    for row in _source_train_rows(plan):
        fold_id = assignments.get(row["local_patient_id"])
        if fold_id is None:
            raise ValueError("source-train patient is absent from fold assignments")
        if fold_id in authorized_folds:
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (row["analysis_identity_id"], row["local_edf_path"]),
    )


def _fold_group_rows(
    plan: Mapping[str, Any], *, held_out_fold_id: int
) -> list[dict[str, Any]]:
    if type(held_out_fold_id) is not int or held_out_fold_id not in range(
        TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1
    ):
        raise ValueError("held-out fold ID is invalid")
    assignments = {
        row["local_patient_id"]: row["held_out_fold_id"]
        for row in plan["patient_fold_assignments"]
    }
    return sorted(
        [
            row
            for row in _source_train_rows(plan)
            if assignments[row["local_patient_id"]] == held_out_fold_id
        ],
        key=lambda row: (row["analysis_identity_id"], row["local_edf_path"]),
    )


def _roster_view(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identities = sorted(str(row["analysis_identity_id"]) for row in rows)
    paths = sorted(str(row["local_edf_path"]) for row in rows)
    patients = sorted({str(row["local_patient_id"]) for row in rows})
    durations = [
        {
            "analysis_identity_id": row["analysis_identity_id"],
            "recording_duration_seconds_fraction": row[
                "recording_duration_seconds_fraction"
            ],
        }
        for row in sorted(rows, key=lambda item: str(item["analysis_identity_id"]))
    ]
    total_duration = sum(
        (
            _fraction(
                row["recording_duration_seconds_fraction"],
                "recording duration",
                positive=True,
            )
            for row in rows
        ),
        Fraction(0, 1),
    )
    return {
        "patient_count": len(patients),
        "recording_count": len(rows),
        "duration_seconds_fraction": _fraction_json(total_duration),
        "patient_roster_sha256": _canonical_sha256(patients),
        "analysis_identity_roster_sha256": _canonical_sha256(identities),
        "local_edf_path_roster_sha256": _canonical_sha256(paths),
        "record_duration_binding_sha256": _canonical_sha256(durations),
    }


def _gate_record_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project the record identities that a typed gate must bind exactly."""

    return [
        {
            "analysis_identity_id": str(row["analysis_identity_id"]),
            "local_patient_id": str(row["local_patient_id"]),
            "source_edf_relative_path": str(row["local_edf_path"]),
            "recording_duration_seconds_fraction": deepcopy(
                row["recording_duration_seconds_fraction"]
            ),
        }
        for row in sorted(
            rows,
            key=lambda item: (
                str(item["analysis_identity_id"]),
                str(item["local_edf_path"]),
            ),
        )
    ]


def _phase_mapping_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for outer_fold_id in range(TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1):
        phases = {}
        for phase in REFERENCE_PHASES_V1:
            rows = _phase_rows(plan, outer_fold_id=outer_fold_id, phase=phase)
            phases[phase] = {
                "authorized_fold_ids": _phase_fold_ids(outer_fold_id, phase),
                "authorized_roster": _roster_view(rows),
                "gate": {
                    "selection_fit": "registry_and_target_blind_fold_plan_freeze",
                    "inner_validation": (
                        "controller_signed_prediction_first_actual_byte_replay_"
                        "before_exact_reference_release"
                    ),
                    "final_refit": (
                        "controller_signed_recomputed_selected_epoch_and_from_"
                        "scratch_final_refit_before_exact_reference_release"
                    ),
                }[phase],
            }
        result.append(
            {
                "outer_fold_id": outer_fold_id,
                "inner_validation_fold_id": (outer_fold_id + 1)
                % TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1,
                "phases": phases,
                "outer_heldout_roster": _roster_view(
                    _fold_group_rows(plan, held_out_fold_id=outer_fold_id)
                ),
            }
        )
    return result


def build_detector_fold_reference_authority_registry_v1(
    *,
    fold_plan: Mapping[str, Any],
    fold_plan_path: str,
    fold_plan_file_sha256: str,
    detector_protocol_binding: Mapping[str, Any],
    acquisition_header_parser_path: str,
    acquisition_header_policy_path: str,
    acquisition_header_policy_file_sha256: str,
    reference_parser_path: str,
    controller_key_id: str = DEFAULT_CONTROLLER_KEY_ID_V1,
    controller_public_key_hex: str = DEFAULT_CONTROLLER_PUBLIC_KEY_HEX_V1,
) -> dict[str, Any]:
    """Build a content-addressed, reference-free phase authority registry."""

    plan = validate_tusz_detector_cleanroom_fold_plan_v1(fold_plan)
    if (
        plan["receipt_sha256"] == CHECKED_TUSZ_FOLD_PLAN_RECEIPT_SHA256_V1
        and (
            controller_key_id != DEFAULT_CONTROLLER_KEY_ID_V1
            or controller_public_key_hex != DEFAULT_CONTROLLER_PUBLIC_KEY_HEX_V1
        )
    ):
        raise PermissionError(
            "checked TUSZ fold plan is permanently rooted in the checked controller public key"
        )
    _sha256(fold_plan_file_sha256, "fold-plan file hash")
    _sha256(
        acquisition_header_policy_file_sha256,
        "acquisition-header policy file hash",
    )
    detector_binding = deepcopy(dict(detector_protocol_binding))
    for key in ("path", "file_sha256", "semantic_receipt_sha256"):
        if key not in detector_binding:
            raise ValueError("detector protocol binding fields drifted")
    if set(detector_binding) != {"path", "file_sha256", "semantic_receipt_sha256"}:
        raise ValueError("detector protocol binding fields drifted")
    _sha256(detector_binding["file_sha256"], "detector protocol file hash")
    _sha256(
        detector_binding["semantic_receipt_sha256"],
        "detector protocol semantic receipt",
    )
    header_policy = build_eeg_acquisition_header_allowlist_policy_v1()
    registry: dict[str, Any] = {
        "schema_version": DETECTOR_FOLD_REFERENCE_AUTHORITY_REGISTRY_SCHEMA_V1,
        "registry_id": "CLINICAL-EEG-DETECTOR-FOLD-REFERENCE-AUTHORITY-V1-20260824",
        "status": "phase_registry_controller_signed_nonselection_executable_v1",
        "method_id": DETECTOR_FOLD_REFERENCE_AUTHORITY_METHOD_ID_V1,
        "materializer_binding": {
            "path": (
                "src/clinical_eeg_long_recording/"
                "detector_fold_reference_authority_v1.py"
            ),
            "source_sha256": detector_reference_authority_source_sha256_v1(),
        },
        "detector_protocol_binding": detector_binding,
        "fold_plan_binding": {
            "path": fold_plan_path,
            "file_sha256": fold_plan_file_sha256,
            "plan_id": plan["plan_id"],
            "plan_receipt_sha256": plan["receipt_sha256"],
            "fold_count": plan["fold_count"],
            "source_train_patient_roster_sha256": plan["source_split_rosters"][
                "source_train"
            ]["patient_roster_sha256"],
            "source_train_analysis_identity_roster_sha256": plan[
                "source_split_rosters"
            ]["source_train"]["analysis_identity_roster_sha256"],
        },
        "acquisition_header_parser_binding": {
            "path": acquisition_header_parser_path,
            "parser_id": EEG_ACQUISITION_HEADER_PARSER_ID_V1,
            "parser_source_sha256": acquisition_header_parser_source_sha256_v1(),
            "policy_id": header_policy["policy_id"],
            "policy_receipt_sha256": header_policy["policy_receipt_sha256"],
            "policy_path": acquisition_header_policy_path,
            "policy_file_sha256": acquisition_header_policy_file_sha256,
            "patient_identity_or_free_text_in_receipt": False,
            "annotation_or_sample_payload_read": False,
        },
        "reference_parser_binding": {
            "path": reference_parser_path,
            "parser_id": SOURCE_DEV_REFERENCE_PARSER_ID,
            "parser_source_sha256": detector_reference_parser_source_sha256_v1(),
            "mapping_id": REFERENCE_SIDECAR_MAPPING_ID_V1,
            "projection": "exact_global_TERM_seiz_intervals_only",
        },
        "phase_gate_validator_binding": {
            "path": (
                "src/clinical_eeg_long_recording/"
                "detector_reference_phase_gate_v1.py"
            ),
            "validator_id": DETECTOR_REFERENCE_PHASE_GATE_VALIDATOR_ID_V1,
            "validator_source_sha256": (
                detector_reference_phase_gate_source_sha256_v1()
            ),
        },
        "controller_signature_authority": {
            "algorithm": "ed25519",
            "controller_key_id": _identifier(
                controller_key_id, "controller key ID"
            ),
            "public_key_hex": controller_public_key_hex,
            "private_key_material_in_repository": False,
            "canonical_signed_message": (
                "canonical_utf8_json_of_ledger_body_including_ledger_body_sha256"
            ),
            "missing_crypto_dependency_behavior": "fail_closed",
        },
        "selection_scorer_authority": {
            "scorer_id": DETECTOR_SELECTION_SCORER_ID_V1,
            "scorer_version": DETECTOR_SELECTION_SCORER_VERSION_V1,
            "supported_prediction_contract_ids": [
                DETECTOR_PREDICTION_CONTRACT_ID_V1
            ],
            "unsupported_provider_or_contract_behavior": "fail_closed",
            "caller_supplied_metric_values_accepted": False,
            "reference_projection": "exact_global_TERM_seiz_intervals_only",
        },
        "phase_mappings": _phase_mapping_rows(plan),
        "access_firewall": {
            "one_materialization_call_scope": "one_outer_fold_and_one_phase",
            "allowed_dataset_and_split": "public_TUSZ_source_train_only",
            "outer_heldout_reference_access_in_this_authority": False,
            "outer_heldout_reference_access_before_prediction_inventory_freeze": False,
            "source_dev_reference_access": False,
            "source_eval_reference_access": False,
            "private_reference_access": False,
            "EDF_annotations_or_annotation_channels_used": False,
            "channel_specific_annotations_or_SOZ_labels_used": False,
            "Excel_doctor_report_clinical_text_or_behaviour_used": False,
        },
        "phase_gate_contract": {
            "selection_fit": "registry_and_target_blind_fold_plan_freeze",
            "inner_validation": (
                "controller_signed_prediction_first_actual_byte_replay_then_"
                "pre_reference_timing_receipt_then_exact_reference_open"
            ),
            "final_refit": (
                "controller_signed_exact_scorer_recomputed_selected_epoch_and_"
                "from_scratch_refit_all_four_outer_train_folds"
            ),
            "execution_status": DETECTOR_REFERENCE_PHASE_GATE_EXECUTION_STATUS_V1,
            "bare_caller_supplied_sha256_accepted": False,
            "caller_supplied_expected_trust_anchor_hash_accepted": False,
            "controller_owned_gate_ledger_materialized": True,
            "controller_signature_scheme": "ed25519",
            "controller_private_key_in_repository": False,
            "typed_prediction_artifact_byte_replay_materialized": True,
            "typed_terminal_prediction_semantics_materialized": True,
            "selected_epoch_metric_scorer_recompute_materialized": True,
            "nonselection_reference_phases_executable": True,
            "gate_validated_before_first_reference_sidecar_open": True,
            "checkpoint_artifact_bytes_replayed": True,
            "complete_candidate_epoch_by_record_prediction_inventory_required": True,
            "actual_prior_phase_exposure_receipts_required": True,
            "outer_heldout_scoring": "not_an_authority_in_this_registry",
        },
        "scientific_scope": {
            "training_target_authority_only": True,
            "detector_trained_by_registry": False,
            "prediction_or_performance_materialized": False,
            "clinical_or_production_use_authorized": False,
        },
        "registry_receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    registry["registry_receipt_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in registry.items()
            if key != "registry_receipt_sha256"
        }
    )
    return registry


def validate_detector_fold_reference_authority_registry_v1(
    value: Mapping[str, Any],
    *,
    fold_plan: Mapping[str, Any],
    verify_bound_files: bool = True,
) -> dict[str, Any]:
    registry = deepcopy(dict(value))
    if registry.get("schema_version") != (
        DETECTOR_FOLD_REFERENCE_AUTHORITY_REGISTRY_SCHEMA_V1
    ):
        raise ValueError("detector reference-authority registry schema drifted")
    if registry.get("method_id") != DETECTOR_FOLD_REFERENCE_AUTHORITY_METHOD_ID_V1:
        raise ValueError("detector reference-authority method drifted")
    observed = registry.get("registry_receipt_sha256")
    _sha256(observed, "registry receipt")
    if observed != _canonical_sha256(
        {
            key: item
            for key, item in registry.items()
            if key != "registry_receipt_sha256"
        }
    ):
        raise ValueError("detector reference-authority registry does not replay")
    plan = validate_tusz_detector_cleanroom_fold_plan_v1(fold_plan)
    materializer_binding = registry["materializer_binding"]
    if materializer_binding.get("source_sha256") != (
        detector_reference_authority_source_sha256_v1()
    ):
        raise ValueError("detector reference-authority materializer source drifted")
    binding = registry["fold_plan_binding"]
    if (
        binding["plan_id"] != plan["plan_id"]
        or binding["plan_receipt_sha256"] != plan["receipt_sha256"]
        or binding["fold_count"] != TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1
    ):
        raise ValueError("detector reference-authority fold-plan binding drifted")
    if registry["phase_mappings"] != _phase_mapping_rows(plan):
        raise ValueError("detector reference-authority phase rosters drifted")
    header = registry["acquisition_header_parser_binding"]
    policy = build_eeg_acquisition_header_allowlist_policy_v1()
    if (
        header["parser_id"] != EEG_ACQUISITION_HEADER_PARSER_ID_V1
        or header["policy_id"] != policy["policy_id"]
        or header["policy_receipt_sha256"] != policy["policy_receipt_sha256"]
        or header["patient_identity_or_free_text_in_receipt"] is not False
        or header["annotation_or_sample_payload_read"] is not False
    ):
        raise ValueError("acquisition-header parser authority drifted")
    if header["parser_source_sha256"] != acquisition_header_parser_source_sha256_v1():
        raise ValueError("acquisition-header parser source binding drifted")
    reference = registry["reference_parser_binding"]
    if (
        reference["parser_id"] != SOURCE_DEV_REFERENCE_PARSER_ID
        or reference["mapping_id"] != REFERENCE_SIDECAR_MAPPING_ID_V1
        or reference["projection"] != "exact_global_TERM_seiz_intervals_only"
    ):
        raise ValueError("detector reference parser authority drifted")
    if (
        reference["parser_source_sha256"]
        != detector_reference_parser_source_sha256_v1()
    ):
        raise ValueError("detector reference parser source binding drifted")
    gate_validator = registry["phase_gate_validator_binding"]
    if (
        gate_validator.get("validator_id")
        != DETECTOR_REFERENCE_PHASE_GATE_VALIDATOR_ID_V1
        or gate_validator.get("validator_source_sha256")
        != detector_reference_phase_gate_source_sha256_v1()
    ):
        raise ValueError("detector reference phase-gate validator binding drifted")
    controller_authority = validate_detector_controller_signature_authority_v1(
        registry["controller_signature_authority"]
    )
    if (
        plan["receipt_sha256"] == CHECKED_TUSZ_FOLD_PLAN_RECEIPT_SHA256_V1
        and (
            controller_authority["controller_key_id"]
            != DEFAULT_CONTROLLER_KEY_ID_V1
            or controller_authority["public_key_hex"]
            != DEFAULT_CONTROLLER_PUBLIC_KEY_HEX_V1
        )
    ):
        raise PermissionError("checked TUSZ controller signature root was replaced")
    scorer = registry["selection_scorer_authority"]
    if scorer != {
        "scorer_id": DETECTOR_SELECTION_SCORER_ID_V1,
        "scorer_version": DETECTOR_SELECTION_SCORER_VERSION_V1,
        "supported_prediction_contract_ids": [
            DETECTOR_PREDICTION_CONTRACT_ID_V1
        ],
        "unsupported_provider_or_contract_behavior": "fail_closed",
        "caller_supplied_metric_values_accepted": False,
        "reference_projection": "exact_global_TERM_seiz_intervals_only",
    }:
        raise PermissionError("detector selection scorer authority drifted")
    firewall = registry["access_firewall"]
    for key in (
        "outer_heldout_reference_access_in_this_authority",
        "outer_heldout_reference_access_before_prediction_inventory_freeze",
        "source_dev_reference_access",
        "source_eval_reference_access",
        "private_reference_access",
        "EDF_annotations_or_annotation_channels_used",
        "channel_specific_annotations_or_SOZ_labels_used",
        "Excel_doctor_report_clinical_text_or_behaviour_used",
    ):
        if firewall.get(key) is not False:
            raise PermissionError(f"detector reference firewall opened: {key}")
    gate_contract = registry["phase_gate_contract"]
    if (
        "all_four_outer_train_folds"
        not in gate_contract["final_refit"]
        or gate_contract.get("execution_status")
        != DETECTOR_REFERENCE_PHASE_GATE_EXECUTION_STATUS_V1
        or gate_contract.get("bare_caller_supplied_sha256_accepted") is not False
        or gate_contract.get(
            "caller_supplied_expected_trust_anchor_hash_accepted"
        )
        is not False
        or gate_contract.get("controller_owned_gate_ledger_materialized")
        is not True
        or gate_contract.get("controller_signature_scheme") != "ed25519"
        or gate_contract.get("controller_private_key_in_repository") is not False
        or gate_contract.get(
            "typed_prediction_artifact_byte_replay_materialized"
        )
        is not True
        or gate_contract.get(
            "typed_terminal_prediction_semantics_materialized"
        )
        is not True
        or gate_contract.get(
            "selected_epoch_metric_scorer_recompute_materialized"
        )
        is not True
        or gate_contract.get("nonselection_reference_phases_executable")
        is not True
        or gate_contract.get("gate_validated_before_first_reference_sidecar_open")
        is not True
        or gate_contract.get("checkpoint_artifact_bytes_replayed") is not True
        or gate_contract.get(
            "complete_candidate_epoch_by_record_prediction_inventory_required"
        )
        is not True
        or gate_contract.get("actual_prior_phase_exposure_receipts_required")
        is not True
    ):
        raise ValueError("non-selection detector controller gate drifted")
    if verify_bound_files:
        for section, hash_key in (
            (materializer_binding, "source_sha256"),
            (registry["detector_protocol_binding"], "file_sha256"),
            (binding, "file_sha256"),
            (header, "parser_source_sha256"),
            (reference, "parser_source_sha256"),
            (gate_validator, "validator_source_sha256"),
        ):
            path = ROOT / str(section["path"])
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"bound authority file missing or symlinked: {path}")
            if _file_sha256(path) != section[hash_key]:
                raise ValueError(f"bound authority file hash drifted: {path}")
        policy_path = ROOT / str(header["policy_path"])
        if not policy_path.is_file() or policy_path.is_symlink():
            raise ValueError("bound acquisition-header policy file is missing")
        if _file_sha256(policy_path) != header["policy_file_sha256"]:
            raise ValueError("bound acquisition-header policy file hash drifted")
        policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy_payload != policy:
            raise ValueError("bound acquisition-header policy semantics drifted")
        detector_protocol = json.loads(
            (ROOT / registry["detector_protocol_binding"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        if (
            detector_protocol.get("receipt_sha256")
            != registry["detector_protocol_binding"]["semantic_receipt_sha256"]
        ):
            raise ValueError("detector protocol semantic receipt drifted")
        frozen_mappings = detector_protocol["outer_patient_oof"][
            "fold_inner_selection_and_final_refit"
        ]["mappings"]
        expected_mappings = [
            {
                "outer_held_out_fold": row["outer_fold_id"],
                "inner_validation_fold": row["inner_validation_fold_id"],
                "selection_fit_folds": row["phases"]["selection_fit"][
                    "authorized_fold_ids"
                ],
                "final_refit_folds": row["phases"]["final_refit"][
                    "authorized_fold_ids"
                ],
            }
            for row in registry["phase_mappings"]
        ]
        if frozen_mappings != expected_mappings:
            raise ValueError("detector protocol phase mapping semantics drifted")
    return registry


def load_default_detector_fold_reference_authority_registry_v1() -> dict[str, Any]:
    registry = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
    fold_binding = registry["fold_plan_binding"]
    fold_plan = json.loads((ROOT / fold_binding["path"]).read_text(encoding="utf-8"))
    return validate_detector_fold_reference_authority_registry_v1(
        registry, fold_plan=fold_plan, verify_bound_files=True
    )


def _safe_reference_relative_path(edf_relative_path: str) -> PurePosixPath:
    value = _identifier(edf_relative_path, "source EDF relative path")
    if "\\" in value:
        raise ValueError("source EDF path must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "train"
        or "." in path.parts
        or ".." in path.parts
        or path.suffix.lower() != ".edf"
    ):
        raise PermissionError("reference path is outside public TUSZ source-train")
    return path.with_suffix(".csv_bi")


def _read_reference_bytes(root_value: Path, relative: PurePosixPath) -> bytes:
    root_input = Path(root_value)
    if root_input.is_symlink():
        raise ValueError("TUSZ reference root must not be a symlink")
    root = root_input.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("TUSZ reference root must be a directory")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("TUSZ reference path must not contain a symlink")
    candidate = cursor.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("TUSZ reference path escaped its canonical root") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError("TUSZ reference sidecar must be a regular file")
    return candidate.read_bytes()


def _phase_mapping(
    registry: Mapping[str, Any], *, outer_fold_id: int
) -> dict[str, Any]:
    rows = [
        row
        for row in registry["phase_mappings"]
        if row["outer_fold_id"] == outer_fold_id
    ]
    if len(rows) != 1:
        raise ValueError("outer-fold phase mapping is not unique")
    return deepcopy(rows[0])


def _replay_nonselection_phase_gate(
    *,
    plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    reference_root: Path,
    outer_fold_id: int,
    phase: str,
    controller_bundle_root: Path,
    controller_ledger_relative_path: str,
    prior_selection_fit_phase_receipt: Mapping[str, Any],
    prior_inner_validation_phase_receipt: Mapping[str, Any] | None,
    selected_epoch_metric_receipt: Mapping[str, Any] | None,
) -> DetectorReferenceGateReplayV1:
    """Replay signed prediction-first authority before any new reference opens."""

    if phase not in {"inner_validation", "final_refit"}:
        raise ValueError("typed gate replay is only valid for non-selection phases")
    mapping = _phase_mapping(registry, outer_fold_id=outer_fold_id)
    selection_mapping = mapping["phases"]["selection_fit"]
    inner_mapping = mapping["phases"]["inner_validation"]
    authorized_mapping = mapping["phases"][phase]
    validated_selection = validate_detector_fold_reference_phase_v1(
        prior_selection_fit_phase_receipt,
        fold_plan=plan,
        registry=registry,
        replay_reference_root=None,
    )
    if (
        validated_selection["phase"] != "selection_fit"
        or validated_selection["outer_fold_id"] != outer_fold_id
    ):
        raise PermissionError("prior selection-fit actual exposure receipt drifted")
    selection_receipt_sha256 = _sha256(
        validated_selection.get("receipt_sha256"),
        "prior selection-fit phase receipt",
    )
    if phase == "final_refit":
        if prior_inner_validation_phase_receipt is None:
            raise PermissionError("final-refit prior inner-validation receipt is missing")
        validated_inner = validate_detector_fold_reference_phase_v1(
            prior_inner_validation_phase_receipt,
            fold_plan=plan,
            registry=registry,
            replay_reference_root=None,
        )
        if (
            validated_inner["phase"] != "inner_validation"
            or validated_inner["outer_fold_id"] != outer_fold_id
            or validated_inner["selection_metric_receipt"]
            != selected_epoch_metric_receipt
        ):
            raise PermissionError("prior inner-validation metric/exposure receipt drifted")
    return replay_detector_reference_phase_gate_v1(
        bundle_root=controller_bundle_root,
        controller_ledger_relative_path=controller_ledger_relative_path,
        controller_signature_authority=registry["controller_signature_authority"],
        registry_id=registry["registry_id"],
        registry_receipt_sha256=registry["registry_receipt_sha256"],
        fold_plan_receipt_sha256=plan["receipt_sha256"],
        outer_fold_id=outer_fold_id,
        opens_phase=phase,
        reference_root=reference_root,
        authorized_fold_ids=authorized_mapping["authorized_fold_ids"],
        authorized_roster=authorized_mapping["authorized_roster"],
        authorized_rows=_gate_record_rows(
            _phase_rows(plan, outer_fold_id=outer_fold_id, phase=phase)
        ),
        selection_fit_fold_ids=selection_mapping["authorized_fold_ids"],
        selection_fit_roster=selection_mapping["authorized_roster"],
        selection_fit_phase_receipt_sha256=selection_receipt_sha256,
        inner_validation_fold_ids=inner_mapping["authorized_fold_ids"],
        inner_validation_roster=inner_mapping["authorized_roster"],
        inner_validation_rows=_gate_record_rows(
            _phase_rows(plan, outer_fold_id=outer_fold_id, phase="inner_validation")
        ),
        prior_inner_validation_phase_receipt=prior_inner_validation_phase_receipt,
        selected_epoch_metric_receipt=selected_epoch_metric_receipt,
    )


def materialize_detector_fold_reference_phase_v1(
    *,
    fold_plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    reference_root: Path,
    outer_fold_id: int,
    phase: str,
    phase_gate_receipt_sha256: str | None = None,
    controller_bundle_root: Path | None = None,
    controller_ledger_relative_path: str | None = None,
    prior_selection_fit_phase_receipt: Mapping[str, Any] | None = None,
    prior_inner_validation_phase_receipt: Mapping[str, Any] | None = None,
    selected_epoch_metric_receipt: Mapping[str, Any] | None = None,
) -> ValidatedDetectorFoldReferencePhaseAuthorityV1:
    """Open only one authorized source-train phase and seal exact references."""

    plan = validate_tusz_detector_cleanroom_fold_plan_v1(fold_plan)
    authority_registry = validate_detector_fold_reference_authority_registry_v1(
        registry, fold_plan=plan, verify_bound_files=False
    )
    phase_fold_ids = _phase_fold_ids(outer_fold_id, phase)
    mapping = _phase_mapping(authority_registry, outer_fold_id=outer_fold_id)
    phase_mapping = mapping["phases"][phase]
    if phase_mapping["authorized_fold_ids"] != phase_fold_ids:
        raise ValueError("phase fold IDs drifted")

    if phase_gate_receipt_sha256 is not None:
        raise PermissionError(
            "bare caller-supplied phase-gate SHA-256 values are forbidden"
        )
    if phase == "selection_fit":
        if any(
            value is not None
            for value in (
                controller_bundle_root,
                controller_ledger_relative_path,
                prior_selection_fit_phase_receipt,
                prior_inner_validation_phase_receipt,
                selected_epoch_metric_receipt,
            )
        ):
            raise PermissionError("selection-fit phase rejects controller/post-reference inputs")
        gate = {
            "gate_type": "registry_and_target_blind_fold_plan_freeze",
            "gate_receipt_sha256": authority_registry["registry_receipt_sha256"],
            "validated_before_first_reference_open": True,
        }
        gate_replay: DetectorReferenceGateReplayV1 | None = None
    else:
        if (
            controller_bundle_root is None
            or controller_ledger_relative_path is None
            or prior_selection_fit_phase_receipt is None
        ):
            raise PermissionError(
                f"{phase} requires a controller-signed ledger bundle and prior selection exposure"
            )
        prior_selection_fit_phase_receipt = (
            require_validated_detector_fold_reference_phase_authority_v1(
                prior_selection_fit_phase_receipt
            )
        )
        if phase == "final_refit":
            prior_inner_validation_phase_receipt = (
                require_validated_detector_fold_reference_phase_authority_v1(
                    prior_inner_validation_phase_receipt
                )
            )
        gate_replay = _replay_nonselection_phase_gate(
            plan=plan,
            registry=authority_registry,
            reference_root=Path(reference_root),
            outer_fold_id=outer_fold_id,
            phase=phase,
            controller_bundle_root=Path(controller_bundle_root),
            controller_ledger_relative_path=controller_ledger_relative_path,
            prior_selection_fit_phase_receipt=prior_selection_fit_phase_receipt,
            prior_inner_validation_phase_receipt=prior_inner_validation_phase_receipt,
            selected_epoch_metric_receipt=selected_epoch_metric_receipt,
        )
        gate = gate_replay.proof

    expected_rows = _phase_rows(plan, outer_fold_id=outer_fold_id, phase=phase)
    expected_roster = _roster_view(expected_rows)
    if expected_roster != phase_mapping["authorized_roster"]:
        raise ValueError("phase authorized roster drifted")
    outer_identities = set(
        plan["folds"][outer_fold_id]["held_out_roster"]["analysis_identity_ids"]
    )
    expected_identities = {row["analysis_identity_id"] for row in expected_rows}
    if outer_identities.intersection(expected_identities):
        raise PermissionError("outer-held-out identity entered reference authority")

    output_rows: list[dict[str, Any]] = []
    open_log_rows: list[dict[str, Any]] = []
    for sequence, row in enumerate(expected_rows):
        relative_reference = _safe_reference_relative_path(row["local_edf_path"])
        payload = _read_reference_bytes(Path(reference_root), relative_reference)
        duration = _fraction(
            row["recording_duration_seconds_fraction"],
            "recording duration",
            positive=True,
        )
        parsed = parse_tusz_term_seiz_reference_bytes(
            payload, duration_seconds=float(duration)
        )
        intervals = parsed.events()
        event_hash = _canonical_sha256(intervals)
        output_rows.append(
            {
                "analysis_identity_id": row["analysis_identity_id"],
                "source_edf_relative_path": row["local_edf_path"],
                "reference_relative_path": relative_reference.as_posix(),
                "recording_duration_seconds_fraction": _fraction_json(duration),
                "reference_file_sha256": parsed.reference_file_sha256,
                "reference_file_bytes": len(payload),
                "selected_term_seiz_event_count": (parsed.selected_term_seiz_row_count),
                "ignored_non_term_seiz_row_count": (
                    parsed.ignored_non_term_seiz_row_count
                ),
                "seizure_intervals": intervals,
                "event_inventory_sha256": event_hash,
            }
        )
        open_log_rows.append(
            {
                "open_sequence": sequence,
                "analysis_identity_id": row["analysis_identity_id"],
                "reference_relative_path": relative_reference.as_posix(),
                "reference_file_sha256": parsed.reference_file_sha256,
                "bytes_read": len(payload),
            }
        )

    parser_bindings = {
        "acquisition_header_parser_id": authority_registry[
            "acquisition_header_parser_binding"
        ]["parser_id"],
        "acquisition_header_parser_source_sha256": authority_registry[
            "acquisition_header_parser_binding"
        ]["parser_source_sha256"],
        "acquisition_header_policy_receipt_sha256": authority_registry[
            "acquisition_header_parser_binding"
        ]["policy_receipt_sha256"],
        "reference_parser_id": authority_registry["reference_parser_binding"][
            "parser_id"
        ],
        "reference_parser_source_sha256": authority_registry[
            "reference_parser_binding"
        ]["parser_source_sha256"],
        "reference_mapping_id": authority_registry["reference_parser_binding"][
            "mapping_id"
        ],
        "phase_gate_validator_id": authority_registry[
            "phase_gate_validator_binding"
        ]["validator_id"],
        "phase_gate_validator_source_sha256": authority_registry[
            "phase_gate_validator_binding"
        ]["validator_source_sha256"],
    }
    reference_file_hashes = [row["reference_file_sha256"] for row in output_rows]
    metric_receipt = None
    if phase == "inner_validation":
        if gate_replay is None:
            raise AssertionError("inner-validation gate replay was not retained")
        metric_receipt = build_detector_selection_metric_receipt_v1(
            prediction_artifacts=gate_replay.prediction_artifacts,
            checkpoint_hash_by_epoch=gate_replay.checkpoint_hash_by_epoch,
            prediction_artifact_inventory_sha256=gate[
                "prediction_artifact_inventory_sha256"
            ],
            reference_records=output_rows,
        )
    receipt: dict[str, Any] = {
        "schema_version": DETECTOR_FOLD_REFERENCE_PHASE_RECEIPT_SCHEMA_V1,
        "authority_id": (
            f"TUSZ-DETECTOR-REFERENCE-OUTER-{outer_fold_id:02d}-"
            f"{phase.upper().replace('_', '-')}-V1"
        ),
        "method_id": DETECTOR_FOLD_REFERENCE_AUTHORITY_METHOD_ID_V1,
        "registry_id": authority_registry["registry_id"],
        "registry_receipt_sha256": authority_registry["registry_receipt_sha256"],
        "outer_fold_id": outer_fold_id,
        "phase": phase,
        "phase_gate": gate,
        "fold_plan_binding": deepcopy(authority_registry["fold_plan_binding"]),
        "parser_bindings": parser_bindings,
        "authorized_fold_ids": phase_fold_ids,
        "authorized_roster": expected_roster,
        "forbidden_outer_heldout_roster": deepcopy(mapping["outer_heldout_roster"]),
        "records": output_rows,
        "selection_metric_receipt": metric_receipt,
        "reference_file_sha256_roster_sha256": _canonical_sha256(reference_file_hashes),
        "reference_event_inventory_sha256": _canonical_sha256(
            [
                {
                    "analysis_identity_id": row["analysis_identity_id"],
                    "event_inventory_sha256": row["event_inventory_sha256"],
                }
                for row in output_rows
            ]
        ),
        "reference_open_log": {
            "rows": open_log_rows,
            "open_log_sha256": _canonical_sha256(open_log_rows),
            "reference_files_opened": len(output_rows),
            "reference_bytes_read": sum(
                row["reference_file_bytes"] for row in output_rows
            ),
            "first_reference_open_after_phase_gate_validation": True,
            "pre_reference_release_receipt_sha256": (
                None
                if phase == "selection_fit"
                else gate["pre_reference_release_receipt"]["receipt_sha256"]
            ),
            "first_reference_open_logical_sequence": (
                1 if phase == "selection_fit" else 4
            ),
            "outer_heldout_reference_files_opened": 0,
            "source_dev_reference_files_opened": 0,
            "source_eval_reference_files_opened": 0,
            "private_reference_files_opened": 0,
        },
        "scope_receipt": {
            "public_TUSZ_source_train_only": True,
            "exact_global_TERM_seiz_intervals_only": True,
            "one_outer_fold_and_one_phase_only": True,
            "outer_heldout_reference_access_authorized": False,
            "outer_heldout_reference_access_before_prediction_inventory_freeze": False,
            "outer_heldout_prediction_or_metric_materialized": False,
            "source_dev_or_eval_reference_used": False,
            "private_reference_used": False,
            "EDF_annotations_or_annotation_channels_used": False,
            "channel_specific_annotations_or_SOZ_labels_used": False,
            "Excel_doctor_report_clinical_text_or_behaviour_used": False,
            "performance_or_clinical_claim_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    receipt["receipt_sha256"] = _canonical_sha256(
        {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    )
    validated_receipt = _validate_detector_fold_reference_phase_receipt_v1(
        receipt,
        fold_plan=plan,
        registry=authority_registry,
        replay_reference_root=None,
    )
    return _issue_opaque_phase_authority_v1(validated_receipt)


def _validate_detector_fold_reference_phase_receipt_v1(
    value: Mapping[str, Any],
    *,
    fold_plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    replay_reference_root: Path | None = None,
) -> dict[str, Any]:
    """Internal structural validator with optional exact sidecar-byte replay."""

    receipt = deepcopy(dict(value))
    if set(receipt) != set(_RECEIPT_FIELDS):
        raise ValueError("detector reference phase receipt fields drifted")
    if receipt["schema_version"] != DETECTOR_FOLD_REFERENCE_PHASE_RECEIPT_SCHEMA_V1:
        raise ValueError("detector reference phase schema drifted")
    if receipt["method_id"] != DETECTOR_FOLD_REFERENCE_AUTHORITY_METHOD_ID_V1:
        raise ValueError("detector reference phase method drifted")
    observed_hash = receipt["receipt_sha256"]
    _sha256(observed_hash, "phase receipt")
    if observed_hash != _canonical_sha256(
        {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    ):
        raise ValueError("detector reference phase receipt does not replay")
    plan = validate_tusz_detector_cleanroom_fold_plan_v1(fold_plan)
    authority_registry = validate_detector_fold_reference_authority_registry_v1(
        registry, fold_plan=plan, verify_bound_files=False
    )
    if (
        receipt["registry_id"] != authority_registry["registry_id"]
        or receipt["registry_receipt_sha256"]
        != authority_registry["registry_receipt_sha256"]
        or receipt["fold_plan_binding"] != authority_registry["fold_plan_binding"]
    ):
        raise ValueError("detector reference phase lineage drifted")
    expected_parser_bindings = {
        "acquisition_header_parser_id": authority_registry[
            "acquisition_header_parser_binding"
        ]["parser_id"],
        "acquisition_header_parser_source_sha256": authority_registry[
            "acquisition_header_parser_binding"
        ]["parser_source_sha256"],
        "acquisition_header_policy_receipt_sha256": authority_registry[
            "acquisition_header_parser_binding"
        ]["policy_receipt_sha256"],
        "reference_parser_id": authority_registry["reference_parser_binding"][
            "parser_id"
        ],
        "reference_parser_source_sha256": authority_registry[
            "reference_parser_binding"
        ]["parser_source_sha256"],
        "reference_mapping_id": authority_registry["reference_parser_binding"][
            "mapping_id"
        ],
        "phase_gate_validator_id": authority_registry[
            "phase_gate_validator_binding"
        ]["validator_id"],
        "phase_gate_validator_source_sha256": authority_registry[
            "phase_gate_validator_binding"
        ]["validator_source_sha256"],
    }
    if receipt["parser_bindings"] != expected_parser_bindings:
        raise ValueError("detector reference parser bindings drifted")
    outer_fold_id = receipt["outer_fold_id"]
    phase = receipt["phase"]
    expected_fold_ids = _phase_fold_ids(outer_fold_id, phase)
    if receipt["authorized_fold_ids"] != expected_fold_ids:
        raise PermissionError("detector phase fold authority drifted")
    expected_rows = _phase_rows(plan, outer_fold_id=outer_fold_id, phase=phase)
    expected_roster = _roster_view(expected_rows)
    if receipt["authorized_roster"] != expected_roster:
        raise ValueError("detector phase authorized roster drifted")
    mapping = _phase_mapping(authority_registry, outer_fold_id=outer_fold_id)
    if receipt["forbidden_outer_heldout_roster"] != mapping["outer_heldout_roster"]:
        raise ValueError("outer-held-out forbidden roster drifted")
    gate = receipt["phase_gate"]
    if gate.get("validated_before_first_reference_open") is not True:
        raise PermissionError("reference phase gate was not validated before open")
    if phase == "selection_fit":
        if gate != {
            "gate_type": "registry_and_target_blind_fold_plan_freeze",
            "gate_receipt_sha256": authority_registry["registry_receipt_sha256"],
            "validated_before_first_reference_open": True,
        }:
            raise ValueError("selection-fit gate drifted")
    else:
        validate_detector_reference_phase_gate_proof_v1(
            gate,
            controller_signature_authority=authority_registry[
                "controller_signature_authority"
            ],
            outer_fold_id=outer_fold_id,
            opens_phase=phase,
            authorized_fold_ids=expected_fold_ids,
            authorized_roster=expected_roster,
        )

    output_rows = receipt["records"]
    if type(output_rows) is not list or len(output_rows) != len(expected_rows):
        raise ValueError("detector phase reference row denominator drifted")
    outer_identities = set(
        plan["folds"][outer_fold_id]["held_out_roster"]["analysis_identity_ids"]
    )
    file_hashes: list[str] = []
    for observed, expected in zip(output_rows, expected_rows):
        if type(observed) is not dict or set(observed) != set(_RECORD_FIELDS):
            raise ValueError("detector phase reference row fields drifted")
        identity = observed["analysis_identity_id"]
        if identity != expected["analysis_identity_id"] or identity in outer_identities:
            raise PermissionError("outer or wrong identity entered detector references")
        expected_reference = _safe_reference_relative_path(
            expected["local_edf_path"]
        ).as_posix()
        if (
            observed["source_edf_relative_path"] != expected["local_edf_path"]
            or observed["reference_relative_path"] != expected_reference
            or observed["recording_duration_seconds_fraction"]
            != expected["recording_duration_seconds_fraction"]
        ):
            raise ValueError("detector phase reference path/duration drifted")
        _sha256(observed["reference_file_sha256"], "reference file hash")
        if (
            type(observed["reference_file_bytes"]) is not int
            or observed["reference_file_bytes"] <= 0
        ):
            raise ValueError("reference file byte count is invalid")
        intervals = observed["seizure_intervals"]
        if type(intervals) is not list or observed["event_inventory_sha256"] != (
            _canonical_sha256(intervals)
        ):
            raise ValueError("detector reference event inventory drifted")
        duration = _fraction(
            observed["recording_duration_seconds_fraction"],
            "recording duration",
            positive=True,
        )
        previous_stop = Fraction(-1, 1)
        for event in intervals:
            if type(event) is not dict or set(event) != {
                "start_seconds",
                "stop_seconds",
            }:
                raise ValueError("detector seizure interval fields drifted")
            start = Fraction(str(event["start_seconds"]))
            stop = Fraction(str(event["stop_seconds"]))
            if start < 0 or stop <= start or stop > duration or start < previous_stop:
                raise ValueError("detector seizure intervals are invalid")
            previous_stop = stop
        if observed["selected_term_seiz_event_count"] != len(intervals):
            raise ValueError("TERM,seiz event count drifted")
        file_hashes.append(observed["reference_file_sha256"])
        if replay_reference_root is not None:
            payload = _read_reference_bytes(
                Path(replay_reference_root), PurePosixPath(expected_reference)
            )
            parsed = parse_tusz_term_seiz_reference_bytes(
                payload, duration_seconds=float(duration)
            )
            if (
                parsed.reference_file_sha256 != observed["reference_file_sha256"]
                or len(payload) != observed["reference_file_bytes"]
                or parsed.events() != intervals
                or parsed.ignored_non_term_seiz_row_count
                != observed["ignored_non_term_seiz_row_count"]
            ):
                raise ValueError("detector reference sidecar replay drifted")
    if receipt["reference_file_sha256_roster_sha256"] != _canonical_sha256(file_hashes):
        raise ValueError("reference sidecar byte-hash roster drifted")
    expected_event_inventory_hash = _canonical_sha256(
        [
            {
                "analysis_identity_id": row["analysis_identity_id"],
                "event_inventory_sha256": row["event_inventory_sha256"],
            }
            for row in output_rows
        ]
    )
    if receipt["reference_event_inventory_sha256"] != expected_event_inventory_hash:
        raise ValueError("reference event inventory hash drifted")
    metric = receipt["selection_metric_receipt"]
    if phase == "inner_validation":
        if type(metric) is not dict:
            raise ValueError("inner-validation exact selection metric is missing")
        metric_hash = metric.get("receipt_sha256")
        _sha256(metric_hash, "inner-validation metric receipt")
        if (
            metric_hash
            != _canonical_sha256(
                {key: item for key, item in metric.items() if key != "receipt_sha256"}
            )
            or metric.get("scorer_binding", {}).get("scorer_id")
            != DETECTOR_SELECTION_SCORER_ID_V1
            or metric.get("scorer_binding", {}).get("scorer_version")
            != DETECTOR_SELECTION_SCORER_VERSION_V1
            or metric.get("metric_values_caller_supplied") is not False
            or metric.get("prediction_artifact_inventory_sha256")
            != gate["prediction_artifact_inventory_sha256"]
        ):
            raise ValueError("inner-validation selection metric receipt drifted")
    elif metric is not None:
        raise ValueError(f"{phase} must not contain an inner-selection metric")
    log = receipt["reference_open_log"]
    expected_log_rows = [
        {
            "open_sequence": index,
            "analysis_identity_id": row["analysis_identity_id"],
            "reference_relative_path": row["reference_relative_path"],
            "reference_file_sha256": row["reference_file_sha256"],
            "bytes_read": row["reference_file_bytes"],
        }
        for index, row in enumerate(output_rows)
    ]
    if (
        log.get("rows") != expected_log_rows
        or log.get("reference_files_opened") != len(output_rows)
        or log.get("reference_bytes_read")
        != sum(row["reference_file_bytes"] for row in output_rows)
        or log.get("open_log_sha256") != _canonical_sha256(log.get("rows"))
        or log.get("first_reference_open_after_phase_gate_validation") is not True
        or log.get("first_reference_open_logical_sequence")
        != (1 if phase == "selection_fit" else 4)
        or log.get("pre_reference_release_receipt_sha256")
        != (
            None
            if phase == "selection_fit"
            else gate["pre_reference_release_receipt"]["receipt_sha256"]
        )
    ):
        raise ValueError("detector reference open log drifted")
    for key in (
        "outer_heldout_reference_files_opened",
        "source_dev_reference_files_opened",
        "source_eval_reference_files_opened",
        "private_reference_files_opened",
    ):
        if log.get(key) != 0:
            raise PermissionError(f"forbidden detector reference opened: {key}")
    scope = receipt["scope_receipt"]
    if scope.get("public_TUSZ_source_train_only") is not True:
        raise PermissionError("detector reference split scope drifted")
    for key in (
        "outer_heldout_reference_access_authorized",
        "outer_heldout_reference_access_before_prediction_inventory_freeze",
        "outer_heldout_prediction_or_metric_materialized",
        "source_dev_or_eval_reference_used",
        "private_reference_used",
        "EDF_annotations_or_annotation_channels_used",
        "channel_specific_annotations_or_SOZ_labels_used",
        "Excel_doctor_report_clinical_text_or_behaviour_used",
        "performance_or_clinical_claim_authorized",
    ):
        if scope.get(key) is not False:
            raise PermissionError(f"detector reference scope opened: {key}")
    return receipt


def validate_detector_fold_reference_phase_v1(
    value: Mapping[str, Any],
    *,
    fold_plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    replay_reference_root: Path | None = None,
) -> dict[str, Any]:
    """Validate only the serialized receipt schema and content addresses.

    A raw JSON mapping is evidence to inspect, never a formal reference
    authority.  Actual reference-byte replay and process-local authority
    issuance are deliberately restricted to the materializer and
    :func:`authorize_detector_fold_reference_phase_receipt_v1`.  Retaining the
    former keyword as a fail-closed seam prevents an old caller from silently
    mistaking sidecar replay plus a caller-owned proof for opaque authority.
    """

    if replay_reference_root is not None:
        raise PermissionError(
            "raw detector phase receipt validation is schema-only; use the "
            "actual-byte replay authority issuer"
        )
    return _validate_detector_fold_reference_phase_receipt_v1(
        value,
        fold_plan=fold_plan,
        registry=registry,
        replay_reference_root=None,
    )


def authorize_detector_fold_reference_phase_receipt_v1(
    value: Mapping[str, Any],
    *,
    fold_plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    replay_reference_root: Path,
    controller_bundle_root: Path | None = None,
    controller_ledger_relative_path: str | None = None,
    prior_selection_fit_authority: object | None = None,
    prior_inner_validation_authority: object | None = None,
    selected_epoch_metric_receipt: Mapping[str, Any] | None = None,
) -> ValidatedDetectorFoldReferencePhaseAuthorityV1:
    """Replay a serialized receipt and issue process-local formal authority.

    For non-selection phases checkpoint/prediction replay occurs before the
    first sidecar replay read in this call.  Selection-fit requires exact
    reference-byte replay.  No path accepts a caller digest as authority.
    """

    plan = validate_tusz_detector_cleanroom_fold_plan_v1(fold_plan)
    authority_registry = validate_detector_fold_reference_authority_registry_v1(
        registry, fold_plan=plan, verify_bound_files=False
    )
    structural = validate_detector_fold_reference_phase_v1(
        value,
        fold_plan=plan,
        registry=authority_registry,
        replay_reference_root=None,
    )
    phase = structural["phase"]
    outer_fold_id = structural["outer_fold_id"]
    if phase == "selection_fit":
        if any(
            item is not None
            for item in (
                controller_bundle_root,
                controller_ledger_relative_path,
                prior_selection_fit_authority,
                prior_inner_validation_authority,
                selected_epoch_metric_receipt,
            )
        ):
            raise PermissionError("selection-fit replay issuer rejects controller/post-reference inputs")
        replayed = _validate_detector_fold_reference_phase_receipt_v1(
            structural,
            fold_plan=plan,
            registry=authority_registry,
            replay_reference_root=Path(replay_reference_root),
        )
        return _issue_opaque_phase_authority_v1(replayed)

    if controller_bundle_root is None or controller_ledger_relative_path is None:
        raise PermissionError("non-selection replay issuer requires the controller bundle and ledger")
    selection = require_validated_detector_fold_reference_phase_authority_v1(
        prior_selection_fit_authority
    )
    inner = None
    if phase == "final_refit":
        inner = require_validated_detector_fold_reference_phase_authority_v1(
            prior_inner_validation_authority
        )
    gate_replay = _replay_nonselection_phase_gate(
        plan=plan,
        registry=authority_registry,
        reference_root=Path(replay_reference_root),
        outer_fold_id=outer_fold_id,
        phase=phase,
        controller_bundle_root=Path(controller_bundle_root),
        controller_ledger_relative_path=controller_ledger_relative_path,
        prior_selection_fit_phase_receipt=selection,
        prior_inner_validation_phase_receipt=inner,
        selected_epoch_metric_receipt=selected_epoch_metric_receipt,
    )
    if gate_replay.proof != structural["phase_gate"]:
        raise PermissionError("serialized phase gate differs from actual controller/artifact replay")
    replayed = _validate_detector_fold_reference_phase_receipt_v1(
        structural,
        fold_plan=plan,
        registry=authority_registry,
        replay_reference_root=Path(replay_reference_root),
    )
    if phase == "inner_validation":
        recomputed_metric = build_detector_selection_metric_receipt_v1(
            prediction_artifacts=gate_replay.prediction_artifacts,
            checkpoint_hash_by_epoch=gate_replay.checkpoint_hash_by_epoch,
            prediction_artifact_inventory_sha256=gate_replay.proof[
                "prediction_artifact_inventory_sha256"
            ],
            reference_records=replayed["records"],
        )
        if recomputed_metric != replayed["selection_metric_receipt"]:
            raise PermissionError("serialized inner metric differs from actual scorer replay")
    return _issue_opaque_phase_authority_v1(replayed)


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "DETECTOR_FOLD_REFERENCE_AUTHORITY_METHOD_ID_V1",
    "DETECTOR_FOLD_REFERENCE_AUTHORITY_REGISTRY_SCHEMA_V1",
    "DETECTOR_FOLD_REFERENCE_PHASE_RECEIPT_SCHEMA_V1",
    "REFERENCE_PHASES_V1",
    "REFERENCE_SIDECAR_MAPPING_ID_V1",
    "ValidatedDetectorFoldReferencePhaseAuthorityV1",
    "authorize_detector_fold_reference_phase_receipt_v1",
    "build_detector_fold_reference_authority_registry_v1",
    "detector_reference_authority_source_sha256_v1",
    "detector_reference_parser_source_sha256_v1",
    "load_default_detector_fold_reference_authority_registry_v1",
    "materialize_detector_fold_reference_phase_v1",
    "require_validated_detector_fold_reference_phase_authority_v1",
    "validate_detector_fold_reference_authority_registry_v1",
    "validate_detector_fold_reference_phase_v1",
]
