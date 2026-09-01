"""Engineering-only composer/replay closure addendum for Findings v1-core.

The frozen Findings v1-core release profile intentionally retains ten
``required_gap`` term queries.  Two subsequently implemented source modules
now expose deterministic composer/replay interfaces for those ten query IDs:
six waveform/rhythm queries and four evolution/recovery queries.  This
addendum content-addresses that narrow engineering fact without modifying or
promoting the frozen base profile.

The receipt proves only that 10/10 named query IDs have a source-bound,
fail-closed composer and independently sourced exact-replay software
interface.  Bridge files and their direct project dependencies are byte-hash
bound; the transitive dependency tree is not.  It explicitly does *not*
prove that a disk producer/runner exists, that the interfaces work on the
target domain, that any capability or model-performance threshold is met,
that positive clinical terms or negative/absence claims are qualified, that
the ledgers enter the SOZ evidence graph, or that any result may enter report
text.  The base profile and its empty report allowlist remain authoritative.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Final, Mapping

from . import event_evolution_recovery_query_bridge_v1 as evolution_bridge
from . import event_waveform_rhythm_query_bridge_v1 as waveform_bridge
from .findings_v1_core_release_profile import (
    DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_PATH,
    DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_SHA256,
    load_findings_v1_core_release_profile,
    materialize_findings_v1_core_readiness_receipt,
)


FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_v1_composer_closure_addendum_v1"
)
FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_METHOD_ID: Final[str] = (
    "CLINICAL-EEG-FINDINGS-V1-COMPOSER-REPLAY-CLOSURE-AUDIT-V1"
)
FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_ID: Final[str] = (
    "CLINICAL-EEG-FINDINGS-V1-COMPOSER-CLOSURE-ADDENDUM-V1"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_PATH: Final[Path] = (
    _ROOT / "configs" / "clinical_eeg_findings_v1_composer_closure_addendum.json"
)
DEFAULT_FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_SHA256: Final[str] = (
    "8d1e49d9042663c6adaa6d8b7c7b1f43d5e6bcfc150a0f8c6b5acadbd7909314"
)

_BASE_PROFILE_FILE_SHA256: Final[str] = (
    "bbf43e69c71b8c08dd18ed56f4ed2ae775abdfd793453864a77da98644cf9793"
)
_WAVEFORM_BRIDGE_RELATIVE_PATH: Final[str] = (
    "src/clinical_eeg_long_recording/event_waveform_rhythm_query_bridge_v1.py"
)
_WAVEFORM_BRIDGE_FILE_SHA256: Final[str] = (
    "9dcb9fbfc162c4284bb7b5d322517f9853c1a2ac7493afa0cf51fc5b4a72d9cf"
)
_EVOLUTION_BRIDGE_RELATIVE_PATH: Final[str] = (
    "src/clinical_eeg_long_recording/event_evolution_recovery_query_bridge_v1.py"
)
_EVOLUTION_BRIDGE_FILE_SHA256: Final[str] = (
    "504c2c5def4c4e21c95cf97b43dd88428ee6c2a55bbe3316ad289758f6715ed9"
)

_DIRECT_DEPENDENCY_FILE_SHA256S: Final[dict[str, str]] = {
    (
        "src/clinical_eeg_long_recording/"
        "deterministic_event_morphology_primitives_v1.py"
    ): "a956577e87186b6c7b5c0579099d518fe1cb3ce43698623502517421fc879c04",
    (
        "src/clinical_eeg_long_recording/deterministic_periodicity_candidate.py"
    ): "47e3fd081669437e61944cff3155913fd516a3f97017906c1e91a1f8053c4de9",
    (
        "src/clinical_eeg_long_recording/acns_frequency_evolution_candidate.py"
    ): "6a6c6cbc6671f4ddf7167e70c90d6467af23e20ca51955d2d26814faec48b073",
    (
        "src/clinical_eeg_long_recording/ba_ieg_multireference_field.py"
    ): "2d436718258edebb4e6bae7db2d1c0d3a1dece198a73262aa48682873c70b06c",
    (
        "src/clinical_eeg_long_recording/event_baseline_context_comparability.py"
    ): "a28a79d1fdde2680fea9b7989f45f40766bca383242f5b6a60da410a1c61db29",
    (
        "src/clinical_eeg_long_recording/event_findings_v3_validation.py"
    ): "32fa53fd1c99dee9febe08964b546264888a6fa6b2e08eb80fe46651ac5ce1a9",
}
_WAVEFORM_DIRECT_DEPENDENCY_PATHS: Final[tuple[str, ...]] = (
    "src/clinical_eeg_long_recording/deterministic_event_morphology_primitives_v1.py",
    "src/clinical_eeg_long_recording/deterministic_periodicity_candidate.py",
)
_EVOLUTION_DIRECT_DEPENDENCY_PATHS: Final[tuple[str, ...]] = (
    "src/clinical_eeg_long_recording/acns_frequency_evolution_candidate.py",
    "src/clinical_eeg_long_recording/ba_ieg_multireference_field.py",
    "src/clinical_eeg_long_recording/deterministic_event_morphology_primitives_v1.py",
    "src/clinical_eeg_long_recording/event_baseline_context_comparability.py",
    "src/clinical_eeg_long_recording/event_findings_v3_validation.py",
)

_WAVEFORM_QUERY_IDS: Final[tuple[str, ...]] = (
    "TQ-EVENT-AMPLITUDE-COURSE",
    "TQ-EVENT-RHYTHMICITY-COURSE",
    "TQ-PERIODIC-ELEMENT-INSTANCE",
    "TQ-PHYSICAL-AMPLITUDE-PROFILE",
    "TQ-RHYTHMIC-RUN-INSTANCE",
    "TQ-SHARP-CONTOURED-ICTAL-COMPONENT-INSTANCE",
)
_EVOLUTION_QUERY_COMPOSERS: Final[dict[str, str]] = {
    "TQ-EVOLUTION-FREQUENCY": "compose_frequency_evolution_query_ledger_v1",
    "TQ-EVOLUTION-LOCATION": "compose_location_evolution_query_ledger_v1",
    "TQ-EVOLUTION-MORPHOLOGY": "compose_morphology_evolution_query_ledger_v1",
    "TQ-POST-EVENT-RETURN-COMPARABLE-BACKGROUND": (
        "compose_return_to_comparable_background_query_ledger_v1"
    ),
}
_EVOLUTION_QUERY_REPLAYS: Final[dict[str, str]] = {
    "TQ-EVOLUTION-FREQUENCY": "replay_frequency_evolution_query_ledger_v1",
    "TQ-EVOLUTION-LOCATION": "replay_location_evolution_query_ledger_v1",
    "TQ-EVOLUTION-MORPHOLOGY": "replay_morphology_evolution_query_ledger_v1",
    "TQ-POST-EVENT-RETURN-COMPARABLE-BACKGROUND": (
        "replay_return_to_comparable_background_query_ledger_v1"
    ),
}
_REQUIRED_QUERY_IDS: Final[tuple[str, ...]] = tuple(
    sorted((*_WAVEFORM_QUERY_IDS, *_EVOLUTION_QUERY_COMPOSERS))
)

_UNCLOSED_BOUNDARIES: Final[dict[str, bool | str | list[str]]] = {
    "disk_producer_and_runner_closed": False,
    "real_target_domain_capability_established": False,
    "real_target_domain_performance_established": False,
    "target_domain_term_qualification_closed": False,
    "negative_or_absence_qualification_closed": False,
    "soz_evidence_graph_integration_closed": False,
    "report_allowlist_closed": False,
    "report_eligible_automated_allowlist": [],
    "base_profile_promotion_authorized": False,
    "transitive_dependency_tree_hash_closed": False,
    "golden_target_domain_replay_receipts_closed": False,
    "claim_ceiling": "composer_and_replay_software_interface_only",
}

_TOP_KEYS = frozenset(
    {
        "schema_version",
        "method_id",
        "addendum_id",
        "receipt_id",
        "base_profile_binding",
        "bridge_bindings",
        "query_interface_rows",
        "closure_summary",
        "unclosed_boundaries",
        "semantic_guard",
        "receipt_sha256",
    }
)
_BRIDGE_KEYS = frozenset(
    {
        "bridge_family",
        "module_relative_path",
        "module_file_sha256",
        "schema_version",
        "method_id",
        "method_id_source_constant",
        "query_ids",
        "composer_symbols",
        "replay_symbols",
        "validator_symbols",
        "replay_interface_kind",
        "direct_dependency_bindings",
    }
)
_DEPENDENCY_KEYS = frozenset(
    {
        "module_relative_path",
        "module_file_sha256",
    }
)
_ROW_KEYS = frozenset(
    {
        "term_query_id",
        "bridge_family",
        "base_release_disposition",
        "base_blocking_closure_layers",
        "composer_symbol",
        "replay_symbols",
        "replay_interface_kind",
        "composer_interface_implemented",
        "replay_interface_implemented",
        "interface_claim",
        "base_required_gap_retained",
        "disk_producer_and_runner_status",
        "real_capability_or_performance_claimed",
        "target_domain_term_qualification_closed",
        "negative_or_absence_qualification_closed",
        "soz_evidence_graph_integration_closed",
        "report_promotion_authorized",
    }
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_keys(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} fields drifted")
    return value


def _bridge_file_binding() -> list[dict[str, Any]]:
    waveform_path = _ROOT / _WAVEFORM_BRIDGE_RELATIVE_PATH
    evolution_path = _ROOT / _EVOLUTION_BRIDGE_RELATIVE_PATH
    observed_waveform_hash = _file_sha256(waveform_path)
    observed_evolution_hash = _file_sha256(evolution_path)
    if observed_waveform_hash != _WAVEFORM_BRIDGE_FILE_SHA256:
        raise ValueError("waveform/rhythm bridge file content drifted")
    if observed_evolution_hash != _EVOLUTION_BRIDGE_FILE_SHA256:
        raise ValueError("evolution/recovery bridge file content drifted")
    if waveform_bridge.EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_SCHEMA_VERSION != (
        "clinical_eeg_event_waveform_rhythm_query_bridge_v1"
    ) or waveform_bridge.EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_METHOD_ID != (
        "EVENT-WAVEFORM-RHYTHM-QUERY-BRIDGE-V1"
    ):
        raise ValueError("waveform/rhythm bridge schema/method identity drifted")
    if evolution_bridge.EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_SCHEMA_VERSION != (
        "clinical_eeg_event_evolution_recovery_query_bridge_v1"
    ) or evolution_bridge.EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_ID != (
        "EEG-ONLY-EVENT-EVOLUTION-RECOVERY-QUERY-BRIDGE-V1"
    ):
        raise ValueError("evolution/recovery bridge schema/method identity drifted")
    if set(waveform_bridge._QUERY_SPECS) != set(_WAVEFORM_QUERY_IDS):
        raise ValueError("waveform/rhythm bridge query roster drifted")
    if set(evolution_bridge._QUERY_AXIS) != set(_EVOLUTION_QUERY_COMPOSERS):
        raise ValueError("evolution/recovery bridge query roster drifted")

    def dependency_bindings(paths: tuple[str, ...]) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for relative_path in paths:
            expected_hash = _DIRECT_DEPENDENCY_FILE_SHA256S[relative_path]
            observed_hash = _file_sha256(_ROOT / relative_path)
            if observed_hash != expected_hash:
                raise ValueError(
                    f"direct bridge dependency {relative_path} content drifted"
                )
            result.append(
                {
                    "module_relative_path": relative_path,
                    "module_file_sha256": observed_hash,
                }
            )
        return result

    waveform_symbols = (
        "materialize_event_waveform_rhythm_query_bridge_v1",
    )
    waveform_replay = ("replay_event_waveform_rhythm_query_bridge_v1",)
    waveform_validators = ("validate_event_waveform_rhythm_query_bridge_v1",)
    evolution_symbols = tuple(
        _EVOLUTION_QUERY_COMPOSERS[query_id]
        for query_id in sorted(_EVOLUTION_QUERY_COMPOSERS)
    )
    evolution_replay = tuple(
        _EVOLUTION_QUERY_REPLAYS[query_id]
        for query_id in sorted(_EVOLUTION_QUERY_REPLAYS)
    )
    evolution_validators = (
        "validate_event_evolution_recovery_query_ledger_v1",
    )
    for module, names, family in (
        (
            waveform_bridge,
            (*waveform_symbols, *waveform_replay, *waveform_validators),
            "waveform_rhythm_6",
        ),
        (
            evolution_bridge,
            (*evolution_symbols, *evolution_replay, *evolution_validators),
            "evolution_recovery_4",
        ),
    ):
        for name in names:
            function = getattr(module, name, None)
            if not callable(function) or function.__module__ != module.__name__:
                raise ValueError(f"{family} interface symbol {name} is not source-bound")

    return [
        {
            "bridge_family": "waveform_rhythm_6",
            "module_relative_path": _WAVEFORM_BRIDGE_RELATIVE_PATH,
            "module_file_sha256": observed_waveform_hash,
            "schema_version": (
                waveform_bridge.EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_SCHEMA_VERSION
            ),
            "method_id": waveform_bridge.EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_METHOD_ID,
            "method_id_source_constant": (
                "EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_METHOD_ID"
            ),
            "query_ids": list(_WAVEFORM_QUERY_IDS),
            "composer_symbols": list(waveform_symbols),
            "replay_symbols": list(waveform_replay),
            "validator_symbols": list(waveform_validators),
            "replay_interface_kind": "explicit_materialize_replay_and_validator",
            "direct_dependency_bindings": dependency_bindings(
                _WAVEFORM_DIRECT_DEPENDENCY_PATHS
            ),
        },
        {
            "bridge_family": "evolution_recovery_4",
            "module_relative_path": _EVOLUTION_BRIDGE_RELATIVE_PATH,
            "module_file_sha256": observed_evolution_hash,
            "schema_version": (
                evolution_bridge.EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_SCHEMA_VERSION
            ),
            "method_id": evolution_bridge.EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_ID,
            "method_id_source_constant": "EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_ID",
            "query_ids": sorted(_EVOLUTION_QUERY_COMPOSERS),
            "composer_symbols": list(evolution_symbols),
            "replay_symbols": list(evolution_replay),
            "validator_symbols": list(evolution_validators),
            "replay_interface_kind": (
                "explicit_independent_source_recomposition_exact_equality_and_validator"
            ),
            "direct_dependency_bindings": dependency_bindings(
                _EVOLUTION_DIRECT_DEPENDENCY_PATHS
            ),
        },
    ]


def _interface_rows(
    *, gap_specs: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    waveform_replay = ["replay_event_waveform_rhythm_query_bridge_v1"]
    for query_id in _REQUIRED_QUERY_IDS:
        if query_id in _WAVEFORM_QUERY_IDS:
            family = "waveform_rhythm_6"
            composer = "materialize_event_waveform_rhythm_query_bridge_v1"
            replay = waveform_replay
            replay_kind = "explicit_materialize_replay_and_validator"
        else:
            family = "evolution_recovery_4"
            composer = _EVOLUTION_QUERY_COMPOSERS[query_id]
            replay = [_EVOLUTION_QUERY_REPLAYS[query_id]]
            replay_kind = (
                "explicit_independent_source_recomposition_exact_equality_and_validator"
            )
        rows.append(
            {
                "term_query_id": query_id,
                "bridge_family": family,
                "base_release_disposition": "required_gap",
                "base_blocking_closure_layers": list(
                    gap_specs[query_id]["blocking_closure_layers"]
                ),
                "composer_symbol": composer,
                "replay_symbols": list(replay),
                "replay_interface_kind": replay_kind,
                "composer_interface_implemented": True,
                "replay_interface_implemented": True,
                "interface_claim": "composer_and_replay_interface_implemented",
                "base_required_gap_retained": True,
                "disk_producer_and_runner_status": "not_closed_not_evaluable",
                "real_capability_or_performance_claimed": False,
                "target_domain_term_qualification_closed": False,
                "negative_or_absence_qualification_closed": False,
                "soz_evidence_graph_integration_closed": False,
                "report_promotion_authorized": False,
            }
        )
    return rows


def materialize_findings_v1_composer_closure_addendum_v1() -> dict[str, Any]:
    """Build the non-patient, engineering-interface-only closure receipt."""

    profile_file_hash_before = _file_sha256(
        DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_PATH
    )
    if profile_file_hash_before != _BASE_PROFILE_FILE_SHA256:
        raise ValueError("frozen Findings base profile file content drifted")
    profile = load_findings_v1_core_release_profile()
    if profile["profile_sha256"] != DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_SHA256:
        raise ValueError("frozen Findings base profile semantic hash drifted")
    if profile["profile_status"] != "frozen_public_synthetic_shadow":
        raise ValueError("frozen Findings base profile status was promoted")
    if profile["qualification_boundary"]["report_eligible_automated_allowlist"] != []:
        raise ValueError("base profile report allowlist is no longer empty")

    profile_gap_ids = tuple(
        sorted(
            str(row["term_query_id"])
            for row in profile["required_query_gap_specs"]
        )
    )
    partition_gap_ids = tuple(
        sorted(
            str(item)
            for item in profile["partition_policy"]["term_query_partitions"][
                "required_gap"
            ]
        )
    )
    if profile_gap_ids != _REQUIRED_QUERY_IDS or partition_gap_ids != (
        _REQUIRED_QUERY_IDS
    ):
        raise ValueError("base profile ten-query required-gap denominator drifted")
    gap_specs = {
        str(row["term_query_id"]): row
        for row in profile["required_query_gap_specs"]
    }
    bridge_bindings = _bridge_file_binding()
    rows = _interface_rows(gap_specs=gap_specs)
    composer_count = sum(
        row["composer_interface_implemented"] is True for row in rows
    )
    replay_count = sum(row["replay_interface_implemented"] is True for row in rows)
    readiness = materialize_findings_v1_core_readiness_receipt(profile=profile)
    if readiness["release_ready"] is not False or readiness[
        "readiness_status"
    ] != "not_ready_required_core_gaps":
        raise ValueError("base Findings readiness was unexpectedly promoted")
    if readiness["readiness_summary"]["term_queries"][
        "required_gap_ids"
    ] != list(_REQUIRED_QUERY_IDS):
        raise ValueError("base readiness no longer retains all ten query gaps")

    seed = _canonical_sha256(
        {
            "base_profile_sha256": profile["profile_sha256"],
            "base_profile_file_sha256": profile_file_hash_before,
            "bridge_bindings": bridge_bindings,
        }
    )[:24]
    body: dict[str, Any] = {
        "schema_version": FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_SCHEMA_VERSION,
        "method_id": FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_METHOD_ID,
        "addendum_id": FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_ID,
        "receipt_id": "FINDINGS-V1-COMPOSER-CLOSURE-" + seed,
        "base_profile_binding": {
            "profile_id": profile["profile_id"],
            "profile_status": profile["profile_status"],
            "profile_sha256": profile["profile_sha256"],
            "profile_file_sha256": profile_file_hash_before,
            "base_readiness_receipt_sha256": readiness["receipt_sha256"],
            "base_readiness_status": readiness["readiness_status"],
            "base_release_ready": False,
            "base_required_gap_query_ids": list(_REQUIRED_QUERY_IDS),
            "base_profile_modified_or_promoted": False,
        },
        "bridge_bindings": bridge_bindings,
        "query_interface_rows": rows,
        "closure_summary": {
            "required_query_count": len(rows),
            "composer_interface_implemented_count": composer_count,
            "replay_interface_implemented_count": replay_count,
            "composer_replay_interface_coverage": (
                f"{min(composer_count, replay_count)}/{len(rows)}"
            ),
            "claim_scope": "composer_and_replay_software_interface_only",
            "base_required_query_gap_count_after_addendum": 10,
            "base_profile_promoted": False,
        },
        "unclosed_boundaries": deepcopy(_UNCLOSED_BOUNDARIES),
        "semantic_guard": {
            "patient_finding": False,
            "clinical_term_qualification": False,
            "negative_or_absence_claim": False,
            "soz_evidence": False,
            "report_text": False,
            "real_capability_result": False,
            "model_performance_result": False,
            "software_test_pass_rate_is_model_performance": False,
        },
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    if _file_sha256(DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_PATH) != (
        profile_file_hash_before
    ):
        raise RuntimeError("base Findings profile changed while building addendum")
    return _validate_receipt_shape(body)


def _validate_receipt_shape(value: object) -> dict[str, Any]:
    top = _exact_keys(value, _TOP_KEYS, "composer closure addendum")
    data = deepcopy(dict(top))
    if data["schema_version"] != (
        FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_SCHEMA_VERSION
    ) or data["method_id"] != FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_METHOD_ID or (
        data["addendum_id"] != FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_ID
    ):
        raise ValueError("composer closure addendum identity drifted")
    receipt = _sha(data["receipt_sha256"], "addendum receipt")
    body = deepcopy(data)
    body.pop("receipt_sha256")
    if _canonical_sha256(body) != receipt:
        raise ValueError("composer closure addendum receipt does not replay")
    bindings = data["bridge_bindings"]
    if not isinstance(bindings, list) or len(bindings) != 2:
        raise ValueError("composer closure addendum needs exactly two bridges")
    for binding in bindings:
        row = _exact_keys(binding, _BRIDGE_KEYS, "bridge binding")
        _sha(row["module_file_sha256"], "bridge module file hash")
        dependencies = row["direct_dependency_bindings"]
        if not isinstance(dependencies, list) or not dependencies:
            raise ValueError("bridge binding needs direct dependency hashes")
        dependency_paths: list[str] = []
        for dependency in dependencies:
            dependency_row = _exact_keys(
                dependency, _DEPENDENCY_KEYS, "direct dependency binding"
            )
            relative_path = str(dependency_row["module_relative_path"])
            dependency_paths.append(relative_path)
            _sha(
                dependency_row["module_file_sha256"],
                "direct dependency module file hash",
            )
        if dependency_paths != sorted(set(dependency_paths)):
            raise ValueError("direct dependency bindings must be sorted and unique")
    rows = data["query_interface_rows"]
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("composer closure addendum needs exactly ten query rows")
    ids: list[str] = []
    for row_value in rows:
        row = _exact_keys(row_value, _ROW_KEYS, "query interface row")
        query_id = str(row["term_query_id"])
        if query_id not in _REQUIRED_QUERY_IDS:
            raise ValueError("composer closure query roster drifted")
        ids.append(query_id)
        expected_replay = (
            ["replay_event_waveform_rhythm_query_bridge_v1"]
            if query_id in _WAVEFORM_QUERY_IDS
            else [_EVOLUTION_QUERY_REPLAYS[query_id]]
        )
        if (
            row["base_release_disposition"] != "required_gap"
            or row["composer_interface_implemented"] is not True
            or row["replay_interface_implemented"] is not True
            or row["interface_claim"]
            != "composer_and_replay_interface_implemented"
            or row["base_required_gap_retained"] is not True
            or row["disk_producer_and_runner_status"]
            != "not_closed_not_evaluable"
            or row["real_capability_or_performance_claimed"] is not False
            or row["target_domain_term_qualification_closed"] is not False
            or row["negative_or_absence_qualification_closed"] is not False
            or row["soz_evidence_graph_integration_closed"] is not False
            or row["report_promotion_authorized"] is not False
            or row["replay_symbols"] != expected_replay
        ):
            raise ValueError("query interface claim exceeded engineering closure")
    if tuple(ids) != _REQUIRED_QUERY_IDS:
        raise ValueError("composer closure query roster/order drifted")
    if data["unclosed_boundaries"] != _UNCLOSED_BOUNDARIES:
        raise ValueError("composer closure unclosed boundaries drifted")
    summary = data["closure_summary"]
    if summary != {
        "required_query_count": 10,
        "composer_interface_implemented_count": 10,
        "replay_interface_implemented_count": 10,
        "composer_replay_interface_coverage": "10/10",
        "claim_scope": "composer_and_replay_software_interface_only",
        "base_required_query_gap_count_after_addendum": 10,
        "base_profile_promoted": False,
    }:
        raise ValueError("composer closure summary overclaimed or drifted")
    base = data["base_profile_binding"]
    if (
        base.get("profile_sha256")
        != DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_SHA256
        or base.get("profile_file_sha256") != _BASE_PROFILE_FILE_SHA256
        or base.get("base_release_ready") is not False
        or base.get("base_readiness_status") != "not_ready_required_core_gaps"
        or base.get("base_required_gap_query_ids") != list(_REQUIRED_QUERY_IDS)
        or base.get("base_profile_modified_or_promoted") is not False
    ):
        raise ValueError("base Findings profile non-promotion binding drifted")
    guard = data["semantic_guard"]
    if not isinstance(guard, Mapping) or any(
        value is not False for value in guard.values()
    ):
        raise ValueError("composer closure semantic guard must remain all false")
    return data


def validate_findings_v1_composer_closure_addendum_v1(
    value: object,
    *,
    trusted_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay source files/profile and reject stale or overclaiming receipts."""

    candidate = _validate_receipt_shape(value)
    if trusted_receipt_sha256 is not None and candidate["receipt_sha256"] != _sha(
        trusted_receipt_sha256, "trusted addendum receipt"
    ):
        raise ValueError("composer closure addendum is not host trusted")
    expected = materialize_findings_v1_composer_closure_addendum_v1()
    if candidate != expected:
        raise ValueError("composer closure addendum does not replay from sources")
    return candidate


def load_findings_v1_composer_closure_addendum_v1(
    path: str | Path = DEFAULT_FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_PATH,
    *,
    trusted_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if trusted_receipt_sha256 is None:
        if resolved != DEFAULT_FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_PATH.resolve():
            raise ValueError("non-default composer addendum needs a trust anchor")
        trusted_receipt_sha256 = (
            DEFAULT_FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_SHA256
        )
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_findings_v1_composer_closure_addendum_v1(
        payload, trusted_receipt_sha256=trusted_receipt_sha256
    )


def write_findings_v1_composer_closure_addendum_v1(
    path: str | Path = DEFAULT_FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_PATH,
) -> Path:
    payload = materialize_findings_v1_composer_closure_addendum_v1()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise FileExistsError("refusing to overwrite a different composer addendum")
        return destination
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = [
    "DEFAULT_FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_PATH",
    "DEFAULT_FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_SHA256",
    "FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_ID",
    "FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_METHOD_ID",
    "FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_SCHEMA_VERSION",
    "load_findings_v1_composer_closure_addendum_v1",
    "materialize_findings_v1_composer_closure_addendum_v1",
    "validate_findings_v1_composer_closure_addendum_v1",
    "write_findings_v1_composer_closure_addendum_v1",
]
