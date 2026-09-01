"""Host-replayed raw-dependency adapter for P0 inner-router candidates.

Adapter v2 deliberately reuses adapter v1's channel-neutral physical cells,
tree construction and deterministic scores.  It changes only provenance:
every synthetic v1 transitive token hash is replaced by the real dependency
row ID/SHA from :mod:`ba_ieg_p0_raw_sample_dependency_v1`.

The public validator requires projection v2 plus host canonical/view roots and
replays the complete artifact.  A candidate remains an interval/scale/
permission cell containing all eligible channels and references; this adapter
never selects either of them.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Final, Mapping, Sequence

import torch

from .ba_ieg_event_model_input_projection_v2 import (
    BAIEGEventModelInputProjectionV2,
    validate_ba_ieg_event_model_input_projection_v2,
)
from .ba_ieg_inner_ragged_router_v1 import (
    BAIEGInnerRaggedRouterPolicyV1,
    materialize_ba_ieg_inner_ragged_router_v1,
)
from . import ba_ieg_p0_inner_router_adapter_v1 as _adapter_v1
from .ba_ieg_training_contract import BAIEGP0TokenizationPolicy
from .canonical_signal_views import validate_canonical_signal_receipt


BA_IEG_P0_INNER_ROUTER_ADAPTER_SCHEMA_VERSION_V2: Final[
    str
] = "clinical_eeg_ba_ieg_p0_inner_router_candidate_adapter_v2"
BA_IEG_P0_INNER_ROUTER_ADAPTER_METHOD_ID_V2: Final[
    str
] = "ba_ieg_p0_host_raw_dependency_channel_neutral_candidate_adapter_v2"

_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_SCOPE_RECEIPT_V2: Final[dict[str, bool]] = {
    "eeg_signal_rows_used": True,
    "target_free_projection_v2_required": True,
    "deterministic_target_sidecar_used_for_features": False,
    "deterministic_target_sidecar_integrity_bound": True,
    "host_replayed_raw_dependency_sidecar_used": True,
    "synthetic_transitive_token_dependency_hash_used": False,
    "raw_dependency_closure_asserted_only_if_all_active_tokens_closed": True,
    "channel_or_reference_subset_selected": False,
    "new_physical_eeg_support_acquired": False,
    "native_onset_authority_granted": False,
    "causal_lane_requires_positive_clinical_onset_authorization": False,
    "public_or_private_label_used": False,
    "edf_annotation_used": False,
    "spreadsheet_used": False,
    "clinical_text_used": False,
    "clinical_finding_or_soz_claim_authorized": False,
}

_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "method_id",
        "status",
        "reason_code",
        "reason_detail",
        "source_binding",
        "policy_binding",
        "outer_support_union",
        "candidate_cells",
        "cell_source_ledgers",
        "permission_lanes",
        "deterministic_score_policy_receipt",
        "lane_score_execution_receipts",
        "diagnostics",
        "scope_receipt",
        "artifact_sha256",
    }
)
_SOURCE_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "event_id",
        "recording_id",
        "patient_uid",
        "model_split",
        "p0_materialization_receipt_sha256",
        "event_input_receipt_sha256",
        "projection_v2_receipt_sha256",
        "deterministic_target_sidecar_receipt_sha256",
        "deterministic_target_receipt_sha256",
        "raw_dependency_sidecar_id",
        "raw_dependency_sidecar_sha256",
        "raw_dependency_roster_sha256",
        "canonical_signal_sha256",
        "canonical_receipt_sha256",
        "outer_support_receipt_sha256",
        "target_free_model_input_used",
        "host_replayed_raw_dependency_used",
    }
)
_LEDGER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "cell_id",
        "nominal_interval_seconds",
        "shared_actual_interval_seconds",
        "expected_eligible_unit_count",
        "present_source_row_count",
        "active_source_row_count",
        "all_eligible_rows_grouped_before_qc",
        "source_rows",
        "active_raw_dependency_ids",
        "active_raw_dependency_sha256s",
        "active_raw_dependency_closure_proven",
        "ledger_sha256",
    }
)
_SOURCE_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source_token_index",
        "view_index",
        "view_id",
        "view_receipt_sha256",
        "transform_sha256",
        "temporal_evidence_sha256",
        "reference_family",
        "physical_reference_row_sha256",
        "reference_transform_fingerprint_sha256",
        "unit_index",
        "unit_id",
        "unit_source_id",
        "unit_type",
        "actual_interval_seconds",
        "signal_eligible",
        "future_sample_access",
        "onset_evidence_authorized",
        "canonical_signal_sha256",
        "canonical_receipt_sha256",
        "token_values_sha256",
        "token_feature_mask_sha256",
        "facet_fingerprint_sha256",
        "raw_dependency_id",
        "raw_dependency_sha256",
        "raw_dependency_closure_proven",
    }
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in _SHA256_CHARACTERS for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


def _projection_source_binding(
    projection: BAIEGEventModelInputProjectionV2,
    *,
    canonical: Mapping[str, Any],
    outer_support_receipt_sha256: str,
) -> dict[str, Any]:
    event = projection.model_input_event
    raw = projection.raw_sample_dependency_sidecar
    return {
        "event_id": event.event_id,
        "recording_id": event.recording_id,
        "patient_uid": event.patient_uid,
        "model_split": event.model_split,
        "p0_materialization_receipt_sha256": (
            projection.source_p0_materialization_receipt_sha256
        ),
        "event_input_receipt_sha256": event.input_receipt_sha256,
        "projection_v2_receipt_sha256": projection.receipt_sha256,
        "deterministic_target_sidecar_receipt_sha256": (
            projection.deterministic_target_sidecar.receipt_sha256
        ),
        "deterministic_target_receipt_sha256": (
            projection.deterministic_target_sidecar.target_receipt_sha256
        ),
        "raw_dependency_sidecar_id": raw["sidecar_id"],
        "raw_dependency_sidecar_sha256": raw["sidecar_sha256"],
        "raw_dependency_roster_sha256": raw["dependency_roster_sha256"],
        "canonical_signal_sha256": canonical["source_signal_sha256"],
        "canonical_receipt_sha256": canonical["receipt_sha256"],
        "outer_support_receipt_sha256": outer_support_receipt_sha256,
        "target_free_model_input_used": True,
        "host_replayed_raw_dependency_used": True,
    }


def _replace_synthetic_dependencies(
    legacy: Mapping[str, Any],
    projection: BAIEGEventModelInputProjectionV2,
    *,
    canonical: Mapping[str, Any],
    outer_support_receipt_sha256: str,
) -> dict[str, Any]:
    result = deepcopy(dict(legacy))
    event = projection.model_input_event
    dependencies = projection.raw_sample_dependency_sidecar["dependencies"]
    dependency_by_index = {
        int(row["source_token_index"]): row for row in dependencies
    }
    if list(dependency_by_index) != list(range(len(dependencies))):
        raise ValueError("projection raw dependencies lost exact token order")

    ledger_by_id: dict[str, dict[str, Any]] = {}
    for ledger in result["cell_source_ledgers"]:
        active_ids: list[str] = []
        active_hashes: list[str] = []
        active_closure: list[bool] = []
        for row in ledger["source_rows"]:
            token_index = int(row["source_token_index"])
            dependency = dependency_by_index[token_index]
            if "p0_token_dependency_sha256" not in row:
                raise ValueError("legacy adapter row lacks its synthetic binding")
            row.pop("p0_token_dependency_sha256")
            row.update(
                {
                    "raw_dependency_id": dependency["dependency_id"],
                    "raw_dependency_sha256": dependency["dependency_sha256"],
                    "raw_dependency_closure_proven": dependency["raw_support"][
                        "raw_dependency_closure_proven"
                    ],
                }
            )
            if row["signal_eligible"]:
                active_ids.append(str(dependency["dependency_id"]))
                active_hashes.append(str(dependency["dependency_sha256"]))
                active_closure.append(
                    bool(
                        dependency["raw_support"][
                            "raw_dependency_closure_proven"
                        ]
                    )
                )
        ledger["active_raw_dependency_ids"] = sorted(active_ids)
        ledger["active_raw_dependency_sha256s"] = sorted(active_hashes)
        ledger["active_raw_dependency_closure_proven"] = bool(
            active_closure and all(active_closure)
        )
        ledger["ledger_sha256"] = "CONTENT-ADDRESS-PENDING"
        ledger["ledger_sha256"] = _canonical_sha256(ledger)
        ledger_by_id[str(ledger["cell_id"])] = ledger

    for candidate in result["candidate_cells"]:
        ledger = ledger_by_id[str(candidate["cell_id"])]
        candidate["raw_dependency_sha256s"] = list(
            ledger["active_raw_dependency_sha256s"]
        )

    active_indices = torch.nonzero(
        event.token_signal_mask, as_tuple=False
    ).flatten().tolist()
    active_closures = [
        bool(
            dependency_by_index[int(index)]["raw_support"][
                "raw_dependency_closure_proven"
            ]
        )
        for index in active_indices
    ]
    closed_active_count = sum(active_closures)
    all_active_closed = bool(active_closures and all(active_closures))

    result["schema_version"] = BA_IEG_P0_INNER_ROUTER_ADAPTER_SCHEMA_VERSION_V2
    result["method_id"] = BA_IEG_P0_INNER_ROUTER_ADAPTER_METHOD_ID_V2
    result["source_binding"] = _projection_source_binding(
        projection,
        canonical=canonical,
        outer_support_receipt_sha256=outer_support_receipt_sha256,
    )
    result["diagnostics"].update(
        {
            "raw_dependency_row_count": len(dependencies),
            "active_token_raw_dependency_count": len(active_indices),
            "closed_active_token_raw_dependency_count": closed_active_count,
            "all_active_token_raw_dependency_closure_proven": (
                all_active_closed
            ),
        }
    )
    result["scope_receipt"] = deepcopy(_SCOPE_RECEIPT_V2)
    result["artifact_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["artifact_sha256"] = _canonical_sha256(result)
    return result


def _validate_embedded_artifact_v2(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _TOP_LEVEL_KEYS:
        raise ValueError("P0 adapter v2 artifact has missing/unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != BA_IEG_P0_INNER_ROUTER_ADAPTER_SCHEMA_VERSION_V2:
        raise ValueError("P0 adapter v2 schema drifted")
    if data["method_id"] != BA_IEG_P0_INNER_ROUTER_ADAPTER_METHOD_ID_V2:
        raise ValueError("P0 adapter v2 method drifted")
    if data["scope_receipt"] != _SCOPE_RECEIPT_V2:
        raise ValueError("P0 adapter v2 scope drifted")
    if data["status"] not in {"materialized", "not_evaluable"}:
        raise ValueError("P0 adapter v2 status is invalid")
    source = data["source_binding"]
    if type(source) is not dict or set(source) != _SOURCE_BINDING_KEYS:
        raise ValueError("P0 adapter v2 source binding is invalid")
    for name in (
        "p0_materialization_receipt_sha256",
        "event_input_receipt_sha256",
        "projection_v2_receipt_sha256",
        "deterministic_target_sidecar_receipt_sha256",
        "deterministic_target_receipt_sha256",
        "raw_dependency_sidecar_sha256",
        "raw_dependency_roster_sha256",
        "canonical_signal_sha256",
        "canonical_receipt_sha256",
        "outer_support_receipt_sha256",
    ):
        _sha256(source[name], f"source.{name}")
    _identifier(source["raw_dependency_sidecar_id"], "raw dependency sidecar ID")
    if (
        source["target_free_model_input_used"] is not True
        or source["host_replayed_raw_dependency_used"] is not True
    ):
        raise ValueError("P0 adapter v2 lost its projection/host-replay boundary")

    candidates = data["candidate_cells"]
    ledgers = data["cell_source_ledgers"]
    if not isinstance(candidates, list) or not isinstance(ledgers, list):
        raise TypeError("P0 adapter v2 candidates/ledgers must be arrays")
    ledger_by_id: dict[str, Mapping[str, Any]] = {}
    active_token_indices: set[int] = set()
    for ledger in ledgers:
        if type(ledger) is not dict or set(ledger) != _LEDGER_KEYS:
            raise ValueError("P0 adapter v2 ledger has missing/unknown fields")
        cell_id = _identifier(ledger["cell_id"], "ledger.cell_id")
        if cell_id in ledger_by_id:
            raise ValueError("P0 adapter v2 repeats a cell ledger")
        digest_source = deepcopy(ledger)
        supplied_ledger_sha = _sha256(
            digest_source["ledger_sha256"], "ledger_sha256"
        )
        digest_source["ledger_sha256"] = "CONTENT-ADDRESS-PENDING"
        if supplied_ledger_sha != _canonical_sha256(digest_source):
            raise ValueError("P0 adapter v2 ledger hash drifted")
        active_rows: list[Mapping[str, Any]] = []
        for row in ledger["source_rows"]:
            if type(row) is not dict or set(row) != _SOURCE_ROW_KEYS:
                raise ValueError(
                    "P0 adapter v2 source row has missing/unknown fields"
                )
            if "p0_token_dependency_sha256" in row:
                raise ValueError("P0 adapter v2 retained a synthetic dependency")
            _identifier(row["raw_dependency_id"], "raw_dependency_id")
            _sha256(row["raw_dependency_sha256"], "raw_dependency_sha256")
            if type(row["raw_dependency_closure_proven"]) is not bool:
                raise TypeError("raw dependency closure must be boolean")
            if row["signal_eligible"]:
                token_index = int(row["source_token_index"])
                if token_index in active_token_indices:
                    raise ValueError("one active token entered multiple v2 cells")
                active_token_indices.add(token_index)
                active_rows.append(row)
        expected_ids = sorted(
            str(row["raw_dependency_id"]) for row in active_rows
        )
        expected_hashes = sorted(
            str(row["raw_dependency_sha256"]) for row in active_rows
        )
        expected_closure = bool(
            active_rows
            and all(row["raw_dependency_closure_proven"] for row in active_rows)
        )
        if (
            ledger["active_raw_dependency_ids"] != expected_ids
            or ledger["active_raw_dependency_sha256s"] != expected_hashes
            or ledger["active_raw_dependency_closure_proven"]
            is not expected_closure
        ):
            raise ValueError("P0 adapter v2 ledger dependency binding drifted")
        ledger_by_id[cell_id] = ledger

    for candidate in candidates:
        cell_id = str(candidate.get("cell_id"))
        if cell_id not in ledger_by_id:
            raise ValueError("P0 adapter v2 candidate lacks a ledger")
        ledger = ledger_by_id[cell_id]
        if candidate.get("raw_dependency_sha256s") != ledger[
            "active_raw_dependency_sha256s"
        ]:
            raise ValueError("candidate lost its real raw dependency hashes")
        if (
            candidate.get("permission") == "morphology_native"
            and candidate.get("onset_evidence_authorized") is not False
        ):
            raise ValueError("native candidate acquired onset authorization")
    if set(ledger_by_id) != {str(cell["cell_id"]) for cell in candidates}:
        raise ValueError("P0 adapter v2 persisted an orphan ledger")
    if data["status"] == "materialized":
        if not candidates:
            raise ValueError("materialized P0 adapter v2 needs candidates")
        policy = BAIEGInnerRaggedRouterPolicyV1.from_dict(
            data["policy_binding"]["inner_router_policy"]
        )
        materialize_ba_ieg_inner_ragged_router_v1(
            event_id=source["event_id"],
            canonical_signal_sha256=source["canonical_signal_sha256"],
            outer_support_receipt_sha256=source["outer_support_receipt_sha256"],
            outer_support_union=data["outer_support_union"],
            candidate_cells=candidates,
            policy=policy,
        )
    elif candidates or ledgers:
        raise ValueError("not-evaluable P0 adapter v2 cannot claim candidates")

    supplied_hash = _sha256(data["artifact_sha256"], "artifact_sha256")
    digest_source = deepcopy(data)
    digest_source["artifact_sha256"] = "CONTENT-ADDRESS-PENDING"
    if supplied_hash != _canonical_sha256(digest_source):
        raise ValueError("P0 adapter v2 artifact hash drifted")
    return data


def _build_ba_ieg_p0_inner_router_candidates_v2(
    projection: BAIEGEventModelInputProjectionV2,
    *,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
    outer_support_receipt_sha256: str,
    outer_support_union: Sequence[Sequence[float]],
    p0_policy: BAIEGP0TokenizationPolicy,
    router_policy: BAIEGInnerRaggedRouterPolicyV1,
) -> dict[str, Any]:
    validate_ba_ieg_event_model_input_projection_v2(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    if not isinstance(p0_policy, BAIEGP0TokenizationPolicy):
        raise TypeError("p0_policy must be BAIEGP0TokenizationPolicy")
    if not isinstance(router_policy, BAIEGInnerRaggedRouterPolicyV1):
        raise TypeError("router_policy must be BAIEGInnerRaggedRouterPolicyV1")
    outer_receipt = _sha256(
        outer_support_receipt_sha256, "outer_support_receipt_sha256"
    )
    canonical = validate_canonical_signal_receipt(canonical_signal_receipt)
    legacy = (
        _adapter_v1.materialize_ba_ieg_target_free_p0_inner_router_candidates_v1(
            projection.model_input_event,
            source_p0_materialization_receipt_sha256=(
                projection.source_p0_materialization_receipt_sha256
            ),
            p0_policy=p0_policy,
            canonical_signal_receipt=canonical_signal_receipt,
            outer_support_receipt_sha256=outer_receipt,
            outer_support_union=outer_support_union,
            router_policy=router_policy,
        )
    )
    artifact = _replace_synthetic_dependencies(
        legacy,
        projection,
        canonical=canonical,
        outer_support_receipt_sha256=outer_receipt,
    )
    return _validate_embedded_artifact_v2(artifact)


def materialize_ba_ieg_p0_inner_router_candidates_v2(
    projection: BAIEGEventModelInputProjectionV2,
    *,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
    outer_support_receipt_sha256: str,
    outer_support_union: Sequence[Sequence[float]] | None = None,
    p0_policy: BAIEGP0TokenizationPolicy,
    router_policy: BAIEGInnerRaggedRouterPolicyV1 = BAIEGInnerRaggedRouterPolicyV1(),
) -> dict[str, Any]:
    """Materialize channel-neutral cells with real raw-dependency bindings."""

    if not isinstance(projection, BAIEGEventModelInputProjectionV2):
        raise TypeError("adapter v2 requires BAIEGEventModelInputProjectionV2")
    if outer_support_union is None:
        outer_support_union = [
            list(projection.model_input_event.analysis_interval_seconds)
        ]
    return _build_ba_ieg_p0_inner_router_candidates_v2(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
        outer_support_receipt_sha256=outer_support_receipt_sha256,
        outer_support_union=outer_support_union,
        p0_policy=p0_policy,
        router_policy=router_policy,
    )


def validate_ba_ieg_p0_inner_router_candidate_materialization_v2(
    payload: object,
    *,
    projection: BAIEGEventModelInputProjectionV2,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
    outer_support_receipt_sha256: str,
    outer_support_union: Sequence[Sequence[float]] | None = None,
    p0_policy: BAIEGP0TokenizationPolicy,
    router_policy: BAIEGInnerRaggedRouterPolicyV1 = BAIEGInnerRaggedRouterPolicyV1(),
) -> dict[str, Any]:
    """Replay projection, raw rows and complete adapter output from host roots."""

    data = _validate_embedded_artifact_v2(payload)
    if outer_support_union is None:
        outer_support_union = [
            list(projection.model_input_event.analysis_interval_seconds)
        ]
    expected = _build_ba_ieg_p0_inner_router_candidates_v2(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
        outer_support_receipt_sha256=outer_support_receipt_sha256,
        outer_support_union=outer_support_union,
        p0_policy=p0_policy,
        router_policy=router_policy,
    )
    if data != expected:
        raise ValueError(
            "P0 adapter v2 does not replay from projection and host roots"
        )
    return data


def route_ba_ieg_p0_inner_router_candidates_v2(
    payload: object,
    *,
    projection: BAIEGEventModelInputProjectionV2,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
    outer_support_receipt_sha256: str,
    outer_support_union: Sequence[Sequence[float]] | None = None,
    p0_policy: BAIEGP0TokenizationPolicy,
    router_policy: BAIEGInnerRaggedRouterPolicyV1 = BAIEGInnerRaggedRouterPolicyV1(),
) -> dict[str, Any]:
    """Host-replay adapter v2 before entering the unchanged inner router."""

    artifact = validate_ba_ieg_p0_inner_router_candidate_materialization_v2(
        payload,
        projection=projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
        outer_support_receipt_sha256=outer_support_receipt_sha256,
        outer_support_union=outer_support_union,
        p0_policy=p0_policy,
        router_policy=router_policy,
    )
    if artifact["status"] != "materialized":
        raise ValueError("a not-evaluable adapter v2 artifact cannot be routed")
    source = artifact["source_binding"]
    return materialize_ba_ieg_inner_ragged_router_v1(
        event_id=source["event_id"],
        canonical_signal_sha256=source["canonical_signal_sha256"],
        outer_support_receipt_sha256=source["outer_support_receipt_sha256"],
        outer_support_union=artifact["outer_support_union"],
        candidate_cells=artifact["candidate_cells"],
        policy=router_policy,
    )


__all__ = [
    "BA_IEG_P0_INNER_ROUTER_ADAPTER_METHOD_ID_V2",
    "BA_IEG_P0_INNER_ROUTER_ADAPTER_SCHEMA_VERSION_V2",
    "materialize_ba_ieg_p0_inner_router_candidates_v2",
    "route_ba_ieg_p0_inner_router_candidates_v2",
    "validate_ba_ieg_p0_inner_router_candidate_materialization_v2",
]
