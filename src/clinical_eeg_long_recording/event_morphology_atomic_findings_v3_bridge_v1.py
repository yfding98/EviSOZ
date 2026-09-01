"""Bridge signal-only morphology proposals into measured Findings v3 atoms.

The bridge is deliberately narrow.  A deterministic query enumerator, a
frozen atomic proposal head, or a synthetic signal injector may nominate a
``(native morphology view, unit, physical interval)`` query.  The query is
then measured by :mod:`deterministic_event_morphology_primitives_v1` and the
replayable numerical values are projected into ``event_eeg_findings_v3``.

Proposal scores remain routing metadata.  They never become measurements.
The bridge does not qualify a clinical waveform term, an event boundary, an
event, an onset source, a localization claim, or report text.  Every positive
measurement uses an instantaneous physical-amplitude view and carries an
atom-local raw-sample dependency with onset authorization disabled.

All original proposals remain in a denominator roster.  Boundary clipping,
rejection, exact-query de-duplication, canonical ordering, and overlap
components are deterministic.  Non-identical overlapping intervals remain
separate; adjacent half-open intervals are not overlapping.  External
annotations, spreadsheets, clinical text, physician labels, and language
models are not accepted by this API.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Final, Mapping, Sequence

import torch

from .canonical_signal_views import (
    validate_canonical_signal_receipt,
    validate_signal_view_receipt,
)
from .deterministic_event_findings import (
    _canonical_sha256,
    deterministic_view_tensor_sha256,
)
from .deterministic_event_findings_v2 import (
    DEFAULT_EVENT_FINDINGS_V2_REGISTRY_BINDINGS,
    _UNIT_CATALOG as _BASE_UNIT_CATALOG,
    _finalize_raw_sample_dependency,
)
from .deterministic_event_morphology_primitives_v1 import (
    DEFAULT_EVENT_MORPHOLOGY_PRIMITIVE_POLICY,
    EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES,
    EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS,
    EventMorphologyPrimitivePolicy,
    EventMorphologyPrimitiveQuery,
    EventMorphologyPrimitiveViewInput,
    materialize_event_morphology_primitive_supervision_v1,
    validate_event_morphology_primitive_supervision_v1,
)
from .event_findings_v3_validation import (
    validate_event_eeg_findings_v3_payload,
)


EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_event_morphology_atomic_findings_v3_bridge_v1"
EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_METHOD_ID: Final[
    str
] = "EVENT-MORPHOLOGY-ATOMIC-FINDINGS-V3-BRIDGE-V1"
EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_POLICY_ID: Final[
    str
] = "EVENT-MORPHOLOGY-ATOMIC-FINDINGS-V3-BRIDGE-POLICY-V1"

_MORPHOLOGY_TERM_ID = "deterministic_morphology_candidate"
_PLACEHOLDER_EVIDENCE_ID = "E-MORPHOLOGY-UNQUALIFIED"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOL = 1e-9

_SOURCE_KIND_TO_QUERY_AUTHORITY = {
    "deterministic_signal_enumerator": "deterministic_signal_proposal",
    "frozen_atomic_head": "frozen_model_proposal",
    "synthetic_signal_injection": "synthetic_signal_injection",
}
_QUERY_AUTHORITY_ORDER = {
    "deterministic_signal_proposal": 0,
    "frozen_model_proposal": 1,
    "synthetic_signal_injection": 2,
}

_PROPOSAL_FIREWALL = {
    "eeg_samples_used": True,
    "edf_annotation_api_called": False,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_used": False,
    "sleep_or_activation_labels_used": False,
    "qwen_or_other_llm_used": False,
}

_AUTHORIZATION = {
    "assertion_scope": "replayable_numerical_morphology_measurements_only",
    "clinical_term_qualification_authorized": False,
    "negative_clinical_assertion_authorized": False,
    "event_boundary_authorized": False,
    "event_qualification_authorized": False,
    "onset_claim_authorized": False,
    "soz_or_ez_claim_authorized": False,
    "report_text_authorized": False,
    "proposal_scores_are_signal_facts": False,
}

EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_UNIT_CATALOG: Final[dict[str, str]] = {
    **deepcopy(_BASE_UNIT_CATALOG),
    "uV_per_s": "amplitude_rate_microvolts_per_second",
    "uV_per_s2": "amplitude_curvature_microvolts_per_second_squared",
    "count": "dimensionless_event_count",
}

EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_UNIT_REGISTRY_BINDING: Final[dict[str, object]] = {
    "registry_id": "DETERMINISTIC-EEG-UNIT-REGISTRY-MORPHOLOGY-BRIDGE",
    "version": "2.1.0",
    "registry_sha256": _canonical_sha256(
        EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_UNIT_CATALOG
    ),
    "trust_status": "host_trusted",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _sha256(body)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be an event-contract compatible ID")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _interval(value: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    start = _finite(value[0], f"{name}[0]", minimum=0.0)
    stop = _finite(value[1], f"{name}[1]", minimum=0.0)
    if stop <= start + _TOL:
        raise ValueError(f"{name} must have positive duration")
    return start, stop


def _sorted_reasons(values: Sequence[str]) -> list[str]:
    result = sorted(set(str(item) for item in values))
    if any(
        not item or item != item.strip() or not _ID_PATTERN.fullmatch(item)
        for item in result
    ):
        raise ValueError("reason codes must be event-contract compatible IDs")
    return result


@dataclass(frozen=True)
class EventMorphologyAtomicBridgePolicy:
    """Frozen query-canonicalization policy for the bridge."""

    minimum_clipped_duration_seconds: float = 0.02
    duplicate_key_policy: str = "view_unit_exact_clipped_half_open_interval_v1"
    overlap_policy: str = "strict_half_open_overlap_keep_all_v1"
    mixed_authority_policy: str = "lowest_frozen_authority_rank_v1"

    def __post_init__(self) -> None:
        value = _finite(
            self.minimum_clipped_duration_seconds,
            "minimum_clipped_duration_seconds",
            minimum=0.0,
        )
        if value <= _TOL:
            raise ValueError("minimum_clipped_duration_seconds must be positive")
        object.__setattr__(self, "minimum_clipped_duration_seconds", value)
        if self.duplicate_key_policy != (
            "view_unit_exact_clipped_half_open_interval_v1"
        ):
            raise ValueError("v1 duplicate-key policy is frozen")
        if self.overlap_policy != "strict_half_open_overlap_keep_all_v1":
            raise ValueError("v1 overlap policy is frozen")
        if self.mixed_authority_policy != "lowest_frozen_authority_rank_v1":
            raise ValueError("v1 mixed-authority policy is frozen")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "policy_id": EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_POLICY_ID,
            "boundary_coordinate_system": "recording_relative_seconds",
            "boundary_intervals_are_half_open": True,
            "routing_score_used_as_measurement": False,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


DEFAULT_EVENT_MORPHOLOGY_ATOMIC_BRIDGE_POLICY = EventMorphologyAtomicBridgePolicy()


@dataclass(frozen=True)
class EventMorphologyProposalProducerBinding:
    """Immutable identity of one signal-only proposal producer."""

    producer_id: str
    producer_version: str
    source_kind: str
    artifact_sha256: str
    code_sha256: str
    policy_sha256: str
    input_contract_sha256: str
    model_weights_sha256: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.producer_id, "producer_id")
        if (
            not isinstance(self.producer_version, str)
            or not self.producer_version.strip()
            or len(self.producer_version) > 64
        ):
            raise ValueError("producer_version must be a non-empty short string")
        if self.source_kind not in _SOURCE_KIND_TO_QUERY_AUTHORITY:
            raise ValueError(
                "source_kind must be a deterministic enumerator, frozen atomic "
                "head, or synthetic signal injection"
            )
        for field in (
            "artifact_sha256",
            "code_sha256",
            "policy_sha256",
            "input_contract_sha256",
        ):
            _hash(getattr(self, field), field)
        if self.source_kind == "frozen_atomic_head":
            _hash(self.model_weights_sha256, "model_weights_sha256")
        elif self.model_weights_sha256 is not None:
            raise ValueError("model_weights_sha256 is reserved for frozen_atomic_head")

    @property
    def query_authority(self) -> str:
        return _SOURCE_KIND_TO_QUERY_AUTHORITY[self.source_kind]

    def to_receipt(self) -> dict[str, object]:
        content = {
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "source_kind": self.source_kind,
            "query_authority": self.query_authority,
            "artifact_sha256": self.artifact_sha256,
            "code_sha256": self.code_sha256,
            "policy_sha256": self.policy_sha256,
            "input_contract_sha256": self.input_contract_sha256,
            "model_weights_sha256": self.model_weights_sha256,
            "firewall": deepcopy(_PROPOSAL_FIREWALL),
            "authorization": {
                "query_proposal_authorized": True,
                "measurement_authorized": False,
                "clinical_term_authorized": False,
                "onset_or_localization_claim_authorized": False,
                "report_text_authorized": False,
            },
        }
        receipt_id = "MORPHPROD-" + _sha256(content)[:24]
        result = {
            "schema_version": "clinical_eeg_morphology_proposal_producer_v1",
            "receipt_id": receipt_id,
            **content,
        }
        result["receipt_sha256"] = _self_hash(result, "receipt_sha256")
        return result


@dataclass(frozen=True)
class EventMorphologyAtomicProposal:
    """One interval proposal; its score is routing metadata only."""

    proposal_id: str
    producer_id: str
    view_id: str
    unit_id: str
    recording_interval_seconds: tuple[float, float]
    routing_score: float

    def __post_init__(self) -> None:
        _identifier(self.proposal_id, "proposal_id")
        _identifier(self.producer_id, "producer_id")
        _identifier(self.view_id, "view_id")
        _identifier(self.unit_id, "unit_id")
        object.__setattr__(
            self,
            "recording_interval_seconds",
            _interval(
                self.recording_interval_seconds,
                "recording_interval_seconds",
            ),
        )
        object.__setattr__(
            self,
            "routing_score",
            _finite(self.routing_score, "routing_score"),
        )


def _validate_producer_receipt(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("proposal producer receipt must be an object")
    receipt = deepcopy(value)
    expected_keys = {
        "schema_version",
        "receipt_id",
        "producer_id",
        "producer_version",
        "source_kind",
        "query_authority",
        "artifact_sha256",
        "code_sha256",
        "policy_sha256",
        "input_contract_sha256",
        "model_weights_sha256",
        "firewall",
        "authorization",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError("proposal producer receipt keys drifted")
    if receipt["schema_version"] != ("clinical_eeg_morphology_proposal_producer_v1"):
        raise ValueError("proposal producer schema version drifted")
    binding = EventMorphologyProposalProducerBinding(
        producer_id=receipt["producer_id"],
        producer_version=receipt["producer_version"],
        source_kind=receipt["source_kind"],
        artifact_sha256=receipt["artifact_sha256"],
        code_sha256=receipt["code_sha256"],
        policy_sha256=receipt["policy_sha256"],
        input_contract_sha256=receipt["input_contract_sha256"],
        model_weights_sha256=receipt["model_weights_sha256"],
    )
    expected = binding.to_receipt()
    if receipt != expected:
        raise ValueError("proposal producer receipt content/hash drifted")
    return receipt


def _canonical_producer_roster(
    values: Sequence[EventMorphologyProposalProducerBinding],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not values:
        raise ValueError("at least one proposal producer binding is required")
    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, binding in enumerate(values):
        if not isinstance(binding, EventMorphologyProposalProducerBinding):
            raise TypeError(
                f"proposal_producers[{index}] must be "
                "EventMorphologyProposalProducerBinding"
            )
        receipt = binding.to_receipt()
        producer_id = str(receipt["producer_id"])
        if producer_id in by_id:
            raise ValueError("proposal producer IDs must be unique")
        by_id[producer_id] = receipt
        rows.append(receipt)
    rows.sort(key=lambda row: (str(row["producer_id"]), str(row["receipt_id"])))
    return rows, by_id


def _clip_disposition(
    requested: tuple[float, float],
    analysis: tuple[float, float],
    *,
    policy: EventMorphologyAtomicBridgePolicy,
) -> tuple[str, tuple[float, float] | None]:
    start = max(requested[0], analysis[0])
    stop = min(requested[1], analysis[1])
    if stop <= start + _TOL:
        return "rejected_outside_analysis_interval", None
    clipped = (float(start), float(stop))
    if stop - start + _TOL < policy.minimum_clipped_duration_seconds:
        return "rejected_clipped_interval_too_short", clipped
    left = requested[0] < analysis[0] - _TOL
    right = requested[1] > analysis[1] + _TOL
    if left and right:
        return "accepted_clipped_both", clipped
    if left:
        return "accepted_clipped_left", clipped
    if right:
        return "accepted_clipped_right", clipped
    return "accepted_unclipped", clipped


def _query_id(
    *,
    event_id: str,
    view_id: str,
    unit_id: str,
    interval: tuple[float, float],
    query_authority: str,
) -> str:
    return (
        "MORPHQ-"
        + _sha256(
            {
                "event_id": event_id,
                "query": {
                    "view_id": view_id,
                    "unit_id": unit_id,
                    "recording_interval_seconds": list(interval),
                    "query_authority": query_authority,
                },
            }
        )[:24]
    )


def _strict_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] - _TOL and right[0] < left[1] - _TOL


def _assign_overlap_components(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault((row["view_id"], row["unit_id"]), []).append(index)
    for key in sorted(grouped):
        indices = grouped[key]
        parent = {index: index for index in indices}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for ordinal, left_index in enumerate(indices):
            left = tuple(rows[left_index]["recording_interval_seconds"])
            for right_index in indices[ordinal + 1 :]:
                right = tuple(rows[right_index]["recording_interval_seconds"])
                if _strict_overlap(left, right):
                    union(left_index, right_index)
        components: dict[int, list[int]] = {}
        for index in indices:
            components.setdefault(find(index), []).append(index)
        for members in components.values():
            member_body = [
                {
                    "canonical_query_id": rows[index]["canonical_query_id"],
                    "recording_interval_seconds": rows[index][
                        "recording_interval_seconds"
                    ],
                }
                for index in sorted(
                    members,
                    key=lambda item: (
                        rows[item]["recording_interval_seconds"][0],
                        rows[item]["recording_interval_seconds"][1],
                        rows[item]["canonical_query_id"],
                    ),
                )
            ]
            component_id = (
                "MORPHOVL-"
                + _sha256(
                    {
                        "view_id": key[0],
                        "unit_id": key[1],
                        "members": member_body,
                        "policy": "strict_half_open_overlap_keep_all_v1",
                    }
                )[:24]
            )
            for index in members:
                rows[index]["overlap_component_id"] = component_id


def _canonicalize_proposals(
    *,
    event_id: str,
    proposals: Sequence[EventMorphologyAtomicProposal],
    producers_by_id: Mapping[str, Mapping[str, Any]],
    analysis_interval: tuple[float, float],
    policy: EventMorphologyAtomicBridgePolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not proposals:
        raise ValueError("at least one morphology proposal is required")
    seen_ids: set[str] = set()
    staged: list[
        tuple[
            EventMorphologyAtomicProposal,
            dict[str, Any],
            str,
            tuple[float, float] | None,
        ]
    ] = []
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, EventMorphologyAtomicProposal):
            raise TypeError(f"proposals[{index}] must be EventMorphologyAtomicProposal")
        if proposal.proposal_id in seen_ids:
            raise ValueError("proposal IDs must be unique")
        seen_ids.add(proposal.proposal_id)
        producer = producers_by_id.get(proposal.producer_id)
        if producer is None:
            raise ValueError("proposal references an unknown producer binding")
        disposition, clipped = _clip_disposition(
            proposal.recording_interval_seconds,
            analysis_interval,
            policy=policy,
        )
        staged.append((proposal, dict(producer), disposition, clipped))
    staged.sort(
        key=lambda item: (
            item[0].view_id,
            item[0].unit_id,
            item[0].recording_interval_seconds[0],
            item[0].recording_interval_seconds[1],
            item[0].proposal_id,
            item[0].producer_id,
        )
    )

    accepted_groups: dict[
        tuple[str, str, float, float],
        list[tuple[EventMorphologyAtomicProposal, dict[str, Any]]],
    ] = {}
    for proposal, producer, disposition, clipped in staged:
        if disposition.startswith("accepted_"):
            assert clipped is not None
            accepted_groups.setdefault(
                (proposal.view_id, proposal.unit_id, clipped[0], clipped[1]), []
            ).append((proposal, producer))

    canonical_queries: list[dict[str, Any]] = []
    query_id_by_proposal: dict[str, str] = {}
    for key in sorted(accepted_groups):
        members = accepted_groups[key]
        authorities = sorted(
            {str(producer["query_authority"]) for _, producer in members},
            key=lambda item: (_QUERY_AUTHORITY_ORDER[item], item),
        )
        authority = authorities[0]
        interval = (float(key[2]), float(key[3]))
        canonical_id = _query_id(
            event_id=event_id,
            view_id=key[0],
            unit_id=key[1],
            interval=interval,
            query_authority=authority,
        )
        contributor_ids = sorted(proposal.proposal_id for proposal, _ in members)
        producer_receipt_ids = sorted(
            {str(producer["receipt_id"]) for _, producer in members}
        )
        producer_receipt_sha256s = sorted(
            {str(producer["receipt_sha256"]) for _, producer in members}
        )
        row = {
            "canonical_query_id": canonical_id,
            "view_id": key[0],
            "unit_id": key[1],
            "recording_interval_seconds": list(interval),
            "query_authority": authority,
            "contributing_proposal_ids": contributor_ids,
            "proposal_producer_receipt_ids": producer_receipt_ids,
            "proposal_producer_receipt_sha256s": producer_receipt_sha256s,
            "overlap_component_id": "PENDING",
        }
        canonical_queries.append(row)
        for proposal, _ in members:
            query_id_by_proposal[proposal.proposal_id] = canonical_id
    canonical_queries.sort(
        key=lambda row: (
            row["view_id"],
            row["unit_id"],
            row["recording_interval_seconds"][0],
            row["recording_interval_seconds"][1],
            row["canonical_query_id"],
        )
    )
    _assign_overlap_components(canonical_queries)

    denominator: list[dict[str, Any]] = []
    for proposal, producer, disposition, clipped in staged:
        denominator.append(
            {
                "proposal_id": proposal.proposal_id,
                "producer_id": proposal.producer_id,
                "producer_receipt_id": str(producer["receipt_id"]),
                "producer_receipt_sha256": str(producer["receipt_sha256"]),
                "source_kind": str(producer["source_kind"]),
                "view_id": proposal.view_id,
                "unit_id": proposal.unit_id,
                "requested_recording_interval_seconds": list(
                    proposal.recording_interval_seconds
                ),
                "routing_score": float(proposal.routing_score),
                "routing_score_semantics": "proposal_routing_only_not_signal_fact",
                "boundary_disposition": disposition,
                "clipped_recording_interval_seconds": (
                    None if clipped is None else list(clipped)
                ),
                "canonical_query_id": query_id_by_proposal.get(proposal.proposal_id),
            }
        )
    return denominator, canonical_queries


def _validate_native_morphology_views(
    *,
    canonical: Mapping[str, Any],
    views: Sequence[EventMorphologyPrimitiveViewInput],
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, Any]]:
    if not views:
        raise ValueError("at least one native morphology view is required")
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(views):
        if not isinstance(item, EventMorphologyPrimitiveViewInput):
            raise TypeError(f"views[{index}] must be EventMorphologyPrimitiveViewInput")
        receipt = validate_signal_view_receipt(
            item.view_receipt,
            canonical,
            trusted_parent_views=trusted_parent_views,
        )
        if receipt["task_role"] != "findings_native_morphology":
            raise ValueError(
                "bridge views must use findings_native_morphology task role"
            )
        temporal = receipt["temporal_evidence"]
        if (
            temporal["future_sample_access"] is not False
            or temporal["onset_evidence_authorized"] is not False
            or temporal["dependency_policy"] != "instantaneous"
            or temporal["raw_support_end_policy"]
            != "at_or_before_unshifted_evidence_sample_v1"
        ):
            raise ValueError(
                "bridge morphology views must be instantaneous and onset-closed"
            )
        transform = receipt["transform_spec"]
        if (
            transform["filter"]["phase_policy"] != "none"
            or transform["normalization"]["preserves_physical_amplitude"] is not True
            or transform["clipping"]["applied"] is not False
        ):
            raise ValueError(
                "bridge morphology views must retain unclipped physical morphology"
            )
        view_id = str(receipt["view_id"])
        if view_id in result:
            raise ValueError("native morphology view IDs must be unique")
        unit_ids = tuple(str(row["unit_id"]) for row in receipt["output_units"])
        if any(row["physical_unit"] != "V" for row in receipt["output_units"]):
            raise ValueError("native morphology view units must be volts")
        tensor = item.tensor.detach().cpu().to(torch.float32).contiguous()
        expected_shape = (
            len(unit_ids),
            int(receipt["tensor_layout"]["tensor_sample_count"]),
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"view tensor shape {tuple(tensor.shape)} != receipt "
                f"{expected_shape}"
            )
        if (
            deterministic_view_tensor_sha256(tensor, unit_ids=unit_ids)
            != receipt["processed_view_sha256"]
        ):
            raise ValueError("native morphology view tensor hash mismatch")
        result[view_id] = receipt
    return result


def _assert_proposal_units_exist(
    proposals: Sequence[EventMorphologyAtomicProposal],
    view_map: Mapping[str, Mapping[str, Any]],
) -> None:
    unit_ids_by_view = {
        view_id: {str(row["unit_id"]) for row in receipt["output_units"]}
        for view_id, receipt in view_map.items()
    }
    for proposal in proposals:
        if proposal.view_id not in unit_ids_by_view:
            raise ValueError("proposal references an unknown morphology view")
        if proposal.unit_id not in unit_ids_by_view[proposal.view_id]:
            raise ValueError("proposal references an unknown morphology view unit")


def _materialize_sidecar(
    *,
    event_id: str,
    canonical: Mapping[str, Any],
    views: Sequence[EventMorphologyPrimitiveViewInput],
    analysis_interval: tuple[float, float],
    canonical_queries: Sequence[Mapping[str, Any]],
    primitive_policy: EventMorphologyPrimitivePolicy,
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, Any] | None:
    if not canonical_queries:
        return None
    queries = [
        EventMorphologyPrimitiveQuery(
            view_id=str(row["view_id"]),
            unit_id=str(row["unit_id"]),
            recording_interval_seconds=tuple(
                float(value) for value in row["recording_interval_seconds"]
            ),
            query_authority=str(row["query_authority"]),
        )
        for row in canonical_queries
    ]
    receipt = materialize_event_morphology_primitive_supervision_v1(
        event_id=event_id,
        canonical_receipt=canonical,
        views=views,
        analysis_interval_seconds=analysis_interval,
        queries=queries,
        policy=primitive_policy,
        trusted_parent_views=trusted_parent_views,
    )
    expected = [
        {
            "view_id": row["view_id"],
            "unit_id": row["unit_id"],
            "recording_interval_seconds": row["recording_interval_seconds"],
            "query_authority": row["query_authority"],
        }
        for row in canonical_queries
    ]
    if receipt["query_roster"] != expected:
        raise ValueError("morphology sidecar query roster drifted from bridge")
    if [row["query_id"] for row in receipt["rows"]] != [
        row["canonical_query_id"] for row in canonical_queries
    ]:
        raise ValueError("morphology sidecar query IDs drifted from bridge")
    return receipt


def _mask_component_sha256(receipt: Mapping[str, Any], key: str) -> str:
    return _sha256(
        {
            "view_receipt_id": receipt["view_receipt_id"],
            key: receipt["masks"][key],
        }
    )


def _raw_sample_dependency(
    *,
    source: Mapping[str, Any],
    view_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    interval = [float(value) for value in source["recording_interval_seconds"]]
    tensor_interval = [int(value) for value in source["tensor_sample_interval"]]
    clock = view_receipt["transform_spec"]["output_clock"]
    raw_intervals = []
    for row in source["raw_sample_intervals"]:
        raw_intervals.append(
            {
                "channel_id": str(row["channel_id"]),
                "sample_rate_numerator": int(row["sample_rate_numerator"]),
                "sample_rate_denominator": int(row["sample_rate_denominator"]),
                "channel_sample_count": int(row["channel_sample_count"]),
                "raw_start_sample": int(row["raw_start_sample"]),
                "raw_stop_sample_exclusive": int(row["raw_stop_sample_exclusive"]),
                "reported_evidence_start_sample": int(row["raw_start_sample"]),
                "reported_evidence_stop_sample_exclusive": int(
                    row["raw_stop_sample_exclusive"]
                ),
                "unshifted_decision_available_stop_sample_exclusive": int(
                    row["raw_stop_sample_exclusive"]
                ),
            }
        )
    body = {
        "schema_version": "clinical_eeg_raw_sample_dependency_v1",
        "dependency_status": "exact_instantaneous",
        "canonical_signal_sha256": str(source["source_signal_sha256"]),
        "source_view_id": str(source["view_id"]),
        "view_role": "canonical_physical_evidence",
        "evidence_recording_interval": interval,
        "support_components": [
            {
                "role": "reported_evidence_interval",
                "recording_interval": interval,
            }
        ],
        "decision_available_recording_seconds": float(interval[1]),
        "confirmation_latency_samples_on_view_clock": 0.0,
        "confirmation_latency_seconds": 0.0,
        "confirmation_policy": "none",
        "view_tensor_sample_interval": tensor_interval,
        "view_sampling_rate_numerator": int(clock["sampling_rate_numerator"]),
        "view_sampling_rate_denominator": int(clock["sampling_rate_denominator"]),
        "raw_sample_intervals": raw_intervals,
        "dependency_policy": "instantaneous",
        "future_sample_access": False,
        "onset_evidence_authorized": False,
        "onset_support_eligible": False,
        "processing_latency_samples_on_view_clock": 0.0,
        "processing_latency_seconds": 0.0,
        "processing_latency_policy": "none",
        "raw_support_end_policy": "at_or_before_unshifted_evidence_sample_v1",
        "receipt_lineage": {
            "canonical_receipt_sha256": str(source["canonical_receipt_sha256"]),
            "source_view_id": str(source["view_id"]),
            "source_view_receipt_id": str(source["view_receipt_id"]),
            "source_view_receipt_sha256": str(source["view_receipt_sha256"]),
            "source_transform_spec_sha256": str(source["transform_spec_sha256"]),
            "temporal_evidence_sha256": _sha256(view_receipt["temporal_evidence"]),
            "parent_view_bindings": deepcopy(view_receipt["parent_view_bindings"]),
        },
    }
    return _finalize_raw_sample_dependency(body)


def _overlap_duration(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _state_membership(
    payload: Mapping[str, Any],
    interval: tuple[float, float] | None,
    *,
    force_zero: bool,
) -> dict[str, float]:
    result = {name: 0.0 for name in ("S0", "S1", "S2", "S3")}
    if (
        interval is None
        or force_zero
        or payload["window"]["state_posterior_status"] == "not_evaluable"
    ):
        return result
    for segment in payload["window"]["state_segments"]:
        span = (
            float(segment["interval"]["start"]),
            float(segment["interval"]["stop"]),
        )
        weight = _overlap_duration(interval, span)
        for name in result:
            result[name] += weight * float(segment["posterior"][name])
    total = sum(result.values())
    if total <= _TOL:
        return {name: 0.0 for name in result}
    return {name: float(value / total) for name, value in result.items()}


def _old_morphology_placeholder(
    payload: Mapping[str, Any],
) -> tuple[int, int, Mapping[str, Any]]:
    findings = [
        (index, row)
        for index, row in enumerate(payload["findings"])
        if row["family"] == "morphology"
    ]
    if len(findings) != 1:
        raise ValueError(
            "bridge v1 requires exactly one unqualified morphology placeholder"
        )
    finding_index, finding = findings[0]
    if (
        finding["evidence_id"] != _PLACEHOLDER_EVIDENCE_ID
        or finding["term"]["term_id"] != _MORPHOLOGY_TERM_ID
        or finding["status"] != "not_evaluable"
    ):
        raise ValueError(
            "bridge v1 cannot replace an already-qualified morphology slice"
        )
    opportunity_id = str(finding["evaluation_opportunity_id"])
    opportunity_indices = [
        index
        for index, row in enumerate(payload["evaluation_opportunities"])
        if row["evaluation_opportunity_id"] == opportunity_id
    ]
    if len(opportunity_indices) != 1:
        raise ValueError("morphology placeholder opportunity is not unique")
    return finding_index, opportunity_indices[0], finding


def _contains_exact(value: object, target: str) -> bool:
    if value == target:
        return True
    if isinstance(value, Mapping):
        return any(_contains_exact(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_exact(item, target) for item in value)
    return False


def _patched_registry_trust(
    base_trust: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {
        str(key): deepcopy(dict(value)) for key, value in base_trust.items()
    }
    result["unit_registry"] = deepcopy(
        EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_UNIT_REGISTRY_BINDING
    )
    return result


def _atom_from_sidecar_row(
    *,
    payload: Mapping[str, Any],
    term: Mapping[str, Any],
    canonical_query: Mapping[str, Any],
    sidecar_row: Mapping[str, Any],
    sidecar_receipt: Mapping[str, Any],
    view_receipt: Mapping[str, Any],
    bridge_policy: EventMorphologyAtomicBridgePolicy,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = sidecar_row["source_binding"]
    if (
        source["view_id"] != canonical_query["view_id"]
        or source["unit_id"] != canonical_query["unit_id"]
        or source["query_authority"] != canonical_query["query_authority"]
        or source["requested_recording_interval_seconds"]
        != canonical_query["recording_interval_seconds"]
    ):
        raise ValueError("sidecar row source does not match its canonical query")
    unit_id = str(canonical_query["unit_id"])
    input_rows = [
        row for row in payload["montage"]["input_units"] if row["unit_id"] == unit_id
    ]
    montage_reason: list[str] = []
    if len(input_rows) != 1:
        montage_reason = ["query_unit_missing_from_v3_montage"]
        input_row = None
    else:
        input_row = input_rows[0]
        if (
            input_row["observation_status"] != "observed"
            or not input_row["evidence_eligible"]
        ):
            montage_reason = ["query_unit_not_v3_evidence_eligible"]

    masks = [bool(value) for value in sidecar_row["opportunity"]["target_value_mask"]]
    reasons = [
        [str(item) for item in values]
        for values in sidecar_row["opportunity"]["target_reason_codes"]
    ]
    target_specs = list(EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS)
    morphology_indices = [
        index
        for index, (_, _, family) in enumerate(target_specs)
        if family == "morphology"
    ]
    morphology_available = any(masks[index] for index in morphology_indices)
    if montage_reason or not morphology_available:
        status = "not_evaluable"
    elif all(masks):
        status = "present"
    else:
        status = "uncertain"

    query_id = str(canonical_query["canonical_query_id"])
    opportunity_id = (
        "OPP-MORPH-"
        + _sha256(
            {
                "query_id": query_id,
                "sidecar_row_binding_sha256": sidecar_row["row_binding_sha256"],
                "status": status,
            }
        )[:20]
    )
    evidence_id = (
        "E-MORPH-"
        + _sha256(
            {
                "query_id": query_id,
                "sidecar_row_binding_sha256": sidecar_row["row_binding_sha256"],
            }
        )[:20]
    )
    unavailable_reasons = _sorted_reasons(
        [item for index, row in enumerate(reasons) if not masks[index] for item in row]
        + [
            str(item)
            for item in sidecar_row["opportunity"]["aggregate_opportunity_reason_codes"]
        ]
        + montage_reason
    )

    if status == "not_evaluable":
        reason_codes = _sorted_reasons(
            unavailable_reasons + ["no_replayable_morphology_target"]
        )
        opportunity = {
            "evaluation_opportunity_id": opportunity_id,
            "family": "morphology",
            "term_id": _MORPHOLOGY_TERM_ID,
            "interval": None,
            "spatial_unit_keys": [],
            "source_view_ids": [],
            "status": "not_evaluable",
            "usable_fraction": 0.0,
            "effective_bandwidth_hz": None,
            "quality_mask_sha256": None,
            "reason_codes": reason_codes,
        }
        finding = {
            "evidence_id": evidence_id,
            "finding_group_id": str(canonical_query["overlap_component_id"]),
            "pattern_instance_id": None,
            "family": "morphology",
            "term": deepcopy(dict(term)),
            "assertion_level": "model_candidate",
            "status": "not_evaluable",
            "intrinsic_evidence_role": "limitation",
            "signal_temporal_context": "unknown",
            "ownership": {
                "owner_event_ids": [str(payload["event_id"])],
                "event_group_id": None,
                "protection_zone_id": str(
                    payload["window"]["protection_zone"]["protection_zone_id"]
                ),
                "ownership_status": "event_owned",
                "protection_zone_overlap_fraction": 1.0,
            },
            "state_membership": {name: 0.0 for name in ("S0", "S1", "S2", "S3")},
            "time_interval": None,
            "spatial_support": [],
            "measurements": [],
            "uncertainty": {
                "boundary": 1.0,
                "quality": 1.0,
                "background": 0.0,
                "model": 1.0,
                "reference_stability": 1.0,
                "semantics": "componentwise_descriptive_not_individual_correctness_probability",
            },
            "evaluation_opportunity_id": opportunity_id,
            "capability_receipt_id": None,
            "sensitivity_receipt_id": None,
            "term_decision_receipt_id": None,
            "waveform_evidence_ids": [],
            "raw_sample_dependency_ids": [],
        }
        return opportunity, finding

    interval = tuple(float(value) for value in source["recording_interval_seconds"])
    clock = view_receipt["transform_spec"]["output_clock"]
    resolution = float(clock["sampling_rate_denominator"]) / float(
        clock["sampling_rate_numerator"]
    )
    protection = tuple(
        float(value) for value in payload["window"]["protection_zone"]["interval"]
    )
    overlap_fraction = _overlap_duration(interval, protection) / (
        interval[1] - interval[0]
    )
    outside = overlap_fraction <= _TOL
    if input_row is None:
        raise AssertionError("evaluable morphology atom lost its montage unit")
    unit_type = str(input_row["unit_type"])
    spatial_key_type = "lead" if unit_type == "bipolar_lead" else "electrode"
    opportunity_reason_codes = (
        []
        if status == "present"
        else _sorted_reasons(
            unavailable_reasons + ["partial_morphology_targets_unavailable"]
        )
    )
    opportunity = {
        "evaluation_opportunity_id": opportunity_id,
        "family": "morphology",
        "term_id": _MORPHOLOGY_TERM_ID,
        "interval": {
            "start": float(interval[0]),
            "stop": float(interval[1]),
            "resolution_seconds": resolution,
        },
        "spatial_unit_keys": [f"{spatial_key_type}:{unit_id}"],
        "source_view_ids": [str(source["view_id"])],
        "status": "sufficient" if status == "present" else "limited",
        # Any overlap with a morphology-disabled mask closes every morphology
        # target in the numerical sidecar.  Reaching this branch therefore
        # establishes complete morphology-family sample opportunity; a
        # limited status reflects target geometry/other-family availability,
        # not a fabricated fractional sample denominator.
        "usable_fraction": 1.0,
        "effective_bandwidth_hz": [
            float(value) for value in source["effective_bandwidth_hz"]
        ],
        "quality_mask_sha256": str(source["quality_mask_sha256"]),
        "reason_codes": opportunity_reason_codes,
    }

    dependency = _raw_sample_dependency(
        source=source,
        view_receipt=view_receipt,
    )
    atom_policy_sha256 = _sha256(
        {
            "bridge_policy_sha256": bridge_policy.sha256,
            "primitive_policy_sha256": sidecar_receipt["policy_sha256"],
            "sidecar_receipt_sha256": sidecar_receipt["receipt_sha256"],
            "sidecar_row_binding_sha256": sidecar_row["row_binding_sha256"],
            "proposal_producer_receipt_sha256s": canonical_query[
                "proposal_producer_receipt_sha256s"
            ],
        }
    )
    measurements: list[dict[str, Any]] = []
    for index, ((name, unit, _), value, available) in enumerate(
        zip(target_specs, sidecar_row["values"], masks)
    ):
        if not available:
            continue
        numeric = float(value)
        tolerance = max(1e-12, abs(numeric) * 1e-12)
        measurements.append(
            {
                "measurement_id": "MEAS-MORPH-"
                + _sha256(
                    {
                        "query_id": query_id,
                        "target_index": index,
                        "target_name": name,
                        "sidecar_row_binding_sha256": sidecar_row["row_binding_sha256"],
                    }
                )[:20],
                "name_id": str(name),
                "value": numeric,
                "unit_id": str(unit),
                "unit_registry_status": "registered",
                "baseline_delta": None,
                "numerical_uncertainty": {
                    "status": "deterministic_replay_tolerance",
                    "lower": numeric - tolerance,
                    "upper": numeric + tolerance,
                    "coverage": None,
                    "calibration_receipt_id": None,
                },
                "producer_type": "deterministic_signal_measurement",
                "source_binding": {
                    "canonical_signal_sha256": str(source["source_signal_sha256"]),
                    "source_view_id": str(source["view_id"]),
                    "view_role": "canonical_physical_evidence",
                    "view_receipt_id": str(source["view_receipt_id"]),
                    "view_receipt_sha256": str(source["view_receipt_sha256"]),
                    "transform_spec_sha256": str(source["transform_spec_sha256"]),
                    "processed_view_sha256": str(source["processed_view_sha256"]),
                    "source_unit_ids": [unit_id],
                    "recording_interval": list(interval),
                    "tensor_sample_interval": [
                        int(value) for value in source["tensor_sample_interval"]
                    ],
                    "effective_bandwidth_hz": [
                        float(value) for value in source["effective_bandwidth_hz"]
                    ],
                    "reference_type": str(source["reference_type"]),
                    "evidence_family": "morphology",
                    "quality_mask_sha256": str(source["quality_mask_sha256"]),
                    "edge_mask_sha256": _mask_component_sha256(
                        view_receipt, "edge_invalid_intervals"
                    ),
                    "padding_mask_sha256": _mask_component_sha256(
                        view_receipt, "padding_intervals"
                    ),
                    "imputation_mask_sha256": None,
                    "evidence_eligible": True,
                    "ineligibility_reason_codes": [],
                    "background_reference_ids": [],
                    "method_id": EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_METHOD_ID,
                    "policy_sha256": atom_policy_sha256,
                    "raw_sample_dependency": deepcopy(dependency),
                },
            }
        )

    ownership = {
        "owner_event_ids": [] if outside else [str(payload["event_id"])],
        "event_group_id": None,
        "protection_zone_id": str(
            payload["window"]["protection_zone"]["protection_zone_id"]
        ),
        "ownership_status": "outside_protection" if outside else "event_owned",
        "protection_zone_overlap_fraction": float(0.0 if outside else overlap_fraction),
    }
    finding = {
        "evidence_id": evidence_id,
        "finding_group_id": str(canonical_query["overlap_component_id"]),
        "pattern_instance_id": None,
        "family": "morphology",
        "term": deepcopy(dict(term)),
        "assertion_level": "measured",
        "status": status,
        "intrinsic_evidence_role": (
            "non_event_context" if outside else "early_context"
        ),
        "signal_temporal_context": (
            "outside_candidate_protection" if outside else "unknown"
        ),
        "ownership": ownership,
        "state_membership": _state_membership(
            payload,
            interval,
            force_zero=outside,
        ),
        "time_interval": {
            "start": float(interval[0]),
            "stop": float(interval[1]),
            "resolution_seconds": resolution,
        },
        "spatial_support": [
            {
                "unit_type": spatial_key_type,
                "id": unit_id,
                "mapping_status": "direct",
                "observation_status": "observed",
                "evidence_eligible": True,
                "missing_reason_codes": [],
                "support_score": None,
                "field_observation": None,
            }
        ],
        "measurements": measurements,
        "uncertainty": {
            "boundary": 0.5,
            "quality": float(1.0 - sum(masks) / len(masks)),
            "background": 0.0,
            "model": 0.0,
            "reference_stability": 0.5,
            "semantics": "componentwise_descriptive_not_individual_correctness_probability",
        },
        "evaluation_opportunity_id": opportunity_id,
        "capability_receipt_id": None,
        "sensitivity_receipt_id": None,
        "term_decision_receipt_id": None,
        "waveform_evidence_ids": [],
        "raw_sample_dependency_ids": [str(dependency["dependency_id"])],
    }
    return opportunity, finding


def _patch_findings_v3(
    *,
    base: Mapping[str, Any],
    canonical_queries: Sequence[Mapping[str, Any]],
    sidecar: Mapping[str, Any] | None,
    view_map: Mapping[str, Mapping[str, Any]],
    bridge_policy: EventMorphologyAtomicBridgePolicy,
) -> dict[str, Any]:
    result = deepcopy(dict(base))
    finding_index, opportunity_index, placeholder = _old_morphology_placeholder(result)
    old_opportunity_id = str(placeholder["evaluation_opportunity_id"])
    term = deepcopy(placeholder["term"])
    result["findings"].pop(finding_index)
    result["evaluation_opportunities"].pop(opportunity_index)

    new_opportunities: list[dict[str, Any]] = []
    new_findings: list[dict[str, Any]] = []
    if canonical_queries:
        if sidecar is None:
            raise ValueError("accepted canonical queries require a morphology sidecar")
        if len(sidecar["rows"]) != len(canonical_queries):
            raise ValueError("sidecar row count does not match canonical queries")
        for query, row in zip(canonical_queries, sidecar["rows"]):
            opportunity, finding = _atom_from_sidecar_row(
                payload=result,
                term=term,
                canonical_query=query,
                sidecar_row=row,
                sidecar_receipt=sidecar,
                view_receipt=view_map[str(query["view_id"])],
                bridge_policy=bridge_policy,
            )
            new_opportunities.append(opportunity)
            new_findings.append(finding)
    elif sidecar is not None:
        raise ValueError("empty canonical query roster cannot carry a sidecar")

    result["evaluation_opportunities"][
        opportunity_index:opportunity_index
    ] = new_opportunities
    result["findings"][finding_index:finding_index] = new_findings

    if _contains_exact(result, _PLACEHOLDER_EVIDENCE_ID):
        raise ValueError("retired morphology placeholder remains referenced")
    if _contains_exact(result, old_opportunity_id):
        # The quality-family row is updated immediately below.  Any other
        # reference would make silent replacement unsafe.
        quality_refs = [
            row
            for row in result["quality"]["feature_evaluability"]
            if row["family"] == "morphology"
            and old_opportunity_id in row["evaluation_opportunity_ids"]
        ]
        if len(quality_refs) != 1:
            raise ValueError("retired morphology opportunity remains referenced")

    quality_rows = [
        row
        for row in result["quality"]["feature_evaluability"]
        if row["family"] == "morphology"
    ]
    if len(quality_rows) != 1:
        raise ValueError("v3 quality ledger lacks one morphology family row")
    quality = quality_rows[0]
    quality["evaluation_opportunity_ids"] = [
        str(row["evaluation_opportunity_id"]) for row in new_opportunities
    ]
    statuses = [str(row["status"]) for row in new_findings]
    if statuses and set(statuses) == {"present"}:
        quality["status"] = "available"
        quality["reason_codes"] = []
    elif any(status in {"present", "uncertain"} for status in statuses):
        quality["status"] = "limited"
        quality["reason_codes"] = ["morphology_atomic_query_evaluability_incomplete"]
    else:
        quality["status"] = "not_evaluable"
        quality["reason_codes"] = ["no_replayable_morphology_atomic_query"]

    if _contains_exact(result, old_opportunity_id):
        raise ValueError("retired morphology opportunity remains referenced")
    result["registry_bindings"]["unit_registry"] = deepcopy(
        EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_UNIT_REGISTRY_BINDING
    )
    model_ids = list(result["provenance"]["model_ids"])
    if EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_METHOD_ID not in model_ids:
        model_ids.append(EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_METHOD_ID)
    result["provenance"]["model_ids"] = model_ids
    return result


def _validate_bridge_base(payload: Mapping[str, Any]) -> None:
    if payload["migration"] is not None or payload["v3_migration"] is not None:
        raise ValueError("morphology bridge requires a native v3 payload")
    if payload["event_qualification"]["status"] not in {
        "unqualified_candidate",
        "not_evaluable",
    }:
        raise ValueError("morphology bridge cannot modify a qualified event payload")
    for key in (
        "calibration_receipts",
        "capability_qualification_receipts",
        "sensitivity_receipts",
        "term_decision_receipts",
    ):
        if payload[key]:
            raise ValueError(
                "morphology bridge v1 requires an unqualified deterministic "
                f"base without {key}"
            )
    if any(
        value is not False
        for value in payload["provenance"]["inference_exclusions"].values()
    ):
        raise ValueError("morphology bridge requires explicit EEG-only exclusions")
    if payload["registry_bindings"]["unit_registry"] != (
        DEFAULT_EVENT_FINDINGS_V2_REGISTRY_BINDINGS["unit_registry"]
    ):
        raise ValueError(
            "bridge v1 can only augment the frozen deterministic v2 unit registry"
        )


def _bundle_source_binding_sha256(bundle: Mapping[str, Any]) -> str:
    return _sha256(
        {
            "schema_version": bundle["schema_version"],
            "method_id": bundle["method_id"],
            "event_id": bundle["event_id"],
            "record_id": bundle["record_id"],
            "canonical_signal_sha256": bundle["canonical_signal_sha256"],
            "canonical_receipt_sha256": bundle["canonical_receipt_sha256"],
            "analysis_interval_seconds": bundle["analysis_interval_seconds"],
            "bridge_policy_sha256": bundle["bridge_policy_sha256"],
            "primitive_policy_sha256": bundle["primitive_policy_sha256"],
            "base_findings_v3_payload_sha256": bundle[
                "base_findings_v3_payload_sha256"
            ],
            "proposal_producer_roster_sha256": bundle[
                "proposal_producer_roster_sha256"
            ],
            "proposal_denominator_roster_sha256": bundle[
                "proposal_denominator_roster_sha256"
            ],
            "canonical_query_roster_sha256": bundle["canonical_query_roster_sha256"],
            "primitive_sidecar_receipt_sha256": bundle[
                "primitive_sidecar_receipt_sha256"
            ],
            "patched_findings_v3_payload_sha256": bundle[
                "patched_findings_v3_payload_sha256"
            ],
        }
    )


def materialize_event_morphology_atomic_findings_v3_bridge_v1(
    *,
    base_findings_v3_payload: object,
    canonical_receipt: object,
    views: Sequence[EventMorphologyPrimitiveViewInput],
    proposal_producers: Sequence[EventMorphologyProposalProducerBinding],
    proposals: Sequence[EventMorphologyAtomicProposal],
    trusted_v3_producer_receipts: Mapping[str, Mapping[str, object]],
    trusted_v3_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
    bridge_policy: EventMorphologyAtomicBridgePolicy = (
        DEFAULT_EVENT_MORPHOLOGY_ATOMIC_BRIDGE_POLICY
    ),
    primitive_policy: EventMorphologyPrimitivePolicy = (
        DEFAULT_EVENT_MORPHOLOGY_PRIMITIVE_POLICY
    ),
) -> dict[str, Any]:
    """Materialize a replayable proposal -> sidecar -> Findings v3 bridge."""

    if not isinstance(bridge_policy, EventMorphologyAtomicBridgePolicy):
        raise TypeError("bridge_policy must be EventMorphologyAtomicBridgePolicy")
    if not isinstance(primitive_policy, EventMorphologyPrimitivePolicy):
        raise TypeError("primitive_policy must be EventMorphologyPrimitivePolicy")
    base_registry_trust = (
        deepcopy(DEFAULT_EVENT_FINDINGS_V2_REGISTRY_BINDINGS)
        if trusted_v3_registry_bindings is None
        else deepcopy(dict(trusted_v3_registry_bindings))
    )
    base = validate_event_eeg_findings_v3_payload(
        base_findings_v3_payload,
        trusted_producer_receipts=trusted_v3_producer_receipts,
        trusted_registry_bindings=base_registry_trust,
    )
    _validate_bridge_base(base)
    canonical = validate_canonical_signal_receipt(canonical_receipt)
    if base["event_id"] == "":
        raise ValueError("base v3 event ID is empty")
    if base["provenance"]["record_id"] != canonical["recording_id"]:
        raise ValueError("base v3 record and canonical receipt disagree")
    if (
        base["provenance"]["canonical_signal_sha256"]
        != canonical["source_signal_sha256"]
    ):
        raise ValueError("base v3 canonical signal hash disagrees with receipt")
    if (
        abs(
            float(base["coordinates"]["recording_duration_seconds"])
            - float(canonical["recording_duration_seconds"])
        )
        > _TOL
    ):
        raise ValueError("base v3 duration disagrees with canonical receipt")

    analysis_interval = tuple(
        float(value) for value in base["window"]["final_interval"]
    )
    view_map = _validate_native_morphology_views(
        canonical=canonical,
        views=views,
        trusted_parent_views=trusted_parent_views,
    )
    _assert_proposal_units_exist(proposals, view_map)
    producer_roster, producers_by_id = _canonical_producer_roster(proposal_producers)
    denominator, canonical_queries = _canonicalize_proposals(
        event_id=str(base["event_id"]),
        proposals=proposals,
        producers_by_id=producers_by_id,
        analysis_interval=analysis_interval,
        policy=bridge_policy,
    )
    sidecar = _materialize_sidecar(
        event_id=str(base["event_id"]),
        canonical=canonical,
        views=views,
        analysis_interval=analysis_interval,
        canonical_queries=canonical_queries,
        primitive_policy=primitive_policy,
        trusted_parent_views=trusted_parent_views,
    )
    patched = _patch_findings_v3(
        base=base,
        canonical_queries=canonical_queries,
        sidecar=sidecar,
        view_map=view_map,
        bridge_policy=bridge_policy,
    )
    patched_registry_trust = _patched_registry_trust(base_registry_trust)
    patched = validate_event_eeg_findings_v3_payload(
        patched,
        trusted_producer_receipts=trusted_v3_producer_receipts,
        trusted_registry_bindings=patched_registry_trust,
    )

    bundle: dict[str, Any] = {
        "schema_version": EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_SCHEMA_VERSION,
        "method_id": EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_METHOD_ID,
        "event_id": str(base["event_id"]),
        "record_id": str(base["provenance"]["record_id"]),
        "canonical_signal_sha256": str(canonical["source_signal_sha256"]),
        "canonical_receipt_sha256": str(canonical["receipt_sha256"]),
        "analysis_interval_seconds": list(analysis_interval),
        "bridge_policy": bridge_policy.to_dict(),
        "bridge_policy_sha256": bridge_policy.sha256,
        "primitive_policy": primitive_policy.to_dict(),
        "primitive_policy_sha256": primitive_policy.sha256,
        "base_findings_v3_payload_sha256": _sha256(base),
        "proposal_producer_roster": producer_roster,
        "proposal_producer_roster_sha256": _sha256(producer_roster),
        "proposal_denominator_roster": denominator,
        "proposal_denominator_roster_sha256": _sha256(denominator),
        "canonical_query_roster": canonical_queries,
        "canonical_query_roster_sha256": _sha256(canonical_queries),
        "primitive_sidecar_receipt": deepcopy(sidecar),
        "primitive_sidecar_receipt_sha256": (
            None if sidecar is None else str(sidecar["receipt_sha256"])
        ),
        "patched_findings_v3_payload": patched,
        "patched_findings_v3_payload_sha256": _sha256(patched),
        "unit_registry_binding": deepcopy(
            EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_UNIT_REGISTRY_BINDING
        ),
        "firewall": deepcopy(_PROPOSAL_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
    }
    bundle["source_binding_sha256"] = _bundle_source_binding_sha256(bundle)
    bundle["receipt_sha256"] = _self_hash(bundle, "receipt_sha256")
    return validate_event_morphology_atomic_findings_v3_bridge_v1(
        bundle,
        trusted_v3_producer_receipts=trusted_v3_producer_receipts,
        trusted_v3_registry_bindings=base_registry_trust,
    )


def _bindings_from_roster(
    roster: Sequence[Mapping[str, Any]],
) -> tuple[list[EventMorphologyProposalProducerBinding], dict[str, dict[str, Any]]]:
    bindings: list[EventMorphologyProposalProducerBinding] = []
    by_id: dict[str, dict[str, Any]] = {}
    for raw in roster:
        receipt = _validate_producer_receipt(raw)
        producer_id = str(receipt["producer_id"])
        if producer_id in by_id:
            raise ValueError("proposal producer roster repeats a producer ID")
        by_id[producer_id] = receipt
        bindings.append(
            EventMorphologyProposalProducerBinding(
                producer_id=producer_id,
                producer_version=str(receipt["producer_version"]),
                source_kind=str(receipt["source_kind"]),
                artifact_sha256=str(receipt["artifact_sha256"]),
                code_sha256=str(receipt["code_sha256"]),
                policy_sha256=str(receipt["policy_sha256"]),
                input_contract_sha256=str(receipt["input_contract_sha256"]),
                model_weights_sha256=(
                    None
                    if receipt["model_weights_sha256"] is None
                    else str(receipt["model_weights_sha256"])
                ),
            )
        )
    expected, _ = _canonical_producer_roster(bindings)
    if list(roster) != expected:
        raise ValueError("proposal producer roster is not canonical")
    return bindings, by_id


def _proposals_from_denominator(
    roster: Sequence[Mapping[str, Any]],
) -> list[EventMorphologyAtomicProposal]:
    expected_keys = {
        "proposal_id",
        "producer_id",
        "producer_receipt_id",
        "producer_receipt_sha256",
        "source_kind",
        "view_id",
        "unit_id",
        "requested_recording_interval_seconds",
        "routing_score",
        "routing_score_semantics",
        "boundary_disposition",
        "clipped_recording_interval_seconds",
        "canonical_query_id",
    }
    result = []
    for index, row in enumerate(roster):
        if type(row) is not dict or set(row) != expected_keys:
            raise ValueError(f"proposal_denominator_roster[{index}] keys drifted")
        if row["routing_score_semantics"] != ("proposal_routing_only_not_signal_fact"):
            raise ValueError("proposal routing score gained fact semantics")
        result.append(
            EventMorphologyAtomicProposal(
                proposal_id=row["proposal_id"],
                producer_id=row["producer_id"],
                view_id=row["view_id"],
                unit_id=row["unit_id"],
                recording_interval_seconds=tuple(
                    row["requested_recording_interval_seconds"]
                ),
                routing_score=row["routing_score"],
            )
        )
    return result


def _validate_patch_alignment(
    *,
    payload: Mapping[str, Any],
    canonical_queries: Sequence[Mapping[str, Any]],
    sidecar: Mapping[str, Any] | None,
    bridge_policy: EventMorphologyAtomicBridgePolicy,
) -> None:
    if any(
        row["evidence_id"] == _PLACEHOLDER_EVIDENCE_ID for row in payload["findings"]
    ):
        raise ValueError("patched v3 retained the morphology placeholder")
    morphology_findings = [
        row for row in payload["findings"] if row["family"] == "morphology"
    ]
    morphology_opportunities = [
        row
        for row in payload["evaluation_opportunities"]
        if row["family"] == "morphology"
    ]
    if len(morphology_findings) != len(canonical_queries) or len(
        morphology_opportunities
    ) != len(canonical_queries):
        raise ValueError("patched morphology atom count does not match queries")
    if not canonical_queries:
        if sidecar is not None:
            raise ValueError("empty canonical roster cannot carry a sidecar")
        return
    if sidecar is None or len(sidecar["rows"]) != len(canonical_queries):
        raise ValueError("canonical queries and sidecar rows do not align")
    findings_by_id = {str(row["evidence_id"]): row for row in morphology_findings}
    opportunities_by_id = {
        str(row["evaluation_opportunity_id"]): row for row in morphology_opportunities
    }
    morphology_target_indices = {
        index
        for index, (_, _, family) in enumerate(EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS)
        if family == "morphology"
    }
    montage = {str(row["unit_id"]): row for row in payload["montage"]["input_units"]}
    for query, sidecar_row in zip(canonical_queries, sidecar["rows"]):
        query_id = str(query["canonical_query_id"])
        expected_evidence_id = (
            "E-MORPH-"
            + _sha256(
                {
                    "query_id": query_id,
                    "sidecar_row_binding_sha256": sidecar_row["row_binding_sha256"],
                }
            )[:20]
        )
        masks = [
            bool(value) for value in sidecar_row["opportunity"]["target_value_mask"]
        ]
        unit = montage.get(str(query["unit_id"]))
        montage_eligible = bool(
            unit is not None
            and unit["observation_status"] == "observed"
            and unit["evidence_eligible"]
        )
        morphology_available = any(masks[index] for index in morphology_target_indices)
        expected_status = (
            "not_evaluable"
            if not montage_eligible or not morphology_available
            else "present"
            if all(masks)
            else "uncertain"
        )
        finding = findings_by_id.get(expected_evidence_id)
        if finding is None or finding["status"] != expected_status:
            raise ValueError("patched morphology Finding status/identity drifted")
        expected_opportunity_id = (
            "OPP-MORPH-"
            + _sha256(
                {
                    "query_id": query_id,
                    "sidecar_row_binding_sha256": sidecar_row["row_binding_sha256"],
                    "status": expected_status,
                }
            )[:20]
        )
        if finding["evaluation_opportunity_id"] != expected_opportunity_id:
            raise ValueError("patched morphology opportunity identity drifted")
        opportunity = opportunities_by_id.get(expected_opportunity_id)
        if opportunity is None:
            raise ValueError("patched morphology opportunity is missing")
        if finding["finding_group_id"] != query["overlap_component_id"]:
            raise ValueError("patched morphology overlap component drifted")
        if expected_status == "not_evaluable":
            if finding["measurements"] or opportunity["status"] != "not_evaluable":
                raise ValueError("not-evaluable morphology query gained measurements")
            continue
        expected_measurements = [
            (index, name, unit_id, float(sidecar_row["values"][index]))
            for index, (name, unit_id, _) in enumerate(
                EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS
            )
            if masks[index]
        ]
        actual = finding["measurements"]
        if len(actual) != len(expected_measurements):
            raise ValueError("patched morphology measurement mask drifted")
        atom_policy_sha256 = _sha256(
            {
                "bridge_policy_sha256": bridge_policy.sha256,
                "primitive_policy_sha256": sidecar["policy_sha256"],
                "sidecar_receipt_sha256": sidecar["receipt_sha256"],
                "sidecar_row_binding_sha256": sidecar_row["row_binding_sha256"],
                "proposal_producer_receipt_sha256s": query[
                    "proposal_producer_receipt_sha256s"
                ],
            }
        )
        for measurement, (index, name, unit_id, value) in zip(
            actual, expected_measurements
        ):
            expected_measurement_id = (
                "MEAS-MORPH-"
                + _sha256(
                    {
                        "query_id": query_id,
                        "target_index": index,
                        "target_name": name,
                        "sidecar_row_binding_sha256": sidecar_row["row_binding_sha256"],
                    }
                )[:20]
            )
            binding = measurement["source_binding"]
            if (
                measurement["measurement_id"] != expected_measurement_id
                or measurement["name_id"] != name
                or measurement["unit_id"] != unit_id
                or float(measurement["value"]) != value
                or binding["evidence_family"] != "morphology"
                or binding["method_id"]
                != EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_METHOD_ID
                or binding["policy_sha256"] != atom_policy_sha256
                or binding["source_view_id"] != query["view_id"]
                or binding["source_unit_ids"] != [query["unit_id"]]
                or binding["recording_interval"]
                != sidecar_row["source_binding"]["recording_interval_seconds"]
            ):
                raise ValueError("patched morphology measurement drifted from sidecar")
            dependency = binding["raw_sample_dependency"]
            if (
                dependency["view_role"] != "canonical_physical_evidence"
                or dependency["dependency_status"] != "exact_instantaneous"
                or dependency["future_sample_access"] is not False
                or dependency["onset_evidence_authorized"] is not False
                or dependency["onset_support_eligible"] is not False
            ):
                raise ValueError("patched morphology dependency gained onset authority")


def validate_event_morphology_atomic_findings_v3_bridge_v1(
    value: object,
    *,
    trusted_v3_producer_receipts: Mapping[str, Mapping[str, object]],
    trusted_v3_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Validate hashes, canonical rosters, sidecar alignment, and patched v3."""

    if type(value) is not dict:
        raise TypeError("morphology Findings v3 bridge must be an object")
    bundle = deepcopy(value)
    expected_keys = {
        "schema_version",
        "method_id",
        "event_id",
        "record_id",
        "canonical_signal_sha256",
        "canonical_receipt_sha256",
        "analysis_interval_seconds",
        "bridge_policy",
        "bridge_policy_sha256",
        "primitive_policy",
        "primitive_policy_sha256",
        "base_findings_v3_payload_sha256",
        "proposal_producer_roster",
        "proposal_producer_roster_sha256",
        "proposal_denominator_roster",
        "proposal_denominator_roster_sha256",
        "canonical_query_roster",
        "canonical_query_roster_sha256",
        "primitive_sidecar_receipt",
        "primitive_sidecar_receipt_sha256",
        "patched_findings_v3_payload",
        "patched_findings_v3_payload_sha256",
        "unit_registry_binding",
        "firewall",
        "authorization",
        "source_binding_sha256",
        "receipt_sha256",
    }
    if set(bundle) != expected_keys:
        raise ValueError("morphology Findings v3 bridge keys drifted")
    if bundle["schema_version"] != (
        EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_SCHEMA_VERSION
    ) or bundle["method_id"] != (EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_METHOD_ID):
        raise ValueError("morphology Findings v3 bridge identity drifted")
    _identifier(bundle["event_id"], "event_id")
    _identifier(bundle["record_id"], "record_id")
    for field in (
        "canonical_signal_sha256",
        "canonical_receipt_sha256",
        "bridge_policy_sha256",
        "primitive_policy_sha256",
        "base_findings_v3_payload_sha256",
        "proposal_producer_roster_sha256",
        "proposal_denominator_roster_sha256",
        "canonical_query_roster_sha256",
        "patched_findings_v3_payload_sha256",
        "source_binding_sha256",
        "receipt_sha256",
    ):
        _hash(bundle[field], field)
    analysis = _interval(
        bundle["analysis_interval_seconds"], "analysis_interval_seconds"
    )
    policy_data = bundle["bridge_policy"]
    if not isinstance(policy_data, Mapping):
        raise TypeError("bridge_policy must be an object")
    bridge_policy = EventMorphologyAtomicBridgePolicy(
        minimum_clipped_duration_seconds=policy_data.get(
            "minimum_clipped_duration_seconds"
        ),
        duplicate_key_policy=policy_data.get("duplicate_key_policy"),
        overlap_policy=policy_data.get("overlap_policy"),
        mixed_authority_policy=policy_data.get("mixed_authority_policy"),
    )
    if policy_data != bridge_policy.to_dict() or (
        bundle["bridge_policy_sha256"] != bridge_policy.sha256
    ):
        raise ValueError("bridge policy content/hash drifted")
    primitive_data = bundle["primitive_policy"]
    if not isinstance(primitive_data, Mapping):
        raise TypeError("primitive_policy must be an object")
    primitive_policy = EventMorphologyPrimitivePolicy(
        minimum_samples=primitive_data.get("minimum_samples"),
        centering=primitive_data.get("centering"),
        crossing_interpolation=primitive_data.get("crossing_interpolation"),
        tie_breaking=primitive_data.get("tie_breaking"),
    )
    if primitive_data != primitive_policy.to_dict() or (
        bundle["primitive_policy_sha256"] != primitive_policy.sha256
    ):
        raise ValueError("primitive policy content/hash drifted")

    if not isinstance(bundle["proposal_producer_roster"], list):
        raise TypeError("proposal producer roster must be an array")
    _, producers_by_id = _bindings_from_roster(bundle["proposal_producer_roster"])
    if bundle["proposal_producer_roster_sha256"] != _sha256(
        bundle["proposal_producer_roster"]
    ):
        raise ValueError("proposal producer roster hash mismatch")
    if not isinstance(bundle["proposal_denominator_roster"], list):
        raise TypeError("proposal denominator roster must be an array")
    proposals = _proposals_from_denominator(bundle["proposal_denominator_roster"])
    expected_denominator, expected_queries = _canonicalize_proposals(
        event_id=str(bundle["event_id"]),
        proposals=proposals,
        producers_by_id=producers_by_id,
        analysis_interval=analysis,
        policy=bridge_policy,
    )
    if bundle["proposal_denominator_roster"] != expected_denominator:
        raise ValueError("proposal denominator disposition/order drifted")
    if bundle["proposal_denominator_roster_sha256"] != _sha256(expected_denominator):
        raise ValueError("proposal denominator roster hash mismatch")
    if bundle["canonical_query_roster"] != expected_queries:
        raise ValueError("canonical morphology query roster drifted")
    if bundle["canonical_query_roster_sha256"] != _sha256(expected_queries):
        raise ValueError("canonical morphology query roster hash mismatch")

    sidecar_value = bundle["primitive_sidecar_receipt"]
    if sidecar_value is None:
        if expected_queries or bundle["primitive_sidecar_receipt_sha256"] is not None:
            raise ValueError("missing sidecar is incompatible with canonical queries")
        sidecar = None
    else:
        sidecar = validate_event_morphology_primitive_supervision_v1(sidecar_value)
        if not expected_queries:
            raise ValueError("sidecar cannot exist without canonical queries")
        if bundle["primitive_sidecar_receipt_sha256"] != sidecar["receipt_sha256"]:
            raise ValueError("primitive sidecar receipt hash mismatch")
        expected_sidecar_queries = [
            {
                "view_id": row["view_id"],
                "unit_id": row["unit_id"],
                "recording_interval_seconds": row["recording_interval_seconds"],
                "query_authority": row["query_authority"],
            }
            for row in expected_queries
        ]
        if (
            sidecar["event_id"] != bundle["event_id"]
            or sidecar["analysis_interval_seconds"]
            != bundle["analysis_interval_seconds"]
            or sidecar["canonical_receipt_sha256"] != bundle["canonical_receipt_sha256"]
            or sidecar["source_signal_sha256"] != bundle["canonical_signal_sha256"]
            or sidecar["query_roster"] != expected_sidecar_queries
            or [row["query_id"] for row in sidecar["rows"]]
            != [row["canonical_query_id"] for row in expected_queries]
        ):
            raise ValueError("primitive sidecar source/query binding drifted")

    if bundle["unit_registry_binding"] != (
        EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_UNIT_REGISTRY_BINDING
    ):
        raise ValueError("bridge unit registry binding drifted")
    if bundle["firewall"] != _PROPOSAL_FIREWALL or (
        bundle["authorization"] != _AUTHORIZATION
    ):
        raise ValueError("bridge firewall/authorization drifted")
    patched_registry_trust = _patched_registry_trust(
        deepcopy(DEFAULT_EVENT_FINDINGS_V2_REGISTRY_BINDINGS)
        if trusted_v3_registry_bindings is None
        else trusted_v3_registry_bindings
    )
    patched = validate_event_eeg_findings_v3_payload(
        bundle["patched_findings_v3_payload"],
        trusted_producer_receipts=trusted_v3_producer_receipts,
        trusted_registry_bindings=patched_registry_trust,
    )
    if (
        patched["event_id"] != bundle["event_id"]
        or patched["provenance"]["record_id"] != bundle["record_id"]
        or patched["provenance"]["canonical_signal_sha256"]
        != bundle["canonical_signal_sha256"]
        or patched["window"]["final_interval"] != bundle["analysis_interval_seconds"]
    ):
        raise ValueError("patched v3 identity/coordinates drifted")
    if patched["registry_bindings"]["unit_registry"] != (
        EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_UNIT_REGISTRY_BINDING
    ):
        raise ValueError("patched v3 unit registry was not augmented")
    if bundle["patched_findings_v3_payload_sha256"] != _sha256(patched):
        raise ValueError("patched Findings v3 payload hash mismatch")
    _validate_patch_alignment(
        payload=patched,
        canonical_queries=expected_queries,
        sidecar=sidecar,
        bridge_policy=bridge_policy,
    )
    if bundle["source_binding_sha256"] != _bundle_source_binding_sha256(bundle):
        raise ValueError("bridge source binding hash mismatch")
    if bundle["receipt_sha256"] != _self_hash(bundle, "receipt_sha256"):
        raise ValueError("bridge receipt hash mismatch")
    return bundle


def replay_event_morphology_atomic_findings_v3_bridge_v1(
    receipt: object,
    *,
    base_findings_v3_payload: object,
    canonical_receipt: object,
    views: Sequence[EventMorphologyPrimitiveViewInput],
    proposal_producers: Sequence[EventMorphologyProposalProducerBinding],
    proposals: Sequence[EventMorphologyAtomicProposal],
    trusted_v3_producer_receipts: Mapping[str, Mapping[str, object]],
    trusted_v3_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
    bridge_policy: EventMorphologyAtomicBridgePolicy = (
        DEFAULT_EVENT_MORPHOLOGY_ATOMIC_BRIDGE_POLICY
    ),
    primitive_policy: EventMorphologyPrimitivePolicy = (
        DEFAULT_EVENT_MORPHOLOGY_PRIMITIVE_POLICY
    ),
) -> dict[str, Any]:
    """Replay the complete bridge and reject any signal/proposal drift."""

    validated = validate_event_morphology_atomic_findings_v3_bridge_v1(
        receipt,
        trusted_v3_producer_receipts=trusted_v3_producer_receipts,
        trusted_v3_registry_bindings=trusted_v3_registry_bindings,
    )
    expected = materialize_event_morphology_atomic_findings_v3_bridge_v1(
        base_findings_v3_payload=base_findings_v3_payload,
        canonical_receipt=canonical_receipt,
        views=views,
        proposal_producers=proposal_producers,
        proposals=proposals,
        trusted_v3_producer_receipts=trusted_v3_producer_receipts,
        trusted_v3_registry_bindings=trusted_v3_registry_bindings,
        trusted_parent_views=trusted_parent_views,
        bridge_policy=bridge_policy,
        primitive_policy=primitive_policy,
    )
    if validated != expected:
        raise ValueError("morphology Findings v3 bridge replay mismatch")
    return expected


__all__ = [
    "DEFAULT_EVENT_MORPHOLOGY_ATOMIC_BRIDGE_POLICY",
    "EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_METHOD_ID",
    "EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_POLICY_ID",
    "EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_BRIDGE_SCHEMA_VERSION",
    "EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_UNIT_CATALOG",
    "EVENT_MORPHOLOGY_ATOMIC_FINDINGS_V3_UNIT_REGISTRY_BINDING",
    "EventMorphologyAtomicBridgePolicy",
    "EventMorphologyAtomicProposal",
    "EventMorphologyProposalProducerBinding",
    "materialize_event_morphology_atomic_findings_v3_bridge_v1",
    "replay_event_morphology_atomic_findings_v3_bridge_v1",
    "validate_event_morphology_atomic_findings_v3_bridge_v1",
]
