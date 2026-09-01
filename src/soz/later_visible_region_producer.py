"""Deterministic target-free mapping of later-visible TCP edges to region text.

The input to this module is limited to already observed common-TCP bipolar
derivations.  It does not read EEG, SOZ labels, private labels, localization
scores, or propagation labels.  A bipolar edge is never resolved to one of
its physical endpoints: cross-region support is reported as ``multi-region``
and cross-laterality support remains explicitly side-uncertain.

This producer is intentionally independent of the event/report adapters.  A
validated result can be bound to typed event facts in a separate step; this
module itself cannot alter a patient SOZ ranking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Final

from .clinical_reporting import (
    CLINICAL_SCALP_REGIONS,
    EvidenceProvenanceReceipt,
    LATERALITY_GROUPS,
    LATER_VISIBLE_REGIONS_ZH,
)
from .geometry import TCP_20_EDGES


LATER_VISIBLE_REGION_PRODUCER_SCHEMA: Final[str] = (
    "soz_target_free_later_visible_edge_to_region_v1"
)
LATER_VISIBLE_REGION_USE_POLICY: Final[str] = (
    "event_report_fact_or_abstention_only_not_propagation_or_soz_scoring"
)
LATER_VISIBLE_REGION_RECEIPT_SCHEMA: Final[str] = (
    "soz_target_free_later_visible_region_receipt_v1"
)
LATER_VISIBLE_EVENT_CORE_HASH_SEMANTICS: Final[str] = (
    "sha256_event_evidence_receipt_excluding_optional_reference_adapters_v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_REGION_ZH: Final[dict[str, str]] = {
    "frontal": "额区",
    "temporal": "颞区",
    "central": "中央区",
    "parietal": "顶区",
    "occipital": "枕区",
}
_LATERALITY_ZH: Final[dict[str, str]] = {
    "left": "左",
    "right": "右",
    "midline": "中线",
    "bilateral": "双侧",
    "uncertain": "侧别不确定的",
}
_REGION_ORDER: Final[dict[str, int]] = {
    value: index for index, value in enumerate(CLINICAL_SCALP_REGIONS)
}
_LATERALITY_ORDER: Final[dict[str, int]] = {
    value: index for index, value in enumerate(("left", "right", "midline"))
}
_CHANNEL_TO_REGION: Final[dict[str, str]] = {
    channel: region
    for region, channels in CLINICAL_SCALP_REGIONS.items()
    for channel in channels
}
_CHANNEL_TO_LATERALITY: Final[dict[str, str]] = {
    channel: laterality
    for laterality, channels in LATERALITY_GROUPS.items()
    for channel in channels
}
_CANONICAL_EDGE_BY_TEXT: Final[dict[str, tuple[int, tuple[str, str]]]] = {
    text: (index, edge)
    for index, edge in enumerate(TCP_20_EDGES)
    for text in (f"{edge[0]}-{edge[1]}", f"{edge[1]}-{edge[0]}")
}


@dataclass(frozen=True)
class LaterVisibleRegionProductionResult:
    """Closed-vocabulary region fact or an explicit deterministic abstention."""

    observed_derivations: tuple[str, ...]
    canonical_derivations: tuple[str, ...]
    later_visible_region_zh: str | None
    status: str
    reason_codes: tuple[str, ...]
    support_regions: tuple[str, ...]
    support_lateralities: tuple[str, ...]
    contains_cross_region_edge: bool
    contains_cross_laterality_edge: bool
    target_labels_used: bool = False
    private_data_used: bool = False
    propagation_labels_used: bool = False
    localization_scores_used: bool = False
    training_performed: bool = False
    use_policy: str = LATER_VISIBLE_REGION_USE_POLICY
    producer_schema: str = LATER_VISIBLE_REGION_PRODUCER_SCHEMA

    def __post_init__(self) -> None:
        if self.status not in {"mapped", "abstained"}:
            raise ValueError("later-visible region status must be mapped or abstained")
        if self.producer_schema != LATER_VISIBLE_REGION_PRODUCER_SCHEMA:
            raise ValueError("Unsupported later-visible region producer schema")
        if self.use_policy != LATER_VISIBLE_REGION_USE_POLICY:
            raise ValueError("Unsupported later-visible region use policy")
        for name in (
            "contains_cross_region_edge",
            "contains_cross_laterality_edge",
            "target_labels_used",
            "private_data_used",
            "propagation_labels_used",
            "localization_scores_used",
            "training_performed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        for name in (
            "target_labels_used",
            "private_data_used",
            "propagation_labels_used",
            "localization_scores_used",
            "training_performed",
        ):
            if getattr(self, name):
                raise ValueError(
                    "Later-visible region production must remain target/private/"
                    "propagation/score free and deterministic"
                )
        if (
            not isinstance(self.observed_derivations, tuple)
            or not isinstance(self.canonical_derivations, tuple)
            or not isinstance(self.reason_codes, tuple)
            or not isinstance(self.support_regions, tuple)
            or not isinstance(self.support_lateralities, tuple)
        ):
            raise TypeError("Later-visible region collections must be tuples")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("reason_codes must be unique")
        if any(value not in _REGION_ZH for value in self.support_regions):
            raise ValueError("support_regions contains an unknown region")
        if any(
            value not in {"left", "right", "midline"}
            for value in self.support_lateralities
        ):
            raise ValueError("support_lateralities contains an unknown laterality")
        canonical_edges = _canonicalize_derivations(self.observed_derivations)
        expected_canonical = tuple(
            f"{left}-{right}" for left, right in canonical_edges
        )
        if self.canonical_derivations != expected_canonical:
            raise ValueError("canonical_derivations disagree with observed TCP support")
        if not canonical_edges:
            expected_empty = (
                self.status == "abstained"
                and self.later_visible_region_zh is None
                and self.reason_codes
                == ("no_observed_later_visible_derivations",)
                and self.support_regions == ()
                and self.support_lateralities == ()
                and not self.contains_cross_region_edge
                and not self.contains_cross_laterality_edge
            )
            if not expected_empty:
                raise ValueError("Empty later-visible support must use frozen abstention")
            return

        (
            expected_region,
            expected_support_regions,
            expected_support_lateralities,
            expected_cross_region,
            expected_cross_laterality,
        ) = _derive_region_fields(canonical_edges)
        if self.status != "mapped" or self.reason_codes:
            raise ValueError("Observed later-visible TCP support must produce a mapping")
        if self.later_visible_region_zh not in LATER_VISIBLE_REGIONS_ZH:
            raise ValueError("Mapped result must use the frozen region vocabulary")
        if (
            self.later_visible_region_zh != expected_region
            or self.support_regions != expected_support_regions
            or self.support_lateralities != expected_support_lateralities
            or self.contains_cross_region_edge != expected_cross_region
            or self.contains_cross_laterality_edge != expected_cross_laterality
        ):
            raise ValueError(
                "Later-visible region fields disagree with deterministic TCP mapping"
            )


@dataclass(frozen=True)
class LaterVisibleRegionReceipt:
    """Bind one mapped region fact to its exact event-evidence receipt."""

    patient_pseudonym: str
    event_pseudonym: str
    evidence_artifact_sha256: str
    source_event_receipt_sha256: str
    observed_derivations: tuple[str, ...]
    canonical_derivations: tuple[str, ...]
    later_visible_region_zh: str
    support_regions: tuple[str, ...]
    support_lateralities: tuple[str, ...]
    contains_cross_region_edge: bool
    contains_cross_laterality_edge: bool
    target_labels_used: bool = False
    private_data_used: bool = False
    propagation_labels_used: bool = False
    localization_scores_used: bool = False
    training_performed: bool = False
    use_policy: str = LATER_VISIBLE_REGION_USE_POLICY
    producer_schema: str = LATER_VISIBLE_REGION_PRODUCER_SCHEMA
    schema_version: str = LATER_VISIBLE_REGION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("patient_pseudonym", self.patient_pseudonym),
            ("event_pseudonym", self.event_pseudonym),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        for name, value in (
            ("evidence_artifact_sha256", self.evidence_artifact_sha256),
            ("source_event_receipt_sha256", self.source_event_receipt_sha256),
        ):
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in (
            "contains_cross_region_edge",
            "contains_cross_laterality_edge",
            "target_labels_used",
            "private_data_used",
            "propagation_labels_used",
            "localization_scores_used",
            "training_performed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        for name in (
            "target_labels_used",
            "private_data_used",
            "propagation_labels_used",
            "localization_scores_used",
            "training_performed",
        ):
            if getattr(self, name):
                raise ValueError(
                    "Later-visible region receipt must remain target/private/"
                    "propagation/score free"
                )
        if self.use_policy != LATER_VISIBLE_REGION_USE_POLICY:
            raise ValueError("Unsupported later-visible region receipt use policy")
        if self.producer_schema != LATER_VISIBLE_REGION_PRODUCER_SCHEMA:
            raise ValueError("Unsupported later-visible region producer schema")
        if self.schema_version != LATER_VISIBLE_REGION_RECEIPT_SCHEMA:
            raise ValueError("Unsupported later-visible region receipt schema")
        replay = produce_later_visible_region(self.observed_derivations)
        if replay.status != "mapped" or replay.later_visible_region_zh is None:
            raise ValueError("A region receipt requires mapped observed support")
        expected = (
            replay.canonical_derivations,
            replay.later_visible_region_zh,
            replay.support_regions,
            replay.support_lateralities,
            replay.contains_cross_region_edge,
            replay.contains_cross_laterality_edge,
        )
        actual = (
            self.canonical_derivations,
            self.later_visible_region_zh,
            self.support_regions,
            self.support_lateralities,
            self.contains_cross_region_edge,
            self.contains_cross_laterality_edge,
        )
        if actual != expected:
            raise ValueError("Later-visible region receipt does not replay")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def event_evidence_core_sha256(receipt: EvidenceProvenanceReceipt) -> str:
    """Hash event evidence independently of later report-only adapters."""

    if not isinstance(receipt, EvidenceProvenanceReceipt):
        raise TypeError("receipt must be EvidenceProvenanceReceipt")
    payload = asdict(receipt)
    for name in (
        "montages",
        "reference_pair_schema_version",
        "reference_pair_role",
        "reference_primary_arm_id",
        "reference_sensitivity_arm_id",
        "reference_disagreement_metric_id",
        "reference_disagreement_receipt_sha256",
    ):
        payload.pop(name)
    return _canonical_sha256(
        {
            "hash_semantics": LATER_VISIBLE_EVENT_CORE_HASH_SEMANTICS,
            "event_evidence_core": payload,
        }
    )


def _canonicalize_derivations(
    observed_derivations: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(observed_derivations, tuple):
        raise TypeError("observed_derivations must be a tuple")
    indexed: list[tuple[int, tuple[str, str]]] = []
    seen: set[int] = set()
    for value in observed_derivations:
        if not isinstance(value, str) or value not in _CANONICAL_EDGE_BY_TEXT:
            raise ValueError(
                "observed_derivations must contain only common-TCP derivations"
            )
        edge_index, edge = _CANONICAL_EDGE_BY_TEXT[value]
        if edge_index in seen:
            raise ValueError(
                "observed_derivations contains a duplicate undirected TCP edge"
            )
        seen.add(edge_index)
        indexed.append((edge_index, edge))
    return tuple(edge for _, edge in sorted(indexed))


def _aggregate_laterality(edge_sets: tuple[frozenset[str], ...]) -> str:
    """Return a side phrase without resolving a cross-side edge endpoint."""

    if all(value == frozenset({"left"}) for value in edge_sets):
        return "left"
    if all(value == frozenset({"right"}) for value in edge_sets):
        return "right"
    if all(value == frozenset({"midline"}) for value in edge_sets):
        return "midline"

    # Bilateral is licensed only by separately lateralized support on both
    # sides.  A single edge crossing a side/midline boundary is not treated as
    # two independently involved sides.
    if all(len(value) == 1 for value in edge_sets):
        singleton_sides = {next(iter(value)) for value in edge_sets}
        if "left" in singleton_sides and "right" in singleton_sides:
            return "bilateral"
    return "uncertain"


def _derive_region_fields(
    canonical_edges: tuple[tuple[str, str], ...],
) -> tuple[str, tuple[str, ...], tuple[str, ...], bool, bool]:
    if not canonical_edges:
        raise ValueError("Region fields require at least one canonical TCP edge")
    edge_regions = tuple(
        frozenset((_CHANNEL_TO_REGION[left], _CHANNEL_TO_REGION[right]))
        for left, right in canonical_edges
    )
    edge_lateralities = tuple(
        frozenset(
            (_CHANNEL_TO_LATERALITY[left], _CHANNEL_TO_LATERALITY[right])
        )
        for left, right in canonical_edges
    )
    support_regions = tuple(
        sorted(
            set().union(*edge_regions),
            key=_REGION_ORDER.__getitem__,
        )
    )
    support_lateralities = tuple(
        sorted(
            set().union(*edge_lateralities),
            key=_LATERALITY_ORDER.__getitem__,
        )
    )
    region_zh = (
        _REGION_ZH[support_regions[0]]
        if len(support_regions) == 1
        else "多区域"
    )
    laterality = _aggregate_laterality(edge_lateralities)
    region_token = f"{_LATERALITY_ZH[laterality]}{region_zh}"
    if region_token not in LATER_VISIBLE_REGIONS_ZH:
        raise RuntimeError("Derived region token escaped the frozen vocabulary")
    return (
        region_token,
        support_regions,
        support_lateralities,
        any(len(value) > 1 for value in edge_regions),
        any(len(value) > 1 for value in edge_lateralities),
    )


def produce_later_visible_region(
    observed_derivations: tuple[str, ...],
) -> LaterVisibleRegionProductionResult:
    """Map observed later-visible common-TCP edges without endpoint guessing.

    Cross-region support produces a ``多区域`` token.  Cross-laterality edges
    produce a ``侧别不确定的`` token.  Distinct, individually unilateral edges
    on both hemispheres can produce ``双侧``.  Empty support yields ``None``
    plus a frozen reason code; malformed/non-TCP support is rejected.
    """

    canonical_edges = _canonicalize_derivations(observed_derivations)
    canonical_text = tuple(f"{left}-{right}" for left, right in canonical_edges)
    if not canonical_edges:
        return LaterVisibleRegionProductionResult(
            observed_derivations=observed_derivations,
            canonical_derivations=(),
            later_visible_region_zh=None,
            status="abstained",
            reason_codes=("no_observed_later_visible_derivations",),
            support_regions=(),
            support_lateralities=(),
            contains_cross_region_edge=False,
            contains_cross_laterality_edge=False,
        )

    (
        region_token,
        support_regions,
        support_lateralities,
        cross_region,
        cross_laterality,
    ) = _derive_region_fields(canonical_edges)

    return LaterVisibleRegionProductionResult(
        observed_derivations=observed_derivations,
        canonical_derivations=canonical_text,
        later_visible_region_zh=region_token,
        status="mapped",
        reason_codes=(),
        support_regions=support_regions,
        support_lateralities=support_lateralities,
        contains_cross_region_edge=cross_region,
        contains_cross_laterality_edge=cross_laterality,
    )


def build_later_visible_region_receipt(
    production: LaterVisibleRegionProductionResult,
    event_receipt: EvidenceProvenanceReceipt,
) -> LaterVisibleRegionReceipt:
    """Bind a mapped deterministic production to one exact event receipt."""

    if not isinstance(production, LaterVisibleRegionProductionResult):
        raise TypeError("production must be LaterVisibleRegionProductionResult")
    if not isinstance(event_receipt, EvidenceProvenanceReceipt):
        raise TypeError("event_receipt must be EvidenceProvenanceReceipt")
    if production.status != "mapped" or production.later_visible_region_zh is None:
        raise ValueError("Only a mapped later-visible region can be receipted")
    return LaterVisibleRegionReceipt(
        patient_pseudonym=event_receipt.patient_pseudonym,
        event_pseudonym=event_receipt.event_pseudonym,
        evidence_artifact_sha256=event_receipt.evidence_artifact_sha256,
        source_event_receipt_sha256=event_evidence_core_sha256(event_receipt),
        observed_derivations=production.observed_derivations,
        canonical_derivations=production.canonical_derivations,
        later_visible_region_zh=production.later_visible_region_zh,
        support_regions=production.support_regions,
        support_lateralities=production.support_lateralities,
        contains_cross_region_edge=production.contains_cross_region_edge,
        contains_cross_laterality_edge=production.contains_cross_laterality_edge,
    )


if set(_CHANNEL_TO_REGION) != set(_CHANNEL_TO_LATERALITY):
    raise RuntimeError("Region and laterality ontologies cover different channels")
if set(_REGION_ZH) != set(CLINICAL_SCALP_REGIONS):
    raise RuntimeError("Chinese region terms do not cover the clinical ontology")


__all__ = [
    "LATER_VISIBLE_EVENT_CORE_HASH_SEMANTICS",
    "LATER_VISIBLE_REGION_PRODUCER_SCHEMA",
    "LATER_VISIBLE_REGION_RECEIPT_SCHEMA",
    "LATER_VISIBLE_REGION_USE_POLICY",
    "LaterVisibleRegionProductionResult",
    "LaterVisibleRegionReceipt",
    "build_later_visible_region_receipt",
    "event_evidence_core_sha256",
    "produce_later_visible_region",
]
