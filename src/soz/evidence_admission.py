"""Typed evidence-admission policy for the frozen trustworthy-SOZ audit.

The module does not define or alter the v29 ranker.  It formalizes which
evidence families may affect localization or patient-facing facts after their
independent qualification decision.  The controlled binary gate below is a
policy-mechanism test only; it does not replace the native TUEV/TUSZ gates used
to assign the formal M/I/V statuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping

import torch


class EvidenceStatus(str, Enum):
    ADMITTED_RANKING_CARRIER = "ADMITTED_RANKING_CARRIER"
    ADMITTED_CLINICAL_CONCEPT = "ADMITTED_CLINICAL_CONCEPT"
    DESCRIPTION_ONLY = "DESCRIPTION_ONLY"
    FAIL_NATIVE = "FAIL_NATIVE"
    NO_GO = "NO_GO"
    NOT_QUALIFIED = "NOT_QUALIFIED"


class ReportPermission(str, Enum):
    CANDIDATE_PROVENANCE = "candidate_provenance"
    PATIENT_CONCEPT_FACT = "patient_concept_fact"
    WAVEFORM_DESCRIPTION = "waveform_description"
    NONE = "none"


_PERMISSIONS = {
    EvidenceStatus.ADMITTED_RANKING_CARRIER: (True, ReportPermission.CANDIDATE_PROVENANCE),
    EvidenceStatus.ADMITTED_CLINICAL_CONCEPT: (True, ReportPermission.PATIENT_CONCEPT_FACT),
    EvidenceStatus.DESCRIPTION_ONLY: (False, ReportPermission.WAVEFORM_DESCRIPTION),
    EvidenceStatus.FAIL_NATIVE: (False, ReportPermission.NONE),
    EvidenceStatus.NO_GO: (False, ReportPermission.NONE),
    EvidenceStatus.NOT_QUALIFIED: (False, ReportPermission.NONE),
}


@dataclass(frozen=True)
class EvidenceAdmissionReceipt:
    family: str
    status: EvidenceStatus
    localization_access: bool
    report_permission: ReportPermission
    qualification_source: str

    def validate(self) -> None:
        if not self.family or not self.qualification_source:
            raise ValueError("evidence receipt requires family and qualification source")
        expected = _PERMISSIONS[self.status]
        actual = (self.localization_access, self.report_permission)
        if actual != expected:
            raise ValueError(
                f"{self.family} permissions {actual!r} conflict with status {self.status.value}"
            )


def formal_v29_evidence_receipts() -> tuple[EvidenceAdmissionReceipt, ...]:
    """Return the frozen v65 evidence-access contract."""

    rows = (
        EvidenceAdmissionReceipt(
            "H_carrier",
            EvidenceStatus.ADMITTED_RANKING_CARRIER,
            True,
            ReportPermission.CANDIDATE_PROVENANCE,
            "frozen_v29_ranker",
        ),
        EvidenceAdmissionReceipt(
            "D_carrier",
            EvidenceStatus.ADMITTED_RANKING_CARRIER,
            True,
            ReportPermission.CANDIDATE_PROVENANCE,
            "frozen_v29_ranker",
        ),
        EvidenceAdmissionReceipt(
            "M_morphology",
            EvidenceStatus.FAIL_NATIVE,
            False,
            ReportPermission.NONE,
            "TUEV_native_qualification",
        ),
        EvidenceAdmissionReceipt(
            "I_ictal_involvement",
            EvidenceStatus.FAIL_NATIVE,
            False,
            ReportPermission.NONE,
            "TUSZ_native_qualification",
        ),
        EvidenceAdmissionReceipt(
            "V_learned_future",
            EvidenceStatus.NO_GO,
            False,
            ReportPermission.NONE,
            "target_free_future_qualification",
        ),
        EvidenceAdmissionReceipt(
            "V_direct_waveform",
            EvidenceStatus.DESCRIPTION_ONLY,
            False,
            ReportPermission.WAVEFORM_DESCRIPTION,
            "target_blind_waveform_descriptor_contract",
        ),
        EvidenceAdmissionReceipt(
            "uncertainty_proxy",
            EvidenceStatus.NOT_QUALIFIED,
            False,
            ReportPermission.NONE,
            "public_private_uncertainty_transport_audit",
        ),
    )
    for row in rows:
        row.validate()
    return rows


def apply_formal_v29_firewall(
    frozen_probability: torch.Tensor,
    injected_evidence: Mapping[str, object],
    receipts: tuple[EvidenceAdmissionReceipt, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Apply the frozen access policy without creating a new fusion route.

    Formal v29 already contains H and D.  Consequently, injected sidecar
    evidence is never fused here, including hypothetical payloads for H/D.
    The only releasable sidecar value is a target-blind direct waveform
    description; it remains separate from the returned ranking tensor.
    """

    if frozen_probability.ndim != 2 or frozen_probability.shape[1] != 19:
        raise ValueError("frozen v29 probability must have shape [N,19]")
    if not torch.isfinite(frozen_probability).all():
        raise ValueError("frozen v29 probability must be finite")
    policy = formal_v29_evidence_receipts() if receipts is None else receipts
    by_family = {row.family: row for row in policy}
    if len(by_family) != len(policy):
        raise ValueError("duplicate evidence family")
    for row in policy:
        row.validate()

    released: dict[str, object] = {}
    for family, value in injected_evidence.items():
        receipt = by_family.get(family)
        if receipt is None:
            continue
        if receipt.report_permission == ReportPermission.WAVEFORM_DESCRIPTION:
            released[family] = value
    return frozen_probability.clone(), released


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.959963984540054) -> float:
    if total <= 0 or successes < 0 or successes > total or z <= 0:
        raise ValueError("invalid binomial count or z")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (centre - radius) / denominator


@dataclass(frozen=True)
class ControlledQualificationPolicy:
    chance_level: float = 0.5
    minimum_coverage: float = 0.8

    def decide(
        self,
        *,
        native_successes: int,
        native_total: int,
        transport_successes: int,
        transport_total: int,
        coverage: float,
        shortcut_control_passed: bool,
        patient_semantic_claim: bool,
    ) -> EvidenceStatus:
        """Exercise the gate on a controlled binary task.

        This deliberately conservative policy requires both native and
        transported lower confidence bounds above chance.  It is used only to
        test controller behavior under known synthetic truth.
        """

        if not 0.0 <= coverage <= 1.0:
            raise ValueError("coverage must be in [0,1]")
        if not patient_semantic_claim:
            return EvidenceStatus.DESCRIPTION_ONLY
        if coverage < self.minimum_coverage or not shortcut_control_passed:
            return EvidenceStatus.NO_GO
        native_lower = wilson_lower_bound(native_successes, native_total)
        if native_lower <= self.chance_level:
            return EvidenceStatus.FAIL_NATIVE
        transport_lower = wilson_lower_bound(transport_successes, transport_total)
        if transport_lower <= self.chance_level:
            return EvidenceStatus.NO_GO
        return EvidenceStatus.ADMITTED_CLINICAL_CONCEPT
