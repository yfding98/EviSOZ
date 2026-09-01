"""Target-free producer for facts-locked event scalp phenotypes.

The producer converts already validated 60-second C-CAR19 EEG and
``FineTemporalEvidence`` into reportable *scalp-visible* facts.  It never
reads SOZ/private labels, never resolves a bipolar edge to one physical
endpoint, and never calls later visibility propagation ground truth.

Frequency and rhythm are emitted only when a frozen local spectral criterion
passes.  Artifact type and montage stability are intentionally left absent;
they require separate, validated evidence producers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Final

import torch

from .clinical_reporting import (
    EventScalpPhenotypeAbstention,
    EventScalpPhenotypeEvidence,
    EvidenceProvenanceReceipt,
)
from .fine_temporal_evidence import (
    FINE_STRIDE_SECONDS,
    FINE_SUSTAINED_WINDOWS,
    FINE_WINDOW_SECONDS,
    FineTemporalEvidence,
)
from .geometry import CHANNEL_INDEX, N_TCP_EDGES, TCP_20_EDGES


EVENT_PHENOTYPE_PRODUCER_SCHEMA_V1: Final[str] = (
    "soz_target_free_event_scalp_phenotype_producer_v1"
)
# Keep the historical public name bound to v1 so every existing materializer
# and sealed artifact remains exactly replayable. New temporal-evolution logic
# must opt in to the distinct v2 entry point and lineage.
EVENT_PHENOTYPE_PRODUCER_SCHEMA: Final[str] = EVENT_PHENOTYPE_PRODUCER_SCHEMA_V1
EVENT_PHENOTYPE_PRODUCER_SCHEMA_V2: Final[str] = (
    "soz_target_free_event_scalp_phenotype_producer_v2"
)
EVENT_PHENOTYPE_PRODUCER_SCHEMAS: Final[frozenset[str]] = frozenset(
    {EVENT_PHENOTYPE_PRODUCER_SCHEMA_V1, EVENT_PHENOTYPE_PRODUCER_SCHEMA_V2}
)
EVENT_PHENOTYPE_POLICY_V1: Final[str] = (
    "target_free_sustained_bipolar_change_then_local_spectral_gate_v1"
)
EVENT_PHENOTYPE_POLICY_V2: Final[str] = (
    "target_free_sustained_bipolar_change_then_temporal_only_spectral_gate_v2"
)
EARLIEST_EDGE_TIE_TOLERANCE_SEC: Final[float] = FINE_STRIDE_SECONDS
LATER_VISIBLE_MIN_DELAY_SEC: Final[float] = 1.0
RHYTHM_BASELINE_START_SEC: Final[float] = -4.0
RHYTHM_BASELINE_STOP_SEC: Final[float] = -1.0
RHYTHM_WINDOW_START_OFFSETS_SEC: Final[tuple[float, ...]] = (0.0, 0.25, 0.5)
# Precision-first reporting gate.  A three-bin peak must contain at least
# half of 1--30 Hz power; broadband ictal/noise changes remain reportable as
# signal changes but do not receive a rhythm or frequency phrase.
RHYTHM_MIN_CONCENTRATION: Final[float] = 0.50
RHYTHM_MIN_BASELINE_LIFT: Final[float] = 0.05
RHYTHM_MIN_BASELINE_FREQUENCY_SHIFT_HZ: Final[float] = 1.0
RHYTHM_MIN_BASELINE_LOG_RMS_LIFT: Final[float] = math.log(1.25)
RHYTHM_EVOLUTION_LOG_RMS_SPAN: Final[float] = math.log(1.25)
RHYTHM_EVOLUTION_FREQUENCY_SPAN_HZ: Final[float] = 1.0
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
EVENT_PHENOTYPE_REFERENCE_ARMS: Final[frozenset[str]] = frozenset(
    {"C-CAR19", "C-REF19"}
)


@dataclass(frozen=True)
class EventPhenotypeProducerIdentity:
    """Pseudonymous identity and lineage supplied by the caller."""

    patient_pseudonym: str
    event_pseudonym: str
    signal_artifact_sha256: str
    evidence_artifact_sha256: str
    extractor_model_id: str = "target-free-fine-temporal-evidence"
    extractor_model_version: str = EVENT_PHENOTYPE_PRODUCER_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("patient_pseudonym", self.patient_pseudonym),
            ("event_pseudonym", self.event_pseudonym),
            ("extractor_model_id", self.extractor_model_id),
            ("extractor_model_version", self.extractor_model_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for name, value in (
            ("signal_artifact_sha256", self.signal_artifact_sha256),
            ("evidence_artifact_sha256", self.evidence_artifact_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in _SHA256_CHARACTERS for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class EventPhenotypeProductionResult:
    """Reportable phenotype or an explicit target-free abstention."""

    phenotype: EventScalpPhenotypeEvidence | None
    status: str
    reason_codes: tuple[str, ...]
    detected_bipolar_edge_count: int
    abstention: EventScalpPhenotypeAbstention | None = None
    producer_schema: str = EVENT_PHENOTYPE_PRODUCER_SCHEMA

    def __post_init__(self) -> None:
        if self.producer_schema not in EVENT_PHENOTYPE_PRODUCER_SCHEMAS:
            raise ValueError("Unsupported event phenotype producer schema")
        if self.status not in {"reportable", "abstained"}:
            raise ValueError("Event phenotype status must be reportable or abstained")
        if type(self.detected_bipolar_edge_count) is not int or not (
            0 <= self.detected_bipolar_edge_count <= N_TCP_EDGES
        ):
            raise ValueError("detected_bipolar_edge_count is invalid")
        if (
            not isinstance(self.reason_codes, tuple)
            or len(set(self.reason_codes)) != len(self.reason_codes)
        ):
            raise ValueError("reason_codes must be a unique tuple")
        if self.status == "reportable":
            if not isinstance(self.phenotype, EventScalpPhenotypeEvidence):
                raise TypeError("A reportable result requires a phenotype")
            if self.reason_codes:
                raise ValueError("A reportable result cannot contain reason codes")
            if self.abstention is not None:
                raise ValueError("A reportable result cannot contain an abstention")
        else:
            if self.phenotype is not None or not self.reason_codes:
                raise ValueError("An abstained result requires reasons and no phenotype")
            # ``None`` keeps direct construction of the pre-existing result
            # schema backward compatible.  The formal producer always emits
            # the typed abstention needed by the report boundary.
            if self.abstention is not None:
                if not isinstance(self.abstention, EventScalpPhenotypeAbstention):
                    raise TypeError("abstention must be a typed event abstention")
                if self.abstention.reason_codes != self.reason_codes:
                    raise ValueError("Abstention reason codes disagree")
                if (
                    self.abstention.detected_bipolar_edge_count
                    != self.detected_bipolar_edge_count
                ):
                    raise ValueError("Abstention detected-edge count disagrees")

    @property
    def report_event(self) -> EventScalpPhenotypeEvidence | EventScalpPhenotypeAbstention:
        """Return the typed report input without fabricating phenotype fields."""

        if self.phenotype is not None:
            return self.phenotype
        if self.abstention is None:
            raise ValueError(
                "A legacy directly constructed abstention has no report receipt"
            )
        return self.abstention


@dataclass(frozen=True)
class _RhythmProfile:
    rhythm_state: str | None
    frequency_range_hz: tuple[float, float] | None


def _require_inputs(eeg: torch.Tensor, evidence: FineTemporalEvidence) -> None:
    if not isinstance(eeg, torch.Tensor) or tuple(eeg.shape) != (19, 12_000):
        raise ValueError("event EEG must have shape [19,12000]")
    if not eeg.is_floating_point() or eeg.requires_grad or not torch.isfinite(eeg).all():
        raise TypeError("event EEG must be detached finite floating point")
    if not isinstance(evidence, FineTemporalEvidence):
        raise TypeError("evidence must be FineTemporalEvidence")
    devices = {
        eeg.device,
        evidence.bipolar_change_detected.device,
        evidence.bipolar_change_latency_sec.device,
    }
    if len(devices) != 1:
        raise ValueError("EEG and temporal evidence must share one device")


def _target_free_receipt(
    identity: EventPhenotypeProducerIdentity,
    *,
    time_coordinate_semantics: str,
    reference_arm_id: str,
    evidence_generation_policy: str = EVENT_PHENOTYPE_POLICY_V1,
) -> EvidenceProvenanceReceipt:
    return EvidenceProvenanceReceipt(
        patient_pseudonym=identity.patient_pseudonym,
        event_pseudonym=identity.event_pseudonym,
        signal_artifact_sha256=identity.signal_artifact_sha256,
        evidence_artifact_sha256=identity.evidence_artifact_sha256,
        extractor_model_id=identity.extractor_model_id,
        extractor_model_version=identity.extractor_model_version,
        time_coordinate_semantics=time_coordinate_semantics,
        causal_prefix_safe=True,
        montages=(reference_arm_id,),
        evidence_generation_policy=evidence_generation_policy,
        soz_labels_used_for_event_evidence=False,
        private_labels_used_for_event_evidence=False,
    )


def _abstained_result(
    *,
    receipt: EvidenceProvenanceReceipt,
    reason_codes: tuple[str, ...],
    detected_bipolar_edge_count: int,
    producer_schema: str = EVENT_PHENOTYPE_PRODUCER_SCHEMA_V1,
) -> EventPhenotypeProductionResult:
    abstention = EventScalpPhenotypeAbstention(
        receipt=receipt,
        reason_codes=reason_codes,
        detected_bipolar_edge_count=detected_bipolar_edge_count,
    )
    return EventPhenotypeProductionResult(
        phenotype=None,
        status="abstained",
        reason_codes=reason_codes,
        detected_bipolar_edge_count=detected_bipolar_edge_count,
        abstention=abstention,
        producer_schema=producer_schema,
    )


def _edge_signal(eeg: torch.Tensor, edge_index: int) -> torch.Tensor:
    left, right = TCP_20_EDGES[edge_index]
    return eeg[CHANNEL_INDEX[left]] - eeg[CHANNEL_INDEX[right]]


def _window_spectrum(
    signal: torch.Tensor,
    *,
    start_sec: float,
    sfreq_hz: float,
) -> tuple[float, float, float]:
    """Return dominant Hz, three-bin concentration, and log RMS."""

    # The formal event tensor begins at -12 seconds relative to global t0.
    first = int(round((float(start_sec) + 12.0) * sfreq_hz))
    count = int(round(FINE_WINDOW_SECONDS * sfreq_hz))
    stop = first + count
    if first < 0 or stop > signal.numel():
        raise ValueError("Rhythm analysis window lies outside the 60-second event")
    segment = signal[first:stop].to(dtype=torch.float64)
    centered = segment - segment.mean()
    eps = torch.finfo(centered.dtype).eps
    log_rms = float((centered.square().mean().sqrt() + eps).log().item())
    taper = torch.hann_window(
        count,
        periodic=False,
        dtype=centered.dtype,
        device=centered.device,
    )
    power = torch.fft.rfft(centered * taper).abs().square()
    frequencies = torch.fft.rfftfreq(count, d=1.0 / sfreq_hz).to(centered.device)
    mask = (frequencies >= 1.0) & (frequencies <= 30.0)
    selected_power = power[mask]
    selected_frequency = frequencies[mask]
    total = selected_power.sum()
    if not bool(total > eps):
        return 0.0, 0.0, log_rms
    peak_index = int(selected_power.argmax().item())
    lower = max(0, peak_index - 1)
    upper = min(selected_power.numel(), peak_index + 2)
    concentration = float(
        (selected_power[lower:upper].sum() / total).item()
    )
    dominant = float(selected_frequency[peak_index].item())
    return dominant, concentration, log_rms


def _baseline_profile(
    signal: torch.Tensor, *, sfreq_hz: float
) -> tuple[float, float, float]:
    starts = torch.arange(
        RHYTHM_BASELINE_START_SEC,
        RHYTHM_BASELINE_STOP_SEC,
        FINE_STRIDE_SECONDS,
        dtype=torch.float64,
    )
    values = [
        _window_spectrum(signal, start_sec=float(start), sfreq_hz=sfreq_hz)
        for start in starts
    ]
    tensor = torch.tensor(values, dtype=torch.float64)
    medians = tensor.median(dim=0).values
    return (
        float(medians[0].item()),
        float(medians[1].item()),
        float(medians[2].item()),
    )


def _rhythm_profile(
    eeg: torch.Tensor,
    *,
    edge_indices: tuple[int, ...],
    earliest_latency_sec: float,
    sfreq_hz: float,
) -> _RhythmProfile:
    frequencies: list[float] = []
    concentrations: list[float] = []
    log_rms_by_offset: list[list[float]] = [
        [] for _ in RHYTHM_WINDOW_START_OFFSETS_SEC
    ]
    baseline_concentrations: list[float] = []
    baseline_frequencies: list[float] = []
    baseline_log_rms: list[float] = []
    for edge_index in edge_indices:
        signal = _edge_signal(eeg, edge_index)
        baseline_frequency, baseline_concentration, baseline_rms = _baseline_profile(
            signal, sfreq_hz=sfreq_hz
        )
        baseline_frequencies.append(baseline_frequency)
        baseline_concentrations.append(baseline_concentration)
        baseline_log_rms.append(baseline_rms)
        for offset_index, offset in enumerate(RHYTHM_WINDOW_START_OFFSETS_SEC):
            frequency, concentration, log_rms = _window_spectrum(
                signal,
                start_sec=earliest_latency_sec + offset,
                sfreq_hz=sfreq_hz,
            )
            frequencies.append(frequency)
            concentrations.append(concentration)
            log_rms_by_offset[offset_index].append(log_rms)

    concentration = float(
        torch.tensor(concentrations, dtype=torch.float64).median().item()
    )
    baseline = float(
        torch.tensor(baseline_concentrations, dtype=torch.float64).median().item()
    )
    valid_frequency = [value for value in frequencies if 1.0 <= value <= 30.0]
    if not valid_frequency:
        return _RhythmProfile(rhythm_state=None, frequency_range_hz=None)
    lower = float(min(valid_frequency))
    upper = float(max(valid_frequency))
    offset_rms = [
        float(torch.tensor(values, dtype=torch.float64).median().item())
        for values in log_rms_by_offset
    ]
    post_frequency = float(
        torch.tensor(valid_frequency, dtype=torch.float64).median().item()
    )
    baseline_frequency = float(
        torch.tensor(baseline_frequencies, dtype=torch.float64).median().item()
    )
    baseline_rms = float(
        torch.tensor(baseline_log_rms, dtype=torch.float64).median().item()
    )
    post_rms = float(torch.tensor(offset_rms, dtype=torch.float64).median().item())
    changed_from_baseline = (
        concentration - baseline >= RHYTHM_MIN_BASELINE_LIFT
        or abs(post_frequency - baseline_frequency)
        >= RHYTHM_MIN_BASELINE_FREQUENCY_SHIFT_HZ
        or post_rms - baseline_rms >= RHYTHM_MIN_BASELINE_LOG_RMS_LIFT
    )
    if concentration < RHYTHM_MIN_CONCENTRATION or not changed_from_baseline:
        return _RhythmProfile(rhythm_state=None, frequency_range_hz=None)
    rms_span = max(offset_rms) - min(offset_rms)
    frequency_span = upper - lower
    evolving = (
        rms_span >= RHYTHM_EVOLUTION_LOG_RMS_SPAN
        or frequency_span >= RHYTHM_EVOLUTION_FREQUENCY_SPAN_HZ
    )
    return _RhythmProfile(
        rhythm_state="evolving_rhythmic" if evolving else "rhythmic",
        frequency_range_hz=(lower, upper),
    )


def _rhythm_profile_v2(
    eeg: torch.Tensor,
    *,
    edge_indices: tuple[int, ...],
    earliest_latency_sec: float,
    sfreq_hz: float,
) -> _RhythmProfile:
    """Temporal-only rhythm variation for the versioned v2 producer.

    V1 pooled dominant frequencies over both edge and time dimensions before
    taking their span. Two simultaneously visible stationary edges with
    different frequencies could therefore be labelled ``evolving``. V2 first
    aggregates frequency across edges within each fixed offset and then takes
    a span only across offsets. Edge heterogeneity remains observable in the
    underlying evidence, but cannot manufacture temporal evolution.
    """

    frequencies_by_offset: list[list[float]] = [
        [] for _ in RHYTHM_WINDOW_START_OFFSETS_SEC
    ]
    concentrations: list[float] = []
    log_rms_by_offset: list[list[float]] = [
        [] for _ in RHYTHM_WINDOW_START_OFFSETS_SEC
    ]
    baseline_concentrations: list[float] = []
    baseline_frequencies: list[float] = []
    baseline_log_rms: list[float] = []
    for edge_index in edge_indices:
        signal = _edge_signal(eeg, edge_index)
        baseline_frequency, baseline_concentration, baseline_rms = _baseline_profile(
            signal, sfreq_hz=sfreq_hz
        )
        baseline_frequencies.append(baseline_frequency)
        baseline_concentrations.append(baseline_concentration)
        baseline_log_rms.append(baseline_rms)
        for offset_index, offset in enumerate(RHYTHM_WINDOW_START_OFFSETS_SEC):
            frequency, concentration, log_rms = _window_spectrum(
                signal,
                start_sec=earliest_latency_sec + offset,
                sfreq_hz=sfreq_hz,
            )
            frequencies_by_offset[offset_index].append(frequency)
            concentrations.append(concentration)
            log_rms_by_offset[offset_index].append(log_rms)

    concentration = float(
        torch.tensor(concentrations, dtype=torch.float64).median().item()
    )
    baseline = float(
        torch.tensor(baseline_concentrations, dtype=torch.float64).median().item()
    )
    offset_frequencies: list[float] = []
    for values in frequencies_by_offset:
        valid = [value for value in values if 1.0 <= value <= 30.0]
        if valid:
            offset_frequencies.append(
                float(torch.tensor(valid, dtype=torch.float64).median().item())
            )
    if not offset_frequencies:
        return _RhythmProfile(rhythm_state=None, frequency_range_hz=None)
    lower = float(min(offset_frequencies))
    upper = float(max(offset_frequencies))
    offset_rms = [
        float(torch.tensor(values, dtype=torch.float64).median().item())
        for values in log_rms_by_offset
    ]
    post_frequency = float(
        torch.tensor(offset_frequencies, dtype=torch.float64).median().item()
    )
    baseline_frequency = float(
        torch.tensor(baseline_frequencies, dtype=torch.float64).median().item()
    )
    baseline_rms = float(
        torch.tensor(baseline_log_rms, dtype=torch.float64).median().item()
    )
    post_rms = float(torch.tensor(offset_rms, dtype=torch.float64).median().item())
    changed_from_baseline = (
        concentration - baseline >= RHYTHM_MIN_BASELINE_LIFT
        or abs(post_frequency - baseline_frequency)
        >= RHYTHM_MIN_BASELINE_FREQUENCY_SHIFT_HZ
        or post_rms - baseline_rms >= RHYTHM_MIN_BASELINE_LOG_RMS_LIFT
    )
    if concentration < RHYTHM_MIN_CONCENTRATION or not changed_from_baseline:
        return _RhythmProfile(rhythm_state=None, frequency_range_hz=None)
    rms_span = max(offset_rms) - min(offset_rms)
    frequency_span = upper - lower
    evolving = (
        rms_span >= RHYTHM_EVOLUTION_LOG_RMS_SPAN
        or frequency_span >= RHYTHM_EVOLUTION_FREQUENCY_SPAN_HZ
    )
    return _RhythmProfile(
        rhythm_state="evolving_rhythmic" if evolving else "rhythmic",
        frequency_range_hz=(lower, upper),
    )


def _produce_event_scalp_phenotype(
    eeg: torch.Tensor,
    evidence: FineTemporalEvidence,
    *,
    identity: EventPhenotypeProducerIdentity,
    event_anchor_coordinate_sec: float,
    time_coordinate_semantics: str,
    sfreq_hz: float = 200.0,
    reference_arm_id: str = "C-CAR19",
    producer_schema: str,
    evidence_generation_policy: str,
    rhythm_profile: Callable[..., _RhythmProfile],
) -> EventPhenotypeProductionResult:
    _require_inputs(eeg, evidence)
    if not isinstance(identity, EventPhenotypeProducerIdentity):
        raise TypeError("identity must be EventPhenotypeProducerIdentity")
    if identity.extractor_model_version != producer_schema:
        raise ValueError("identity producer version disagrees with entry point")
    if not math.isfinite(float(sfreq_hz)) or abs(float(sfreq_hz) - 200.0) > 1e-9:
        raise ValueError("event phenotype producer is frozen at 200 Hz")
    if not math.isfinite(float(event_anchor_coordinate_sec)):
        raise ValueError("event_anchor_coordinate_sec must be finite")
    if time_coordinate_semantics not in {
        "recording_start_seconds",
        "event_window_start_seconds",
    }:
        raise ValueError("Unsupported time_coordinate_semantics")
    if reference_arm_id not in EVENT_PHENOTYPE_REFERENCE_ARMS:
        raise ValueError("Unsupported event-phenotype reference arm")
    receipt = _target_free_receipt(
        identity,
        time_coordinate_semantics=time_coordinate_semantics,
        reference_arm_id=reference_arm_id,
        evidence_generation_policy=evidence_generation_policy,
    )

    detected = evidence.bipolar_change_detected.detach()
    count = int(detected.sum().item())
    if count == 0:
        return _abstained_result(
            receipt=receipt,
            reason_codes=("no_sustained_bipolar_change",),
            detected_bipolar_edge_count=0,
            producer_schema=producer_schema,
        )
    latencies = evidence.bipolar_change_latency_sec.detach()
    earliest_latency = float(latencies[detected].amin().item())
    earliest_mask = detected & (
        latencies <= earliest_latency + EARLIEST_EDGE_TIE_TOLERANCE_SEC + 1e-9
    )
    earliest_indices = tuple(
        int(value) for value in torch.where(earliest_mask)[0].cpu().tolist()
    )
    earliest_edges = tuple(
        f"{TCP_20_EDGES[index][0]}-{TCP_20_EDGES[index][1]}"
        for index in earliest_indices
    )

    later_mask = detected & (
        latencies >= earliest_latency + LATER_VISIBLE_MIN_DELAY_SEC - 1e-9
    )
    later_delay: float | None = None
    later_edges: tuple[str, ...] = ()
    if bool(later_mask.any()):
        first_later = float(latencies[later_mask].amin().item())
        tied_later = later_mask & (
            latencies <= first_later + EARLIEST_EDGE_TIE_TOLERANCE_SEC + 1e-9
        )
        later_indices = tuple(
            int(value) for value in torch.where(tied_later)[0].cpu().tolist()
        )
        later_edges = tuple(
            f"{TCP_20_EDGES[index][0]}-{TCP_20_EDGES[index][1]}"
            for index in later_indices
        )
        later_delay = first_later - earliest_latency

    rhythm = rhythm_profile(
        eeg,
        edge_indices=earliest_indices,
        earliest_latency_sec=earliest_latency,
        sfreq_hz=float(sfreq_hz),
    )
    support_duration = FINE_WINDOW_SECONDS + (
        FINE_SUSTAINED_WINDOWS - 1
    ) * FINE_STRIDE_SECONDS
    onset_start = float(event_anchor_coordinate_sec) + earliest_latency
    onset_end = onset_start + support_duration
    if onset_start < 0:
        return _abstained_result(
            receipt=receipt,
            reason_codes=("negative_report_coordinate",),
            detected_bipolar_edge_count=count,
            producer_schema=producer_schema,
        )
    phenotype = EventScalpPhenotypeEvidence(
        receipt=receipt,
        onset_start_sec=onset_start,
        onset_end_sec=onset_end,
        first_visible_derivations=earliest_edges,
        rhythm_state=rhythm.rhythm_state,
        frequency_range_hz=rhythm.frequency_range_hz,
        later_visible_delay_sec=later_delay,
        later_visible_derivations=later_edges,
        montage_stability=None,
        artifact_assessed=None,
    )
    return EventPhenotypeProductionResult(
        phenotype=phenotype,
        status="reportable",
        reason_codes=(),
        detected_bipolar_edge_count=count,
        producer_schema=producer_schema,
    )


def produce_event_scalp_phenotype(
    eeg: torch.Tensor,
    evidence: FineTemporalEvidence,
    *,
    identity: EventPhenotypeProducerIdentity,
    event_anchor_coordinate_sec: float,
    time_coordinate_semantics: str,
    sfreq_hz: float = 200.0,
    reference_arm_id: str = "C-CAR19",
) -> EventPhenotypeProductionResult:
    """Replay the historical v1 target-free event-phenotype producer.

    ``event_anchor_coordinate_sec`` is a global seizure annotation time, not
    cortical SOZ onset. V1 remains frozen for sealed-artifact reproducibility;
    new work should use the explicitly versioned v2 entry point below.
    """

    return _produce_event_scalp_phenotype(
        eeg,
        evidence,
        identity=identity,
        event_anchor_coordinate_sec=event_anchor_coordinate_sec,
        time_coordinate_semantics=time_coordinate_semantics,
        sfreq_hz=sfreq_hz,
        reference_arm_id=reference_arm_id,
        producer_schema=EVENT_PHENOTYPE_PRODUCER_SCHEMA_V1,
        evidence_generation_policy=EVENT_PHENOTYPE_POLICY_V1,
        rhythm_profile=_rhythm_profile,
    )


def produce_event_scalp_phenotype_v2(
    eeg: torch.Tensor,
    evidence: FineTemporalEvidence,
    *,
    identity: EventPhenotypeProducerIdentity,
    event_anchor_coordinate_sec: float,
    time_coordinate_semantics: str,
    sfreq_hz: float = 200.0,
    reference_arm_id: str = "C-CAR19",
) -> EventPhenotypeProductionResult:
    """Produce v2 evidence with temporal-only rhythm-evolution logic.

    This remains target/private/score free. The corrected engineering state is
    not a clinically validated temporal-evolution label and must stay
    conservatively worded until a label-fresh reader study qualifies it.
    """

    return _produce_event_scalp_phenotype(
        eeg,
        evidence,
        identity=identity,
        event_anchor_coordinate_sec=event_anchor_coordinate_sec,
        time_coordinate_semantics=time_coordinate_semantics,
        sfreq_hz=sfreq_hz,
        reference_arm_id=reference_arm_id,
        producer_schema=EVENT_PHENOTYPE_PRODUCER_SCHEMA_V2,
        evidence_generation_policy=EVENT_PHENOTYPE_POLICY_V2,
        rhythm_profile=_rhythm_profile_v2,
    )


__all__ = [
    "EARLIEST_EDGE_TIE_TOLERANCE_SEC",
    "EVENT_PHENOTYPE_REFERENCE_ARMS",
    "EVENT_PHENOTYPE_PRODUCER_SCHEMA",
    "EVENT_PHENOTYPE_PRODUCER_SCHEMA_V1",
    "EVENT_PHENOTYPE_PRODUCER_SCHEMA_V2",
    "EVENT_PHENOTYPE_POLICY_V1",
    "EVENT_PHENOTYPE_POLICY_V2",
    "EventPhenotypeProducerIdentity",
    "EventPhenotypeProductionResult",
    "LATER_VISIBLE_MIN_DELAY_SEC",
    "produce_event_scalp_phenotype",
    "produce_event_scalp_phenotype_v2",
]
