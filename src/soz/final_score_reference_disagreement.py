"""Typed C-CAR19/C-REF19 disagreement receipts for final SOZ scores.

This module is intentionally separate from :mod:`reference_disagreement`.
That older receipt measures a distance between frozen LaBraM block-9 node
representations.  The receipt below instead measures sensitivity of the
*same frozen localizer's final patient-level fixed-18 candidate scores*.
Neither receipt may stand in for the other.

The loader consumes only the sealed, target-excluding MRSC artifact.  It
requires externally supplied file digests, validates the complete
patient/event roster and tensor vocabulary, and replays the score metric.  It
does not load SOZ targets or private data, train, calibrate, or rerank.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
from typing import Final, Mapping, Sequence

from safetensors import safe_open
import torch

from .mrsc import (
    MRSC_CANDIDATE_CHANNELS,
    MRSC_SCHEMA,
    MRSC_USE_POLICY,
)
from .preprocessing_arm_runtime import (
    CAUSAL_REFERENCE_PAIR_ROLE,
    CAUSAL_REFERENCE_PAIR_SCHEMA,
    CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
)
from .v11_reasoner import V11_CANDIDATE_INDICES


FINAL_SCORE_REFERENCE_SOURCE_SCHEMA: Final[str] = (
    "soz_labram_mrsc_target_free_oof_descriptive_v1"
)
FINAL_SCORE_REFERENCE_SOURCE_STATUS: Final[str] = (
    "completed_target_free_descriptive_mrsc_threshold_undefined"
)
FINAL_SCORE_REFERENCE_RECEIPT_SCHEMA: Final[str] = (
    "soz_final_score_reference_disagreement_receipt_v1"
)
FINAL_SCORE_REFERENCE_METRIC_ID: Final[str] = (
    "normalized_jsd_softmax_fixed18_patient_final_scores_v1"
)
FINAL_SCORE_REFERENCE_SCORE_SEMANTICS: Final[str] = (
    "frozen_v11_1_oof_patient_localizer_logits_fixed18_candidate_order_v1"
)
FINAL_SCORE_REFERENCE_EVENT_HASH_SEMANTICS: Final[str] = (
    "sha256_tensor_dtype_shape_bytes_per_fixed18_event_score_row_v1"
)
FINAL_SCORE_REFERENCE_SCOPE: Final[str] = (
    "same_frozen_localizer_final_patient_candidate_scores_not_block9_representation"
)
FINAL_SCORE_REFERENCE_USE_POLICY: Final[str] = (
    "report_reference_sensitivity_and_abstention_only_not_score_or_ranking_change"
)
FINAL_SCORE_REFERENCE_MODEL_LINEAGE: Final[str] = (
    "frozen_v11_1_outer_fold_full_labram_plus_fine"
)
PRIMARY_REFERENCE_ARM_ID: Final[str] = "C-CAR19"
SOURCE_PRIMARY_REFERENCE: Final[str] = "C-CAR19_preserved"
SOURCE_SENSITIVITY_REFERENCE: Final[str] = (
    "C-REF19_same_event_same_frozen_fold_model"
)
SOURCE_TENSOR_FILENAME: Final[str] = "mrsc_target_free.safetensors"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_MANIFEST_KEYS = frozenset(
    {
        "access_receipt",
        "candidate_channels",
        "claim_boundary",
        "descriptive_results",
        "event_count",
        "event_ids",
        "fold_checkpoint_routing",
        "model_lineage",
        "mrsc_contract",
        "patient_count",
        "patient_ids",
        "primary_reference",
        "quality_contract",
        "schema_version",
        "score_parity",
        "sensitivity_reference",
        "status",
        "tensor_file",
        "tensor_specs",
    }
)
FINAL_SCORE_SOURCE_TENSOR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "candidate_indices",
        "car_event_scores_preserved",
        "car_patient_scores_preserved",
        "event_car_top1_index",
        "event_final_score_reference_disagreement",
        "event_patient_index",
        "event_ref_top1_index",
        "event_structural_quality_valid_mask",
        "event_top1_reference_agreement",
        "event_top3_reference_jaccard",
        "mrsc_abstain",
        "mrsc_abstention_reason_flags",
        "mrsc_car_top1_index",
        "mrsc_event_dispersion",
        "mrsc_event_dispersion_estimable",
        "mrsc_final_score_reference_disagreement",
        "mrsc_ranking_ambiguity",
        "mrsc_raw_nonconformity",
        "mrsc_ref_top1_index",
        "mrsc_report_fact_unavailability",
        "mrsc_review_reason_flags",
        "mrsc_signal_quality_uncertainty",
        "mrsc_top1_reference_agreement",
        "mrsc_top3_reference_jaccard",
        "patient_event_counts",
        "ref_event_scores",
        "ref_patient_scores",
        "report_fact_available_mask",
    }
)


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical_regular_file(path: Path, *, name: str) -> Path:
    lexical = Path(os.path.abspath(path))
    for component in (lexical, *lexical.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{name} cannot traverse a symlink")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a canonical regular file")
    return resolved


def _canonical_directory(path: str | Path) -> Path:
    lexical = Path(os.path.abspath(path))
    for component in (lexical, *lexical.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("MRSC source directory cannot traverse a symlink")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("MRSC source must be a canonical directory")
    return resolved


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = json.dumps(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _score_vector_sha256(
    rows: Sequence["FinalScoreCandidateSummary"],
    *,
    arm: str,
) -> str:
    if arm not in {"primary", "sensitivity"}:
        raise ValueError("arm must be primary or sensitivity")
    key = f"{arm}_score"
    return _canonical_sha256(
        {
            "score_semantics": FINAL_SCORE_REFERENCE_SCORE_SEMANTICS,
            "candidate_channels": [row.channel for row in rows],
            "scores": [getattr(row, key) for row in rows],
        }
    )


def _score_metrics(
    primary_scores: Sequence[float],
    sensitivity_scores: Sequence[float],
) -> tuple[float, int, int, bool, float]:
    primary = torch.tensor(tuple(primary_scores), dtype=torch.float64)
    sensitivity = torch.tensor(tuple(sensitivity_scores), dtype=torch.float64)
    primary_probability = torch.softmax(primary, dim=0)
    sensitivity_probability = torch.softmax(sensitivity, dim=0)
    midpoint = 0.5 * (primary_probability + sensitivity_probability)
    jsd = 0.5 * torch.sum(
        primary_probability
        * (torch.log(primary_probability) - torch.log(midpoint))
    )
    jsd += 0.5 * torch.sum(
        sensitivity_probability
        * (torch.log(sensitivity_probability) - torch.log(midpoint))
    )
    normalized = min(1.0, max(0.0, float(jsd.item() / math.log(2.0))))
    primary_rank = torch.argsort(
        primary, descending=True, stable=True
    ).tolist()
    sensitivity_rank = torch.argsort(
        sensitivity, descending=True, stable=True
    ).tolist()
    primary_top1 = int(primary_rank[0])
    sensitivity_top1 = int(sensitivity_rank[0])
    primary_top3 = set(primary_rank[:3])
    sensitivity_top3 = set(sensitivity_rank[:3])
    top3_jaccard = len(primary_top3 & sensitivity_top3) / len(
        primary_top3 | sensitivity_top3
    )
    return (
        normalized,
        primary_top1,
        sensitivity_top1,
        primary_top1 == sensitivity_top1,
        top3_jaccard,
    )


@dataclass(frozen=True)
class FinalScoreCandidateSummary:
    """One fixed-order candidate's paired final localizer scores."""

    channel: str
    primary_score: float
    sensitivity_score: float

    def __post_init__(self) -> None:
        _require_text(self.channel, name="candidate channel")
        _require_finite(self.primary_score, name="primary_score")
        _require_finite(self.sensitivity_score, name="sensitivity_score")


@dataclass(frozen=True)
class FinalScoreReferenceDisagreementReceipt:
    """Patient-level final-score reference-sensitivity receipt.

    The 18 candidate rows are descriptive copies of immutable C-CAR19 and
    C-REF19 output scores.  ``final_score_reference_disagreement`` is the
    normalized Jensen--Shannon divergence between their fixed-18 softmax
    distributions.  It is not an error probability or permission to alter
    the primary score vector.
    """

    patient_pseudonym: str
    aggregation_event_ids: tuple[str, ...]
    candidate_score_summary: tuple[FinalScoreCandidateSummary, ...]
    primary_score_vector_sha256: str
    sensitivity_score_vector_sha256: str
    primary_event_score_sha256s: tuple[tuple[str, str], ...]
    sensitivity_event_score_sha256s: tuple[tuple[str, str], ...]
    source_manifest_sha256: str
    source_tensor_sha256: str
    model_lineage: str
    reference_pair_schema_version: str
    reference_pair_role: str
    primary_arm_id: str
    sensitivity_arm_id: str
    primary_top1_channel: str
    sensitivity_top1_channel: str
    top1_reference_agreement: bool
    top3_reference_jaccard: float
    final_score_reference_disagreement: float
    same_frozen_model: bool
    primary_scores_preserved: bool
    target_values_loaded: bool
    private_data_loaded: bool
    training_performed: bool
    model_selection_performed: bool
    calibration_performed: bool
    metric_id: str = FINAL_SCORE_REFERENCE_METRIC_ID
    score_semantics: str = FINAL_SCORE_REFERENCE_SCORE_SEMANTICS
    event_score_hash_semantics: str = FINAL_SCORE_REFERENCE_EVENT_HASH_SEMANTICS
    measurement_scope: str = FINAL_SCORE_REFERENCE_SCOPE
    use_policy: str = FINAL_SCORE_REFERENCE_USE_POLICY
    schema_version: str = FINAL_SCORE_REFERENCE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_text(self.patient_pseudonym, name="patient_pseudonym")
        if (
            not isinstance(self.aggregation_event_ids, tuple)
            or not self.aggregation_event_ids
            or len(set(self.aggregation_event_ids))
            != len(self.aggregation_event_ids)
        ):
            raise ValueError("aggregation_event_ids must be non-empty and unique")
        for event_id in self.aggregation_event_ids:
            _require_text(event_id, name="aggregation_event_id")
        if (
            not isinstance(self.candidate_score_summary, tuple)
            or len(self.candidate_score_summary) != len(MRSC_CANDIDATE_CHANNELS)
            or any(
                not isinstance(row, FinalScoreCandidateSummary)
                for row in self.candidate_score_summary
            )
        ):
            raise ValueError("candidate_score_summary must contain fixed-18 rows")
        channels = tuple(row.channel for row in self.candidate_score_summary)
        if channels != MRSC_CANDIDATE_CHANNELS:
            raise ValueError("Final-score candidate order changed")
        for name in (
            "primary_score_vector_sha256",
            "sensitivity_score_vector_sha256",
            "source_manifest_sha256",
            "source_tensor_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        expected_primary_sha = _score_vector_sha256(
            self.candidate_score_summary, arm="primary"
        )
        expected_sensitivity_sha = _score_vector_sha256(
            self.candidate_score_summary, arm="sensitivity"
        )
        if self.primary_score_vector_sha256 != expected_primary_sha:
            raise ValueError("Primary final-score summary hash does not replay")
        if self.sensitivity_score_vector_sha256 != expected_sensitivity_sha:
            raise ValueError("Sensitivity final-score summary hash does not replay")
        for name in (
            "primary_event_score_sha256s",
            "sensitivity_event_score_sha256s",
        ):
            rows = getattr(self, name)
            if not isinstance(rows, tuple) or len(rows) != len(
                self.aggregation_event_ids
            ):
                raise ValueError(f"{name} must match the aggregation roster")
            bound_ids: list[str] = []
            for row in rows:
                if not isinstance(row, tuple) or len(row) != 2:
                    raise ValueError(f"{name} rows must be (event, sha256)")
                event_id, digest = row
                bound_ids.append(_require_text(event_id, name="event score event_id"))
                _require_sha256(digest, name="event_score_sha256")
            if tuple(bound_ids) != self.aggregation_event_ids:
                raise ValueError(f"{name} does not preserve aggregation order")
        if self.model_lineage != FINAL_SCORE_REFERENCE_MODEL_LINEAGE:
            raise ValueError("Unsupported final-score model lineage")
        if self.reference_pair_schema_version != CAUSAL_REFERENCE_PAIR_SCHEMA:
            raise ValueError("Final-score receipt uses another reference schema")
        if self.reference_pair_role != CAUSAL_REFERENCE_PAIR_ROLE:
            raise ValueError("Final-score receipt uses another reference role")
        if self.primary_arm_id != PRIMARY_REFERENCE_ARM_ID:
            raise ValueError("Final-score primary arm must be C-CAR19")
        if self.sensitivity_arm_id != CAUSAL_REFERENCE_SENSITIVITY_ARM_ID:
            raise ValueError("Final-score sensitivity arm must be C-REF19")

        primary = tuple(row.primary_score for row in self.candidate_score_summary)
        sensitivity = tuple(
            row.sensitivity_score for row in self.candidate_score_summary
        )
        disagreement, primary_top1, sensitivity_top1, agreement, top3 = (
            _score_metrics(primary, sensitivity)
        )
        if self.primary_top1_channel != channels[primary_top1]:
            raise ValueError("Primary top-1 channel does not replay from scores")
        if self.sensitivity_top1_channel != channels[sensitivity_top1]:
            raise ValueError("Sensitivity top-1 channel does not replay from scores")
        if type(self.top1_reference_agreement) is not bool or (
            self.top1_reference_agreement != agreement
        ):
            raise ValueError("Top-1 reference agreement does not replay")
        observed_top3 = _require_finite(
            self.top3_reference_jaccard, name="top3_reference_jaccard"
        )
        observed_disagreement = _require_finite(
            self.final_score_reference_disagreement,
            name="final_score_reference_disagreement",
        )
        if not 0 <= observed_top3 <= 1 or not math.isclose(
            observed_top3, top3, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Top-3 reference Jaccard does not replay")
        if not 0 <= observed_disagreement <= 1 or not math.isclose(
            observed_disagreement, disagreement, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Final-score normalized JSD does not replay")
        for name in (
            "same_frozen_model",
            "primary_scores_preserved",
            "target_values_loaded",
            "private_data_loaded",
            "training_performed",
            "model_selection_performed",
            "calibration_performed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if not self.same_frozen_model or not self.primary_scores_preserved:
            raise ValueError(
                "Final-score receipt requires score-preserving paired replay"
            )
        if any(
            (
                self.target_values_loaded,
                self.private_data_loaded,
                self.training_performed,
                self.model_selection_performed,
                self.calibration_performed,
            )
        ):
            raise ValueError("Final-score receipt must remain target/private/fit free")
        if self.metric_id != FINAL_SCORE_REFERENCE_METRIC_ID:
            raise ValueError("Unsupported final-score disagreement metric")
        if self.score_semantics != FINAL_SCORE_REFERENCE_SCORE_SEMANTICS:
            raise ValueError("Unsupported final-score semantics")
        if (
            self.event_score_hash_semantics
            != FINAL_SCORE_REFERENCE_EVENT_HASH_SEMANTICS
        ):
            raise ValueError("Unsupported event-score hash semantics")
        if self.measurement_scope != FINAL_SCORE_REFERENCE_SCOPE:
            raise ValueError("Unsupported final-score measurement scope")
        if self.use_policy != FINAL_SCORE_REFERENCE_USE_POLICY:
            raise ValueError("Unsupported final-score use policy")
        if self.schema_version != FINAL_SCORE_REFERENCE_RECEIPT_SCHEMA:
            raise ValueError("Unsupported final-score receipt schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def _validate_source_manifest(payload: Mapping[str, object]) -> None:
    if set(payload) != set(_SOURCE_MANIFEST_KEYS):
        raise ValueError("MRSC source manifest schema drifted")
    if payload.get("schema_version") != FINAL_SCORE_REFERENCE_SOURCE_SCHEMA:
        raise ValueError("Unsupported final-score MRSC source schema")
    if payload.get("status") != FINAL_SCORE_REFERENCE_SOURCE_STATUS:
        raise ValueError("Final-score MRSC source is not completed")
    if payload.get("model_lineage") != FINAL_SCORE_REFERENCE_MODEL_LINEAGE:
        raise ValueError("Final-score MRSC model lineage changed")
    if payload.get("primary_reference") != SOURCE_PRIMARY_REFERENCE or (
        payload.get("sensitivity_reference") != SOURCE_SENSITIVITY_REFERENCE
    ):
        raise ValueError("Final-score MRSC reference pair changed")
    if payload.get("tensor_file") != SOURCE_TENSOR_FILENAME:
        raise ValueError("Final-score MRSC tensor filename changed")
    if tuple(payload.get("candidate_channels", ())) != MRSC_CANDIDATE_CHANNELS:
        raise ValueError("Final-score MRSC candidate order changed")

    access = _require_mapping(payload.get("access_receipt"), name="access_receipt")
    for name in (
        "target_tensor_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "training_performed",
        "model_selection_performed",
        "threshold_selection_or_calibration_performed",
        "soz_outcome_metrics_computed",
        "label_based_subgrouping_performed",
    ):
        if access.get(name) is not False:
            raise ValueError(f"MRSC source violates target-free contract: {name}")
    if access.get("historical_mixed_prediction_container_opened") is not False:
        raise ValueError("MRSC source opened a mixed target-bearing artifact")

    parity = _require_mapping(payload.get("score_parity"), name="score_parity")
    for name in (
        "car_patient_bitwise_equal_before_after_mrsc",
        "car_event_bitwise_equal_before_after_mrsc",
        "r2_outer_state_car_replay_gate_passed",
    ):
        if parity.get(name) is not True:
            raise ValueError(f"MRSC source fails score-parity gate: {name}")
    if parity.get("maximum_absolute_car_score_change") != 0.0:
        raise ValueError("MRSC source changed primary CAR scores")

    contract = _require_mapping(payload.get("mrsc_contract"), name="mrsc_contract")
    if contract.get("core_schema") != MRSC_SCHEMA or (
        contract.get("use_policy") != MRSC_USE_POLICY
    ):
        raise ValueError("MRSC core contract changed")
    if contract.get("selective_threshold_defined") is not False or (
        contract.get("all_patients_fail_closed") is not True
    ):
        raise ValueError("MRSC source is not the uncalibrated fail-closed artifact")


def _load_source_tensors(
    tensor_path: Path,
    *,
    tensor_specs: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    if set(tensor_specs) != set(FINAL_SCORE_SOURCE_TENSOR_KEYS):
        raise ValueError("MRSC tensor specification vocabulary changed")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(FINAL_SCORE_SOURCE_TENSOR_KEYS):
            raise ValueError("MRSC tensor vocabulary changed")
        for name in sorted(FINAL_SCORE_SOURCE_TENSOR_KEYS):
            value = handle.get_tensor(name).detach().cpu().contiguous()
            spec = _require_mapping(tensor_specs.get(name), name=f"tensor_specs.{name}")
            if set(spec) != {"shape", "dtype"}:
                raise ValueError(f"MRSC tensor spec drifted for {name}")
            if list(value.shape) != spec.get("shape") or (
                str(value.dtype) != spec.get("dtype")
            ):
                raise ValueError(f"MRSC tensor disagrees with its spec: {name}")
            tensors[name] = value
    return tensors


def load_final_score_reference_disagreement_receipt(
    source_directory: str | Path,
    *,
    patient_pseudonym: str,
    aggregation_event_ids: Sequence[str],
    expected_source_manifest_sha256: str,
    expected_source_tensor_sha256: str,
) -> FinalScoreReferenceDisagreementReceipt:
    """Load one patient's final-score receipt from the sealed MRSC output.

    The two expected digests are mandatory trust anchors.  Passing hashes
    calculated opportunistically after a suspected mutation does not provide
    provenance; callers should freeze them in an upstream run or release
    receipt.
    """

    patient = _require_text(patient_pseudonym, name="patient_pseudonym")
    if isinstance(aggregation_event_ids, (str, bytes)):
        raise TypeError("aggregation_event_ids must be a sequence")
    requested_events = tuple(
        _require_text(value, name="aggregation_event_id")
        for value in aggregation_event_ids
    )
    if not requested_events or len(set(requested_events)) != len(requested_events):
        raise ValueError("aggregation_event_ids must be non-empty and unique")
    expected_manifest_sha = _require_sha256(
        expected_source_manifest_sha256,
        name="expected_source_manifest_sha256",
    )
    expected_tensor_sha = _require_sha256(
        expected_source_tensor_sha256,
        name="expected_source_tensor_sha256",
    )

    directory = _canonical_directory(source_directory)
    manifest_path = _canonical_regular_file(
        directory / "manifest.json", name="MRSC source manifest"
    )
    if manifest_path.parent != directory:
        raise ValueError("MRSC source manifest escaped its directory")
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if not hmac.compare_digest(manifest_sha, expected_manifest_sha):
        raise ValueError("MRSC source manifest hash mismatch")
    try:
        payload = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MRSC source manifest is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("MRSC source manifest must contain one object")
    _validate_source_manifest(payload)

    tensor_path = _canonical_regular_file(
        directory / SOURCE_TENSOR_FILENAME, name="MRSC source tensor"
    )
    if tensor_path.parent != directory:
        raise ValueError("MRSC source tensor escaped its directory")
    tensor_sha = _file_sha256(tensor_path)
    if not hmac.compare_digest(tensor_sha, expected_tensor_sha):
        raise ValueError("MRSC source tensor hash mismatch")
    tensor_specs = _require_mapping(payload.get("tensor_specs"), name="tensor_specs")
    tensors = _load_source_tensors(tensor_path, tensor_specs=tensor_specs)

    patient_ids_raw = payload.get("patient_ids")
    event_ids_raw = payload.get("event_ids")
    if not isinstance(patient_ids_raw, list) or not isinstance(event_ids_raw, list):
        raise TypeError("MRSC source lacks patient/event rosters")
    patient_ids = tuple(
        _require_text(value, name="source patient_id") for value in patient_ids_raw
    )
    event_ids = tuple(
        _require_text(value, name="source event_id") for value in event_ids_raw
    )
    patient_count = payload.get("patient_count")
    event_count = payload.get("event_count")
    if type(patient_count) is not int or patient_count < 1 or (
        len(patient_ids) != patient_count
        or len(set(patient_ids)) != patient_count
    ):
        raise ValueError("MRSC patient roster/count changed")
    if type(event_count) is not int or event_count < patient_count or (
        len(event_ids) != event_count
        or len(set(event_ids)) != event_count
    ):
        raise ValueError("MRSC event roster/count changed")
    matches = tuple(
        index for index, value in enumerate(patient_ids) if value == patient
    )
    if len(matches) != 1:
        raise ValueError("Requested patient is absent or duplicated in MRSC roster")
    patient_index = matches[0]

    candidate_indices = tensors["candidate_indices"]
    if (
        candidate_indices.dtype != torch.long
        or tuple(candidate_indices.shape) != (18,)
        or not torch.equal(
            candidate_indices,
            torch.tensor(V11_CANDIDATE_INDICES, dtype=torch.long),
        )
    ):
        raise ValueError("MRSC candidate indices changed")
    event_patient_index = tensors["event_patient_index"]
    patient_event_counts = tensors["patient_event_counts"]
    if event_patient_index.dtype != torch.long or tuple(event_patient_index.shape) != (
        event_count,
    ):
        raise ValueError("MRSC event-to-patient routing is invalid")
    if (
        patient_event_counts.dtype != torch.long
        or tuple(patient_event_counts.shape) != (patient_count,)
    ):
        raise ValueError("MRSC patient event counts are invalid")
    if int(event_patient_index.min()) != 0 or int(event_patient_index.max()) != (
        patient_count - 1
    ):
        raise ValueError("MRSC event-to-patient roster is not contiguous")
    replayed_counts = torch.bincount(
        event_patient_index, minlength=patient_count
    ).long()
    if not torch.equal(replayed_counts, patient_event_counts) or bool(
        (patient_event_counts < 1).any()
    ):
        raise ValueError("MRSC patient event counts do not replay")
    selected_event_indices = torch.nonzero(
        event_patient_index == patient_index, as_tuple=False
    ).flatten()
    source_patient_events = tuple(
        event_ids[index] for index in selected_event_indices.tolist()
    )
    if source_patient_events != requested_events:
        raise ValueError("MRSC receipt and requested aggregation roster mismatch")

    expected_shapes = {
        "car_patient_scores_preserved": (patient_count, 18),
        "ref_patient_scores": (patient_count, 18),
        "car_event_scores_preserved": (event_count, 18),
        "ref_event_scores": (event_count, 18),
        "mrsc_final_score_reference_disagreement": (patient_count,),
        "mrsc_car_top1_index": (patient_count,),
        "mrsc_ref_top1_index": (patient_count,),
        "mrsc_top1_reference_agreement": (patient_count,),
        "mrsc_top3_reference_jaccard": (patient_count,),
        "mrsc_abstain": (patient_count,),
    }
    for name, shape in expected_shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"MRSC final-score tensor has wrong shape: {name}")
    for name in (
        "car_patient_scores_preserved",
        "ref_patient_scores",
        "car_event_scores_preserved",
        "ref_event_scores",
    ):
        if tensors[name].dtype != torch.float32 or not torch.isfinite(
            tensors[name]
        ).all():
            raise ValueError(f"MRSC final-score tensor is not finite: {name}")
    for name in (
        "mrsc_final_score_reference_disagreement",
        "mrsc_top3_reference_jaccard",
    ):
        if tensors[name].dtype != torch.float64 or not torch.isfinite(
            tensors[name]
        ).all():
            raise ValueError(f"MRSC final-score metric tensor is invalid: {name}")
    for name in ("mrsc_car_top1_index", "mrsc_ref_top1_index"):
        if tensors[name].dtype != torch.long:
            raise ValueError(f"MRSC final-score rank tensor must be long: {name}")
    if tensors["mrsc_top1_reference_agreement"].dtype != torch.bool or (
        tensors["mrsc_abstain"].dtype != torch.bool
    ):
        raise ValueError("MRSC agreement/abstention tensors must be bool")
    if not bool(tensors["mrsc_abstain"].all()):
        raise ValueError("Uncalibrated MRSC source must fail closed")

    primary_row = tensors["car_patient_scores_preserved"][patient_index]
    sensitivity_row = tensors["ref_patient_scores"][patient_index]
    summary = tuple(
        FinalScoreCandidateSummary(
            channel=channel,
            primary_score=float(primary_row[index]),
            sensitivity_score=float(sensitivity_row[index]),
        )
        for index, channel in enumerate(MRSC_CANDIDATE_CHANNELS)
    )
    metric, primary_top1, sensitivity_top1, agreement, top3 = _score_metrics(
        tuple(row.primary_score for row in summary),
        tuple(row.sensitivity_score for row in summary),
    )
    stored_metric = float(
        tensors["mrsc_final_score_reference_disagreement"][patient_index]
    )
    stored_primary_top1 = int(tensors["mrsc_car_top1_index"][patient_index])
    stored_sensitivity_top1 = int(tensors["mrsc_ref_top1_index"][patient_index])
    stored_agreement = bool(
        tensors["mrsc_top1_reference_agreement"][patient_index]
    )
    stored_top3 = float(tensors["mrsc_top3_reference_jaccard"][patient_index])
    if not math.isclose(stored_metric, metric, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("MRSC final-score disagreement does not replay")
    if (
        stored_primary_top1 != primary_top1
        or stored_sensitivity_top1 != sensitivity_top1
        or stored_agreement != agreement
        or not math.isclose(stored_top3, top3, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("MRSC final-score ranking summary does not replay")

    primary_event_rows = tensors["car_event_scores_preserved"].index_select(
        0, selected_event_indices
    )
    sensitivity_event_rows = tensors["ref_event_scores"].index_select(
        0, selected_event_indices
    )
    primary_event_hashes = tuple(
        (event_id, _tensor_sha256(row))
        for event_id, row in zip(requested_events, primary_event_rows)
    )
    sensitivity_event_hashes = tuple(
        (event_id, _tensor_sha256(row))
        for event_id, row in zip(requested_events, sensitivity_event_rows)
    )
    return FinalScoreReferenceDisagreementReceipt(
        patient_pseudonym=patient,
        aggregation_event_ids=requested_events,
        candidate_score_summary=summary,
        primary_score_vector_sha256=_score_vector_sha256(summary, arm="primary"),
        sensitivity_score_vector_sha256=_score_vector_sha256(
            summary, arm="sensitivity"
        ),
        primary_event_score_sha256s=primary_event_hashes,
        sensitivity_event_score_sha256s=sensitivity_event_hashes,
        source_manifest_sha256=manifest_sha,
        source_tensor_sha256=tensor_sha,
        model_lineage=FINAL_SCORE_REFERENCE_MODEL_LINEAGE,
        reference_pair_schema_version=CAUSAL_REFERENCE_PAIR_SCHEMA,
        reference_pair_role=CAUSAL_REFERENCE_PAIR_ROLE,
        primary_arm_id=PRIMARY_REFERENCE_ARM_ID,
        sensitivity_arm_id=CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
        primary_top1_channel=MRSC_CANDIDATE_CHANNELS[primary_top1],
        sensitivity_top1_channel=MRSC_CANDIDATE_CHANNELS[sensitivity_top1],
        top1_reference_agreement=agreement,
        top3_reference_jaccard=top3,
        final_score_reference_disagreement=metric,
        same_frozen_model=True,
        primary_scores_preserved=True,
        target_values_loaded=False,
        private_data_loaded=False,
        training_performed=False,
        model_selection_performed=False,
        calibration_performed=False,
    )


__all__ = [
    "FINAL_SCORE_REFERENCE_EVENT_HASH_SEMANTICS",
    "FINAL_SCORE_REFERENCE_METRIC_ID",
    "FINAL_SCORE_REFERENCE_RECEIPT_SCHEMA",
    "FINAL_SCORE_REFERENCE_SCOPE",
    "FINAL_SCORE_REFERENCE_SCORE_SEMANTICS",
    "FINAL_SCORE_REFERENCE_SOURCE_SCHEMA",
    "FINAL_SCORE_REFERENCE_SOURCE_STATUS",
    "FINAL_SCORE_REFERENCE_USE_POLICY",
    "FINAL_SCORE_SOURCE_TENSOR_KEYS",
    "FinalScoreCandidateSummary",
    "FinalScoreReferenceDisagreementReceipt",
    "load_final_score_reference_disagreement_receipt",
]
