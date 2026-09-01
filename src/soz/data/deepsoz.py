"""DeepSOZ patient-level benchmark-reference contract.

DeepSOZ is treated as a label overlay on TUSZ, not as an independent EEG
dataset.  Explicit binary zeros are operational dataset-complement negatives
for the published benchmark only; they are not clinician-confirmed biological
non-SOZ electrodes.  Missing values stay masked and canonical PZ is masked in
the primary policy because the upstream file contains conflicting duplicate
PZ columns.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable, Iterator, Sequence

import pandas as pd
import torch

from ..geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS, STANDARD_19


DEEPSOZ_COLUMN_BY_CHANNEL: dict[str, str] = {
    "FP1": "fp1",
    "FP2": "fp2",
    "F7": "f7",
    "F3": "f3",
    "FZ": "fz",
    "F4": "f4",
    "F8": "f8",
    "T7": "t3",
    "C3": "c3",
    "CZ": "cz",
    "C4": "c4",
    "T8": "t4",
    "P7": "t5",
    "P3": "p3",
    "P4": "p4",
    "P8": "t6",
    "O1": "o1",
    "O2": "o2",
}
OUTSIDE_HEAD_COLUMNS: dict[str, str] = {"OZ": "oz", "A1": "a1", "A2": "a2"}
ALLOWED_MODEL_SPLITS = frozenset({"source_train", "source_dev", "source_eval"})
ALLOWED_INCLUDED_COHORT_STATUSES = frozenset({"included", "included_positive_only"})
ALLOWED_QUARANTINE_COHORT_STATUSES = frozenset(
    {"quarantine_variable_label", "quarantine_no_strict_input_event"}
)
EXPECTED_CONCEPT_OOF_FOLDS = 5
OFFICIAL_TO_MODEL_SPLIT: dict[str, str] = {
    "train": "source_train",
    "dev": "source_dev",
    "eval": "source_eval",
}

BINARY_STATE_EXPLICIT_1 = "explicit_1"
BINARY_STATE_EXPLICIT_0 = "explicit_0"
BINARY_STATE_MISSING = "missing"
BINARY_STATE_PATIENT_VARIABLE = "patient_variable"
PZ_PRIMARY_STATE = "masked_pz_duplicate_schema"
_BINARY_STATES = frozenset(
    {
        BINARY_STATE_EXPLICIT_1,
        BINARY_STATE_EXPLICIT_0,
        BINARY_STATE_MISSING,
        BINARY_STATE_PATIENT_VARIABLE,
    }
)


def normalize_patient_id(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        return text.split(".", 1)[0]
    if not text or text.lower() == "nan":
        raise ValueError("Patient ID is missing")
    return text


@dataclass(frozen=True)
class BinaryFieldAudit:
    """Patient-level audit of one binary source field.

    Missing source cells are counted separately from explicit zeros.  Invalid
    non-binary values never reach this object: parsing fails closed with source
    column, patient, and row context.
    """

    state: str
    stable_value: int | None
    explicit_1_count: int
    explicit_0_count: int
    missing_count: int

    def __post_init__(self) -> None:
        if self.state not in _BINARY_STATES:
            raise ValueError(f"Unsupported binary audit state: {self.state}")
        counts = (self.explicit_1_count, self.explicit_0_count, self.missing_count)
        if any(count < 0 for count in counts) or sum(counts) < 1:
            raise ValueError("Binary audit counts must be non-negative and non-empty")
        if self.stable_value not in {None, 0, 1}:
            raise ValueError("stable_value must be 0, 1, or None")
        expected = {
            BINARY_STATE_EXPLICIT_1: 1,
            BINARY_STATE_EXPLICIT_0: 0,
            BINARY_STATE_MISSING: None,
            BINARY_STATE_PATIENT_VARIABLE: None,
        }[self.state]
        if self.stable_value != expected:
            raise ValueError(
                f"State {self.state} is inconsistent with stable_value={self.stable_value}"
            )

    @property
    def has_observed_positive(self) -> bool:
        return self.explicit_1_count > 0


@dataclass(frozen=True)
class PatientSOZReference:
    """One patient-level standard-19 benchmark reference."""

    patient_id: str
    values: torch.Tensor
    mask: torch.Tensor
    model_split: str
    official_split: str
    concept_oof_fold: int | None
    eligible_for_localization: bool
    exclusion_reason: str
    source_record_count: int
    outside_head_positives: tuple[str, ...]
    masked_target_states: tuple[str, ...]
    target_states: tuple[str, ...]
    pz_first_audit: BinaryFieldAudit
    pz_second_audit: BinaryFieldAudit
    pz_or_audit: BinaryFieldAudit
    outside_head_audits: tuple[tuple[str, BinaryFieldAudit], ...]
    zero_semantics: str = "dataset_complement_negative_not_biological_negative"
    pz_policy: str = "mask_duplicate_schema_conflict"

    def __post_init__(self) -> None:
        if tuple(self.values.shape) != (N_STANDARD_CHANNELS,):
            raise ValueError("Patient target values must have shape [19]")
        if tuple(self.mask.shape) != (N_STANDARD_CHANNELS,):
            raise ValueError("Patient target mask must have shape [19]")
        if not self.values.is_floating_point() or self.mask.dtype != torch.bool:
            raise TypeError("Patient target values must be float and mask must be bool")
        observed = self.values[self.mask]
        if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
            raise ValueError("Observed patient targets must be binary")
        if self.mask[CHANNEL_INDEX["PZ"]]:
            raise ValueError("Canonical PZ must be masked under the primary policy")
        if len(self.target_states) != N_STANDARD_CHANNELS:
            raise ValueError("target_states must contain exactly 19 entries")
        if self.target_states[CHANNEL_INDEX["PZ"]] != PZ_PRIMARY_STATE:
            raise ValueError("Canonical PZ must retain the primary masked state")
        for index, state in enumerate(self.target_states):
            if index == CHANNEL_INDEX["PZ"]:
                continue
            if state == BINARY_STATE_EXPLICIT_1:
                if not self.mask[index] or self.values[index] != 1:
                    raise ValueError("explicit_1 state requires observed value 1")
            elif state == BINARY_STATE_EXPLICIT_0:
                if not self.mask[index] or self.values[index] != 0:
                    raise ValueError("explicit_0 state requires observed value 0")
            elif state in {BINARY_STATE_MISSING, BINARY_STATE_PATIENT_VARIABLE}:
                if self.mask[index]:
                    raise ValueError(f"{state} state must be masked")
            else:
                raise ValueError(f"Unsupported target state: {state}")
        outside_names = tuple(name for name, _ in self.outside_head_audits)
        if outside_names != tuple(OUTSIDE_HEAD_COLUMNS):
            raise ValueError("outside_head_audits must use the frozen OZ/A1/A2 order")
        expected_outside_positive = tuple(
            name for name, audit in self.outside_head_audits if audit.has_observed_positive
        )
        if self.outside_head_positives != expected_outside_positive:
            raise ValueError("outside_head_positives disagrees with outside-head audits")
        if self.eligible_for_localization:
            if self.model_split not in ALLOWED_MODEL_SPLITS:
                raise ValueError("Eligible references require a source train/dev/eval split")
            if not ((self.values == 1) & self.mask).any():
                raise ValueError("Eligible references require an observed in-head positive")


@dataclass(frozen=True)
class SOZTargetBatch:
    patient_ids: tuple[str, ...]
    values: torch.Tensor
    mask: torch.Tensor


class DeepSOZReferenceRegistry(Sequence[PatientSOZReference]):
    """Immutable patient registry with deterministic split selection."""

    def __init__(self, references: Iterable[PatientSOZReference]) -> None:
        ordered = tuple(sorted(references, key=lambda item: item.patient_id))
        by_id = {reference.patient_id: reference for reference in ordered}
        if len(by_id) != len(ordered):
            raise ValueError("DeepSOZ registry contains duplicate patient IDs")
        self._references = ordered
        self._by_id = by_id

    def __len__(self) -> int:
        return len(self._references)

    def __getitem__(self, index: int) -> PatientSOZReference:
        return self._references[index]

    def __iter__(self) -> Iterator[PatientSOZReference]:
        return iter(self._references)

    def get(self, patient_id: object) -> PatientSOZReference:
        normalized = normalize_patient_id(patient_id)
        try:
            return self._by_id[normalized]
        except KeyError as exc:
            raise KeyError(f"Unknown DeepSOZ patient: {normalized}") from exc

    def for_split(
        self, model_split: str, *, eligible_only: bool = True
    ) -> tuple[PatientSOZReference, ...]:
        return tuple(
            reference
            for reference in self._references
            if reference.model_split == model_split
            and (reference.eligible_for_localization or not eligible_only)
        )

    def target_batch(
        self,
        patient_ids: Sequence[object],
        *,
        require_eligible: bool = True,
        device: torch.device | str | None = None,
    ) -> SOZTargetBatch:
        normalized = tuple(normalize_patient_id(value) for value in patient_ids)
        if len(set(normalized)) != len(normalized):
            raise ValueError("A target batch may contain each patient only once")
        references = tuple(self.get(value) for value in normalized)
        if require_eligible:
            excluded = [
                reference.patient_id
                for reference in references
                if not reference.eligible_for_localization
            ]
            if excluded:
                raise ValueError(f"Ineligible patients requested for localization: {excluded}")
        values = torch.stack([reference.values for reference in references]).to(device=device)
        mask = torch.stack([reference.mask for reference in references]).to(device=device)
        return SOZTargetBatch(patient_ids=normalized, values=values, mask=mask)


def _validate_source_columns(source: pd.DataFrame) -> None:
    required = {
        "pt_id",
        *DEEPSOZ_COLUMN_BY_CHANNEL.values(),
        "pz",
        "pz.1",
        *OUTSIDE_HEAD_COLUMNS.values(),
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"DeepSOZ source is missing columns: {missing}")


def _validate_split_columns(split: pd.DataFrame) -> None:
    required = {
        "deepsoz_patient_id",
        "official_split",
        "model_split",
        "concept_oof_fold",
        "cohort_status",
    }
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"Split manifest is missing columns: {missing}")


def _is_missing_binary_cell(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _parse_binary_cell(
    value: object,
    *,
    patient_id: str,
    column: str,
    row_index: object,
) -> int | None:
    """Parse one source cell without coercing malformed values to missing."""

    if _is_missing_binary_cell(value):
        return None
    try:
        numeric = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Invalid DeepSOZ binary value: "
            f"patient={patient_id}, column={column}, row={row_index}, value={value!r}"
        ) from exc
    if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
        raise ValueError(
            "Invalid DeepSOZ binary value: "
            f"patient={patient_id}, column={column}, row={row_index}, value={value!r}"
        )
    return int(numeric)


def _audit_binary_values(
    values: pd.Series,
    *,
    patient_id: str,
    column: str,
) -> BinaryFieldAudit:
    parsed = [
        _parse_binary_cell(
            value,
            patient_id=patient_id,
            column=column,
            row_index=row_index,
        )
        for row_index, value in values.items()
    ]
    ones = sum(value == 1 for value in parsed)
    zeros = sum(value == 0 for value in parsed)
    missing = sum(value is None for value in parsed)
    observed = {value for value in parsed if value is not None}
    if observed == {0, 1}:
        state = BINARY_STATE_PATIENT_VARIABLE
        stable_value = None
    elif missing:
        state = BINARY_STATE_MISSING
        stable_value = None
    elif observed == {1}:
        state = BINARY_STATE_EXPLICIT_1
        stable_value = 1
    elif observed == {0}:
        state = BINARY_STATE_EXPLICIT_0
        stable_value = 0
    else:
        # A non-empty patient group containing no observed value is all-missing.
        state = BINARY_STATE_MISSING
        stable_value = None
    return BinaryFieldAudit(
        state=state,
        stable_value=stable_value,
        explicit_1_count=ones,
        explicit_0_count=zeros,
        missing_count=missing,
    )


def _audit_pz_or(patient_rows: pd.DataFrame, *, patient_id: str) -> BinaryFieldAudit:
    combined: list[int | None] = []
    for row_index, row in patient_rows[["pz", "pz.1"]].iterrows():
        first = _parse_binary_cell(
            row["pz"], patient_id=patient_id, column="pz", row_index=row_index
        )
        second = _parse_binary_cell(
            row["pz.1"], patient_id=patient_id, column="pz.1", row_index=row_index
        )
        combined.append(None if first is None or second is None else max(first, second))
    return _audit_binary_values(
        pd.Series(combined, index=patient_rows.index, dtype=object),
        patient_id=patient_id,
        column="pz_or",
    )


def _parse_concept_oof_fold(
    value: object,
    *,
    patient_id: str,
    model_split: str,
) -> int | None:
    missing = _is_missing_binary_cell(value)
    if model_split == "source_train":
        if missing:
            raise ValueError(
                f"source_train patient {patient_id} requires a concept_oof_fold"
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid concept_oof_fold for patient {patient_id}: {value!r}"
            ) from exc
        if (
            not math.isfinite(numeric)
            or not numeric.is_integer()
            or not 0 <= numeric < EXPECTED_CONCEPT_OOF_FOLDS
        ):
            raise ValueError(
                f"Invalid concept_oof_fold for patient {patient_id}: {value!r}"
            )
        return int(numeric)
    if not missing:
        raise ValueError(
            f"Non-training patient {patient_id} must not have concept_oof_fold={value!r}"
        )
    return None


def _validate_split_row(patient_id: str, row: pd.Series) -> tuple[str, str, int | None]:
    official_split = str(row["official_split"]).strip().lower()
    model_split = str(row["model_split"]).strip()
    cohort_status = str(row["cohort_status"]).strip()
    if official_split not in OFFICIAL_TO_MODEL_SPLIT:
        raise ValueError(
            f"Unsupported official_split for patient {patient_id}: {official_split!r}"
        )
    if model_split in ALLOWED_MODEL_SPLITS:
        expected = OFFICIAL_TO_MODEL_SPLIT[official_split]
        if model_split != expected:
            raise ValueError(
                f"Split mismatch for patient {patient_id}: official={official_split}, "
                f"model={model_split}, expected={expected}"
            )
        if cohort_status not in ALLOWED_INCLUDED_COHORT_STATUSES:
            raise ValueError(
                f"Eligible model split for patient {patient_id} has non-included "
                f"cohort_status={cohort_status!r}"
            )
    elif model_split == "quarantine":
        if cohort_status not in ALLOWED_QUARANTINE_COHORT_STATUSES:
            raise ValueError(
                f"Quarantined patient {patient_id} has invalid cohort_status={cohort_status!r}"
            )
    else:
        raise ValueError(f"Unsupported model_split for patient {patient_id}: {model_split!r}")
    fold = _parse_concept_oof_fold(
        row["concept_oof_fold"], patient_id=patient_id, model_split=model_split
    )
    return official_split, model_split, fold


def build_deepsoz_reference_registry(
    source: pd.DataFrame,
    split: pd.DataFrame,
) -> DeepSOZReferenceRegistry:
    """Build the frozen v2 benchmark policy without overwriting v1 artifacts."""

    _validate_source_columns(source)
    _validate_split_columns(split)
    source = source.copy()
    split = split.copy()
    source["_patient_id"] = source["pt_id"].map(normalize_patient_id)
    split["_patient_id"] = split["deepsoz_patient_id"].map(normalize_patient_id)
    if split["_patient_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate patients")
    source_ids = set(source["_patient_id"])
    split_ids = set(split["_patient_id"])
    if source_ids != split_ids:
        raise ValueError(
            "Source/split patient mismatch: "
            f"source_only={sorted(source_ids - split_ids)[:5]}, "
            f"split_only={sorted(split_ids - source_ids)[:5]}"
        )
    split_by_id = split.set_index("_patient_id", drop=False)

    references: list[PatientSOZReference] = []
    for patient_id, patient_rows in source.groupby("_patient_id", sort=True):
        split_row = split_by_id.loc[patient_id]
        values = torch.zeros(N_STANDARD_CHANNELS, dtype=torch.float32)
        mask = torch.zeros(N_STANDARD_CHANNELS, dtype=torch.bool)
        target_states = ["uninitialized"] * N_STANDARD_CHANNELS
        masked_channel_states: list[str] = []
        fatal_channel_problems: list[str] = []
        for channel, source_column in DEEPSOZ_COLUMN_BY_CHANNEL.items():
            audit = _audit_binary_values(
                patient_rows[source_column],
                patient_id=patient_id,
                column=source_column,
            )
            channel_index = CHANNEL_INDEX[channel]
            target_states[channel_index] = audit.state
            if audit.state in {BINARY_STATE_EXPLICIT_0, BINARY_STATE_EXPLICIT_1}:
                values[channel_index] = float(audit.stable_value)
                mask[CHANNEL_INDEX[channel]] = True
            else:
                masked_channel_states.append(f"{channel}:{audit.state}")
                if audit.state == BINARY_STATE_PATIENT_VARIABLE:
                    fatal_channel_problems.append(f"{channel}:{audit.state}")

        # PZ is deliberately unavailable in the primary policy.  Its two raw
        # columns remain in the source artifact for prespecified sensitivities.
        pz_index = CHANNEL_INDEX["PZ"]
        mask[pz_index] = False
        values[pz_index] = 0.0
        target_states[pz_index] = PZ_PRIMARY_STATE
        pz_first_audit = _audit_binary_values(
            patient_rows["pz"], patient_id=patient_id, column="pz"
        )
        pz_second_audit = _audit_binary_values(
            patient_rows["pz.1"], patient_id=patient_id, column="pz.1"
        )
        pz_or_audit = _audit_pz_or(patient_rows, patient_id=patient_id)

        outside_audits = tuple(
            (
                name,
                _audit_binary_values(
                    patient_rows[source_column],
                    patient_id=patient_id,
                    column=source_column,
                ),
            )
            for name, source_column in OUTSIDE_HEAD_COLUMNS.items()
        )
        outside_positive = tuple(
            name for name, audit in outside_audits if audit.has_observed_positive
        )
        official_split, model_split, concept_oof_fold = _validate_split_row(
            patient_id, split_row
        )
        has_positive = bool(((values == 1) & mask).any())

        reasons: list[str] = []
        if model_split not in ALLOWED_MODEL_SPLITS:
            reasons.append(str(split_row["cohort_status"]))
        if fatal_channel_problems:
            reasons.append("label_schema:" + ",".join(fatal_channel_problems))
        if not has_positive:
            reasons.append("no_unmasked_in_head_positive")
        eligible = not reasons
        references.append(
            PatientSOZReference(
                patient_id=patient_id,
                values=values,
                mask=mask,
                model_split=model_split,
                official_split=official_split,
                concept_oof_fold=concept_oof_fold,
                eligible_for_localization=eligible,
                exclusion_reason=";".join(reasons),
                source_record_count=int(len(patient_rows)),
                outside_head_positives=outside_positive,
                masked_target_states=tuple(masked_channel_states),
                target_states=tuple(target_states),
                pz_first_audit=pz_first_audit,
                pz_second_audit=pz_second_audit,
                pz_or_audit=pz_or_audit,
                outside_head_audits=outside_audits,
            )
        )
    return DeepSOZReferenceRegistry(references)


def load_deepsoz_reference_registry(
    source_csv: str | Path,
    split_csv: str | Path,
) -> DeepSOZReferenceRegistry:
    return build_deepsoz_reference_registry(
        pd.read_csv(Path(source_csv)),
        pd.read_csv(Path(split_csv)),
    )


def registry_to_frame(registry: DeepSOZReferenceRegistry) -> pd.DataFrame:
    """Serialize an auditable v2 view with benchmark values and masks separate."""

    def add_audit_columns(
        row: dict[str, object], prefix: str, audit: BinaryFieldAudit
    ) -> None:
        row[f"{prefix}_state"] = audit.state
        row[f"{prefix}_stable_value"] = (
            "" if audit.stable_value is None else audit.stable_value
        )
        row[f"{prefix}_explicit_1_count"] = audit.explicit_1_count
        row[f"{prefix}_explicit_0_count"] = audit.explicit_0_count
        row[f"{prefix}_missing_count"] = audit.missing_count

    rows: list[dict[str, object]] = []
    for reference in registry:
        row: dict[str, object] = {
            "deepsoz_patient_id": reference.patient_id,
            "model_split": reference.model_split,
            "official_split": reference.official_split,
            "concept_oof_fold": reference.concept_oof_fold,
            "eligible_for_localization": int(reference.eligible_for_localization),
            "exclusion_reason": reference.exclusion_reason,
            "source_record_count": reference.source_record_count,
            "outside_head_positives": "|".join(reference.outside_head_positives),
            "masked_target_states": "|".join(reference.masked_target_states),
            "zero_semantics": reference.zero_semantics,
            "pz_policy": reference.pz_policy,
        }
        for index, channel in enumerate(STANDARD_19):
            row[f"benchmark_value_{channel}"] = int(reference.values[index].item())
            row[f"benchmark_mask_{channel}"] = int(reference.mask[index].item())
            row[f"benchmark_state_{channel}"] = reference.target_states[index]
        add_audit_columns(row, "pz_first", reference.pz_first_audit)
        add_audit_columns(row, "pz_second", reference.pz_second_audit)
        add_audit_columns(row, "pz_or", reference.pz_or_audit)
        for name, audit in reference.outside_head_audits:
            add_audit_columns(row, f"outside_{name}", audit)
        rows.append(row)
    return pd.DataFrame(rows)
