"""Pre-result, immutable promotion-gate policy for formal ictal producers.

The numerical promotion thresholds are intentionally absent from every
training and promotion caller.  They live in one repository policy document
whose bytes are pinned below.  A formal run must consume a strictly reloaded
two-file bundle and externally pin both bundle-file hashes before any optimizer
step.  The source-document hash is an integrity binding, not evidence of
preregistration or wall-clock publication.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping


ICTAL_PROMOTION_GATE_POLICY_SCHEMA = "soz_ictal_promotion_gate_policy_v1"
ICTAL_PROMOTION_GATE_POLICY_DOCUMENT_SCHEMA = (
    "soz_ictal_promotion_gate_policy_document_v1"
)
ICTAL_PROMOTION_GATE_POLICY_ARTIFACT_SCHEMA = (
    "soz_ictal_promotion_gate_policy_artifact_v1"
)
ICTAL_PROMOTION_GATE_POLICY_RECEIPT_SCHEMA = (
    "soz_ictal_promotion_gate_policy_bundle_receipt_v1"
)
ICTAL_PROMOTION_GATE_POLICY_FILENAME = "gate_policy.json"
ICTAL_PROMOTION_GATE_POLICY_RECEIPT_FILENAME = "receipt.json"
ICTAL_LOCKED_PROMOTION_GATE_POLICY_DOCUMENT_RELATIVE_PATH = (
    "configs/ictal_promotion_gate_policy_v1.json"
)
ICTAL_LOCKED_PROMOTION_GATE_POLICY_DOCUMENT_SHA256 = (
    "2a301a4218b23143c85d68d480c4b1e651ffa298e160a74b2c55fa38a5d2b9f2"
)
ICTAL_SCALE_QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_POLICY_FILE_BYTES = 128 * 1024
_VERIFIED_POLICY_MARKER = object()
_POLICY_DOCUMENT_FIELDS = frozenset(
    {
        "maximum_fold_identity_bootstrap_upper_95",
        "maximum_pairwise_scale_quantile_gap",
        "maximum_patient_macro_bce",
        "maximum_patient_macro_brier",
        "minimum_fold_patient_macro_ap_lift_over_prevalence",
        "minimum_fold_identity_permutation_p_value",
        "minimum_patient_macro_bce_improvement_over_prevalence",
        "minimum_patient_macro_brier_improvement_over_prevalence",
        "minimum_shortcut_bce_improvement",
        "scale_quantile_levels",
        "schema_version",
        "source_dev_probability_calibration_forbidden",
        "unknown_native_target_policy",
    }
)
_POLICY_FIELDS = frozenset({"policy_document_sha256", *_POLICY_DOCUMENT_FIELDS})
_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "policy_document_sha256",
        "policy_receipt_sha256",
        "policy",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_sha256",
        "policy_document_sha256",
        "policy_receipt_sha256",
    }
)
_LOCKED_POLICY_DOCUMENT = {
    "maximum_fold_identity_bootstrap_upper_95": 0.4,
    "maximum_pairwise_scale_quantile_gap": 0.1,
    "maximum_patient_macro_bce": 0.6931471805599453,
    "maximum_patient_macro_brier": 0.25,
    "minimum_fold_patient_macro_ap_lift_over_prevalence": 0.01,
    "minimum_fold_identity_permutation_p_value": 0.05,
    "minimum_patient_macro_bce_improvement_over_prevalence": 0.01,
    "minimum_patient_macro_brier_improvement_over_prevalence": 0.01,
    "minimum_shortcut_bce_improvement": 0.01,
    "scale_quantile_levels": list(ICTAL_SCALE_QUANTILE_LEVELS),
    "schema_version": ICTAL_PROMOTION_GATE_POLICY_DOCUMENT_SCHEMA,
    "source_dev_probability_calibration_forbidden": True,
    "unknown_native_target_policy": "masked_never_imputed_as_negative",
}


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Ictal gate-policy artifact is not canonical JSON data") from exc
    return (encoded + "\n").encode("utf-8")


def _receipt_sha256(value: object) -> str:
    """Match the historical typed-policy receipt hash (without a newline)."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Ictal gate policy is not canonical JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _finite(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or (
        minimum is not None and normalized < minimum
    ):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return normalized


@dataclass(frozen=True)
class IctalPromotionGatePolicy:
    """The one numerical policy frozen before any formal producer fit."""

    policy_document_sha256: str
    maximum_patient_macro_bce: float
    maximum_patient_macro_brier: float
    minimum_patient_macro_bce_improvement_over_prevalence: float
    minimum_patient_macro_brier_improvement_over_prevalence: float
    minimum_fold_patient_macro_ap_lift_over_prevalence: float
    minimum_shortcut_bce_improvement: float
    maximum_pairwise_scale_quantile_gap: float
    maximum_fold_identity_bootstrap_upper_95: float
    minimum_fold_identity_permutation_p_value: float
    scale_quantile_levels: tuple[float, ...] = ICTAL_SCALE_QUANTILE_LEVELS
    unknown_native_target_policy: str = "masked_never_imputed_as_negative"
    source_dev_probability_calibration_forbidden: bool = True
    schema_version: str = ICTAL_PROMOTION_GATE_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_document_sha256",
            _require_sha256(
                self.policy_document_sha256,
                field="policy_document_sha256",
            ),
        )
        values = {
            "maximum_patient_macro_bce": _finite(
                self.maximum_patient_macro_bce,
                field="maximum_patient_macro_bce",
                minimum=0.0,
            ),
            "maximum_patient_macro_brier": _finite(
                self.maximum_patient_macro_brier,
                field="maximum_patient_macro_brier",
                minimum=0.0,
            ),
            "minimum_patient_macro_bce_improvement_over_prevalence": _finite(
                self.minimum_patient_macro_bce_improvement_over_prevalence,
                field="minimum_patient_macro_bce_improvement_over_prevalence",
                minimum=0.0,
            ),
            "minimum_patient_macro_brier_improvement_over_prevalence": _finite(
                self.minimum_patient_macro_brier_improvement_over_prevalence,
                field="minimum_patient_macro_brier_improvement_over_prevalence",
                minimum=0.0,
            ),
            "minimum_fold_patient_macro_ap_lift_over_prevalence": _finite(
                self.minimum_fold_patient_macro_ap_lift_over_prevalence,
                field="minimum_fold_patient_macro_ap_lift_over_prevalence",
                minimum=0.0,
            ),
            "minimum_shortcut_bce_improvement": _finite(
                self.minimum_shortcut_bce_improvement,
                field="minimum_shortcut_bce_improvement",
                minimum=0.0,
            ),
            "maximum_pairwise_scale_quantile_gap": _finite(
                self.maximum_pairwise_scale_quantile_gap,
                field="maximum_pairwise_scale_quantile_gap",
                minimum=0.0,
            ),
            "maximum_fold_identity_bootstrap_upper_95": _finite(
                self.maximum_fold_identity_bootstrap_upper_95,
                field="maximum_fold_identity_bootstrap_upper_95",
                minimum=0.2,
            ),
            "minimum_fold_identity_permutation_p_value": _finite(
                self.minimum_fold_identity_permutation_p_value,
                field="minimum_fold_identity_permutation_p_value",
                minimum=0.0,
            ),
        }
        bounded = (
            "maximum_patient_macro_brier",
            "minimum_patient_macro_brier_improvement_over_prevalence",
            "minimum_fold_patient_macro_ap_lift_over_prevalence",
            "maximum_pairwise_scale_quantile_gap",
            "maximum_fold_identity_bootstrap_upper_95",
            "minimum_fold_identity_permutation_p_value",
        )
        if any(values[name] > 1.0 for name in bounded):
            raise ValueError("Probability-scale promotion thresholds must lie in [0,1]")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        levels = tuple(float(value) for value in self.scale_quantile_levels)
        if levels != ICTAL_SCALE_QUANTILE_LEVELS:
            raise ValueError("Ictal scale-alignment quantiles cannot change")
        object.__setattr__(self, "scale_quantile_levels", levels)
        if self.unknown_native_target_policy != "masked_never_imputed_as_negative":
            raise ValueError("Unknown native targets must remain masked")
        if self.source_dev_probability_calibration_forbidden is not True:
            raise ValueError("Source-dev ictal calibration promotion is forbidden")
        if self.schema_version != ICTAL_PROMOTION_GATE_POLICY_SCHEMA:
            raise ValueError("Unsupported ictal promotion policy schema")

    @property
    def receipt_sha256(self) -> str:
        return _receipt_sha256(asdict(self))


def _locked_policy() -> IctalPromotionGatePolicy:
    values = dict(_LOCKED_POLICY_DOCUMENT)
    document_schema = values.pop("schema_version")
    if document_schema != ICTAL_PROMOTION_GATE_POLICY_DOCUMENT_SCHEMA:
        raise RuntimeError("Locked ictal policy-document schema drifted")
    values["schema_version"] = ICTAL_PROMOTION_GATE_POLICY_SCHEMA
    values["scale_quantile_levels"] = tuple(values["scale_quantile_levels"])
    return IctalPromotionGatePolicy(
        policy_document_sha256=(
            ICTAL_LOCKED_PROMOTION_GATE_POLICY_DOCUMENT_SHA256
        ),
        **values,
    )


def _parse_json(raw: bytes, *, label: str) -> Mapping[str, object]:
    if not 1 <= len(raw) <= _MAX_POLICY_FILE_BYTES:
        raise ValueError(f"{label} has an invalid size")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _strict_bundle_directory(path: str | Path) -> Path:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("Ictal gate-policy bundle must be a regular directory")
    if {item.name for item in source.iterdir()} != {
        ICTAL_PROMOTION_GATE_POLICY_FILENAME,
        ICTAL_PROMOTION_GATE_POLICY_RECEIPT_FILENAME,
    }:
        raise ValueError("Ictal gate-policy bundle has missing or unknown files")
    return source


def _read_verified_bundle(
    path: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> tuple[Path, str, str, IctalPromotionGatePolicy]:
    source = _strict_bundle_directory(path)
    artifact_path = source / ICTAL_PROMOTION_GATE_POLICY_FILENAME
    receipt_path = source / ICTAL_PROMOTION_GATE_POLICY_RECEIPT_FILENAME
    for item, label in (
        (artifact_path, "gate_policy.json"),
        (receipt_path, "receipt.json"),
    ):
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"{label} must be a regular file")
    artifact_raw = artifact_path.read_bytes()
    receipt_raw = receipt_path.read_bytes()
    artifact_sha = hashlib.sha256(artifact_raw).hexdigest()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Ictal gate-policy artifact SHA mismatch")
    if receipt_sha != _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    ):
        raise ValueError("Ictal gate-policy bundle receipt SHA mismatch")
    artifact = _parse_json(artifact_raw, label="gate_policy.json")
    receipt = _parse_json(receipt_raw, label="receipt.json")
    if _canonical_json_bytes(artifact) != artifact_raw:
        raise ValueError("gate_policy.json is not canonical JSON")
    if _canonical_json_bytes(receipt) != receipt_raw:
        raise ValueError("receipt.json is not canonical JSON")
    if set(artifact) != _ARTIFACT_FIELDS or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("Ictal gate-policy bundle violates its closed schema")
    if artifact.get("schema_version") != ICTAL_PROMOTION_GATE_POLICY_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported ictal gate-policy artifact schema")
    if receipt.get("schema_version") != ICTAL_PROMOTION_GATE_POLICY_RECEIPT_SCHEMA:
        raise ValueError("Unsupported ictal gate-policy receipt schema")
    document_sha = _require_sha256(
        artifact.get("policy_document_sha256"),
        field="policy_document_sha256",
    )
    if document_sha != ICTAL_LOCKED_PROMOTION_GATE_POLICY_DOCUMENT_SHA256:
        raise ValueError("Ictal gate-policy source document is not the locked document")
    policy_payload = artifact.get("policy")
    if not isinstance(policy_payload, dict) or set(policy_payload) != _POLICY_FIELDS:
        raise ValueError("Ictal gate-policy payload violates its closed schema")
    try:
        policy = IctalPromotionGatePolicy(**policy_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ictal gate-policy payload is invalid") from exc
    locked = _locked_policy()
    if asdict(policy) != asdict(locked):
        raise ValueError("Ictal gate-policy thresholds differ from the locked policy")
    policy_receipt_sha = _require_sha256(
        artifact.get("policy_receipt_sha256"),
        field="policy_receipt_sha256",
    )
    if policy_receipt_sha != policy.receipt_sha256:
        raise ValueError("Ictal gate-policy typed receipt SHA mismatch")
    expected_receipt = {
        "schema_version": ICTAL_PROMOTION_GATE_POLICY_RECEIPT_SCHEMA,
        "artifact_sha256": artifact_sha,
        "policy_document_sha256": document_sha,
        "policy_receipt_sha256": policy_receipt_sha,
    }
    if dict(receipt) != expected_receipt:
        raise ValueError("Ictal gate-policy receipt does not bind its artifact")
    return source, artifact_sha, receipt_sha, policy


@dataclass(frozen=True, init=False)
class VerifiedIctalPromotionGatePolicyArtifact:
    """Opaque policy capability issued only by a strict two-file replay."""

    path: Path
    artifact_sha256: str
    receipt_sha256: str
    policy_document_sha256: str
    policy_receipt_sha256: str
    policy: IctalPromotionGatePolicy

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        artifact_sha256: str,
        receipt_sha256: str,
        policy: IctalPromotionGatePolicy,
    ) -> None:
        if _verification_marker is not _VERIFIED_POLICY_MARKER:
            raise TypeError(
                "VerifiedIctalPromotionGatePolicyArtifact can only be issued "
                "by the strict loader"
            )
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("Verified ictal gate-policy path must be absolute")
        if not isinstance(policy, IctalPromotionGatePolicy):
            raise TypeError("policy must be IctalPromotionGatePolicy")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "artifact_sha256",
            _require_sha256(artifact_sha256, field="artifact_sha256"),
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _require_sha256(receipt_sha256, field="receipt_sha256"),
        )
        object.__setattr__(
            self, "policy_document_sha256", policy.policy_document_sha256
        )
        object.__setattr__(self, "policy_receipt_sha256", policy.receipt_sha256)
        object.__setattr__(self, "policy", policy)

    def assert_unchanged(self) -> None:
        source, artifact_sha, receipt_sha, policy = _read_verified_bundle(
            self.path,
            expected_artifact_sha256=self.artifact_sha256,
            expected_receipt_sha256=self.receipt_sha256,
        )
        if (
            source != self.path
            or artifact_sha != self.artifact_sha256
            or receipt_sha != self.receipt_sha256
            or asdict(policy) != asdict(self.policy)
            or policy.receipt_sha256 != self.policy_receipt_sha256
        ):
            raise ValueError("Verified ictal gate-policy artifact changed after load")


def load_ictal_promotion_gate_policy(
    path: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedIctalPromotionGatePolicyArtifact:
    """Strictly load the unique policy; both external hashes are mandatory."""

    source, artifact_sha, receipt_sha, policy = _read_verified_bundle(
        path,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    return VerifiedIctalPromotionGatePolicyArtifact(
        _verification_marker=_VERIFIED_POLICY_MARKER,
        path=source,
        artifact_sha256=artifact_sha,
        receipt_sha256=receipt_sha,
        policy=policy,
    )


def _safe_output_directory(path: str | Path) -> Path:
    target = Path(os.path.abspath(path))
    if target.name in {"", ".", ".."}:
        raise ValueError("Ictal gate-policy output requires a concrete directory")
    for component in (target.parent, *target.parent.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("Ictal gate-policy output cannot traverse symlinks")
    if not target.parent.is_dir():
        raise FileNotFoundError("Ictal gate-policy output parent does not exist")
    if os.path.lexists(target):
        raise FileExistsError(f"Ictal gate-policy output already exists: {target}")
    return target


def _read_locked_policy_document(path: str | Path) -> IctalPromotionGatePolicy:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise ValueError("Ictal gate-policy source must be a regular file")
    raw = source.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != ICTAL_LOCKED_PROMOTION_GATE_POLICY_DOCUMENT_SHA256:
        raise ValueError("Ictal gate-policy source-document SHA mismatch")
    value = _parse_json(raw, label="ictal promotion policy document")
    if set(value) != _POLICY_DOCUMENT_FIELDS:
        raise ValueError("Ictal gate-policy source document violates its closed schema")
    if dict(value) != _LOCKED_POLICY_DOCUMENT:
        raise ValueError("Ictal gate-policy source document changed locked thresholds")
    return _locked_policy()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize_ictal_promotion_gate_policy(
    *,
    policy_document_path: str | Path,
    output_directory: str | Path,
) -> VerifiedIctalPromotionGatePolicyArtifact:
    """Atomically issue and independently replay the code-pinned policy bundle.

    No numerical threshold is accepted by this API.  ``policy_document_path``
    must have the repository-pinned byte hash and exact closed content.
    """

    policy = _read_locked_policy_document(policy_document_path)
    target = _safe_output_directory(output_directory)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        artifact_payload = {
            "schema_version": ICTAL_PROMOTION_GATE_POLICY_ARTIFACT_SCHEMA,
            "policy_document_sha256": policy.policy_document_sha256,
            "policy_receipt_sha256": policy.receipt_sha256,
            "policy": asdict(policy),
        }
        artifact_raw = _canonical_json_bytes(artifact_payload)
        artifact_sha = hashlib.sha256(artifact_raw).hexdigest()
        receipt_payload = {
            "schema_version": ICTAL_PROMOTION_GATE_POLICY_RECEIPT_SCHEMA,
            "artifact_sha256": artifact_sha,
            "policy_document_sha256": policy.policy_document_sha256,
            "policy_receipt_sha256": policy.receipt_sha256,
        }
        receipt_raw = _canonical_json_bytes(receipt_payload)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        artifact_path = temporary / ICTAL_PROMOTION_GATE_POLICY_FILENAME
        receipt_path = temporary / ICTAL_PROMOTION_GATE_POLICY_RECEIPT_FILENAME
        artifact_path.write_bytes(artifact_raw)
        receipt_path.write_bytes(receipt_raw)
        _fsync_file(artifact_path)
        _fsync_file(receipt_path)
        _fsync_directory(temporary)
        load_ictal_promotion_gate_policy(
            temporary,
            expected_artifact_sha256=artifact_sha,
            expected_receipt_sha256=receipt_sha,
        )
        if os.path.lexists(target):
            raise FileExistsError(
                f"Ictal gate-policy output already exists: {target}"
            )
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        return load_ictal_promotion_gate_policy(
            target,
            expected_artifact_sha256=artifact_sha,
            expected_receipt_sha256=receipt_sha,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = [
    "ICTAL_LOCKED_PROMOTION_GATE_POLICY_DOCUMENT_RELATIVE_PATH",
    "ICTAL_LOCKED_PROMOTION_GATE_POLICY_DOCUMENT_SHA256",
    "ICTAL_PROMOTION_GATE_POLICY_ARTIFACT_SCHEMA",
    "ICTAL_PROMOTION_GATE_POLICY_DOCUMENT_SCHEMA",
    "ICTAL_PROMOTION_GATE_POLICY_RECEIPT_SCHEMA",
    "ICTAL_PROMOTION_GATE_POLICY_SCHEMA",
    "ICTAL_SCALE_QUANTILE_LEVELS",
    "IctalPromotionGatePolicy",
    "VerifiedIctalPromotionGatePolicyArtifact",
    "load_ictal_promotion_gate_policy",
    "materialize_ictal_promotion_gate_policy",
]
