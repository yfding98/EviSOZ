"""Materialize field releases and training envelopes for real private events."""

from __future__ import annotations

from collections import Counter
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    build_raw_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
)
from src.evisoz.data.dataset_policy import (
    CATEGORICAL_LABEL_VALUE_SCHEMA_VERSION,
    CHANNEL_SET_VALUE_SCHEMA_VERSION,
    NODE_LABEL_VALUE_SCHEMA_VERSION,
    REGION_SET_VALUE_SCHEMA_VERSION,
    REPORT_TEXT_VALUE_SCHEMA_VERSION,
    build_field_release,
)
from src.evisoz.data.event_identity import validate_event_identity
from src.evisoz.data.private_stage0_split import (
    build_private_patient_linkage_group,
)
from src.evisoz.data.private_training_authorization import (
    PRIVATE_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
    validate_private_training_authorization,
)
from src.evisoz.data.split_ledger import (
    SPLIT_ROSTER_SCHEMA_VERSION,
    validate_split_roster,
)
from src.evisoz.data.tcp22_views import validate_montage_derivation_receipt
from src.evisoz.forge.training_example import (
    TRAINING_EXAMPLE_SCHEMA_VERSION,
    build_training_example,
    validate_training_example,
)
from src.soz.geometry import STANDARD_19


PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION = (
    "evisoz_private_real_stage0_examples_materialization_v2"
)
LEGACY_PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION = (
    "evisoz_private_real_stage0_examples_materialization_v1"
)
_FIELD_RELEASE_SCHEMA_VERSION = "evisoz_field_release_v1"
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_HASH_PLACEHOLDER = "0" * 64
_ALLOWED_REGIONS = {
    "left_frontal",
    "right_frontal",
    "left_temporal",
    "right_temporal",
    "central_parietal",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("private Stage-0 example JSON input must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError("private Stage-0 example JSON input must be an object")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("private Stage-0 example CSV input must be a regular file")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("private Stage-0 example CSV input is empty")
    return rows


def _safe_relative_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise TypeError("private Stage-0 relative path must be a string")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError("private Stage-0 relative path is unsafe")
    path = root.joinpath(*rel.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError("private Stage-0 referenced artifact must be a regular file")
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    resolved.relative_to(resolved_root)
    return resolved


def _patient_authority(
    source_rows: Sequence[Mapping[str, str]],
    signal_rows: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    raw_patients = sorted(
        {str(row.get("base_patient_id", "")).strip() for row in source_rows}
    )
    if not raw_patients or "" in raw_patients:
        raise ValueError("private source manifest patient authority is invalid")
    result = {
        name: f"PRIV-P{index + 1:03d}"
        for index, name in enumerate(raw_patients)
    }
    if {row.get("patient_id") for row in signal_rows} != set(result.values()):
        raise ValueError("private source patient order does not reproduce signal pseudonyms")
    return result


def _dataset_capability(
    split_roster: Mapping[str, object],
    *,
    private_training_authorization: Mapping[str, object] | None = None,
) -> dict[str, object]:
    authorized_ports_by_field: dict[str, set[str]] = {}
    if private_training_authorization is not None:
        scope = private_training_authorization["field_scope"]
        authorized_ports_by_field = {
            str(row["field_id"]): set(row["loss_ports"])
            for row in scope["field_permissions"]
        }

    def losses(field_id: str) -> dict[str, bool]:
        ports = authorized_ports_by_field.get(field_id, set())
        return {
            "typed_slot_loss": "typed_slot_loss" in ports,
            "node_localization_loss": "node_localization_loss" in ports,
            "report_text_loss": False,
        }

    def row(
        field_id: str,
        field_path: str,
        role: str,
        schema: str,
        losses: Mapping[str, bool],
        *,
        report_target: bool,
    ) -> dict[str, object]:
        return {
            "field_id": field_id,
            "field_path": field_path,
            "semantic_role": role,
            "payload_schema_version": schema,
            "allowed_roles": ["development_cv", "locked_test"],
            "loss_allowed": dict(losses),
            "report_target_allowed": report_target,
            "prompt_or_rag_allowed": False,
        }

    rows = [
        row(
            "PRIVATE-DIFFUSE-SPREAD",
            "clinical_labels.diffuse_spread",
            "spread",
            CATEGORICAL_LABEL_VALUE_SCHEMA_VERSION,
            losses("PRIVATE-DIFFUSE-SPREAD"),
            report_target=True,
        ),
        row(
            "PRIVATE-EARLY-SPREAD-NODES",
            "clinical_labels.early_spread_channels",
            "spread",
            CHANNEL_SET_VALUE_SCHEMA_VERSION,
            losses("PRIVATE-EARLY-SPREAD-NODES"),
            report_target=True,
        ),
        row(
            "PRIVATE-EVOLUTION",
            "clinical_labels.evolution",
            "evolution",
            CATEGORICAL_LABEL_VALUE_SCHEMA_VERSION,
            losses("PRIVATE-EVOLUTION"),
            report_target=True,
        ),
        row(
            "PRIVATE-LATERALITY",
            "clinical_labels.laterality",
            "laterality_label",
            CATEGORICAL_LABEL_VALUE_SCHEMA_VERSION,
            losses("PRIVATE-LATERALITY"),
            report_target=True,
        ),
        row(
            "PRIVATE-LOCALIZABILITY",
            "clinical_labels.localizability",
            "localizability",
            CATEGORICAL_LABEL_VALUE_SCHEMA_VERSION,
            losses("PRIVATE-LOCALIZABILITY"),
            report_target=True,
        ),
        row(
            "PRIVATE-MORPHOLOGY",
            "clinical_labels.morphology",
            "morphology",
            CATEGORICAL_LABEL_VALUE_SCHEMA_VERSION,
            losses("PRIVATE-MORPHOLOGY"),
            report_target=True,
        ),
        row(
            "PRIVATE-ONSET-NODES",
            "clinical_labels.onset_candidate_channels",
            "node_label",
            NODE_LABEL_VALUE_SCHEMA_VERSION,
            losses("PRIVATE-ONSET-NODES"),
            report_target=True,
        ),
        row(
            "PRIVATE-ONSET-REGIONS",
            "clinical_labels.onset_candidate_regions",
            "region_label",
            REGION_SET_VALUE_SCHEMA_VERSION,
            losses("PRIVATE-ONSET-REGIONS"),
            report_target=True,
        ),
        row(
            "PRIVATE-PHYSICIAN-REPORT-TEXT",
            "physician_report.text",
            "text",
            REPORT_TEXT_VALUE_SCHEMA_VERSION,
            losses("PRIVATE-PHYSICIAN-REPORT-TEXT"),
            report_target=True,
        ),
        row(
            "PRIVATE-QUALITY",
            "clinical_labels.signal_quality",
            "quality",
            CATEGORICAL_LABEL_VALUE_SCHEMA_VERSION,
            losses("PRIVATE-QUALITY"),
            report_target=True,
        ),
    ]
    rows.sort(key=lambda item: str(item["field_id"]))
    return {
        "dataset_id": "private",
        "patient_roster_sha256": split_roster["receipt_sha256"],
        "field_roster": rows,
    }


def _unavailable(
    capability: Mapping[str, object],
    *,
    state: str = "not_provided",
    authority: str = "physician",
) -> dict[str, object]:
    return {
        "field_id": capability["field_id"],
        "field_path": capability["field_path"],
        "state": state,
        "authority": authority,
        "quality_tier": "not_applicable",
        "semantic_role": capability["semantic_role"],
        "value_ref": None,
        "claim_permission": "none",
        "loss_permissions": {
            "typed_slot_loss": False,
            "node_localization_loss": False,
            "report_text_loss": False,
        },
    }


def _provided(
    capability: Mapping[str, object],
    payload: Mapping[str, object],
    trusted_values: dict[str, object],
    *,
    authority: str,
    quality_tier: str,
    claim_permission: str,
    typed_slot_loss: bool,
    node_localization_loss: bool = False,
) -> dict[str, object]:
    ref = build_json_artifact_ref(
        payload,
        artifact_kind="field_value",
        payload_schema_version=str(capability["payload_schema_version"]),
    )
    trusted_values[ref["artifact_id"]] = deepcopy(dict(payload))
    return {
        "field_id": capability["field_id"],
        "field_path": capability["field_path"],
        "state": "provided",
        "authority": authority,
        "quality_tier": quality_tier,
        "semantic_role": capability["semantic_role"],
        "value_ref": ref,
        "claim_permission": claim_permission,
        "loss_permissions": {
            "typed_slot_loss": typed_slot_loss,
            "node_localization_loss": node_localization_loss,
            "report_text_loss": False,
        },
    }


def _parse_channel_list(value: object, context: str) -> list[str]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context} is not a JSON channel array") from exc
    if not isinstance(raw, list):
        raise ValueError(f"{context} is not a channel array")
    channels = sorted({str(item) for item in raw})
    if len(channels) != len(raw) or any(channel not in STANDARD_19 for channel in channels):
        raise ValueError(f"{context} contains duplicate or non-Standard19 channels")
    return channels


def _direct_fields(
    *,
    target: Mapping[str, str],
    source: Mapping[str, str],
    capability: Mapping[str, object],
    role: str,
    private_training_authorization: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_id = {row["field_id"]: row for row in capability["field_roster"]}
    trusted: dict[str, object] = {}
    confidence = float(source.get("label_confidence", "nan"))
    high_confidence = confidence == 1.0
    quality = "gold_lite" if high_confidence else "uncertain"
    claim = "direct" if high_confidence else "none"

    authorized_ports_by_field: dict[str, set[str]] = {}
    if private_training_authorization is not None:
        authorized_ports_by_field = {
            str(row["field_id"]): set(row["loss_ports"])
            for row in private_training_authorization["field_scope"][
                "field_permissions"
            ]
        }

    def field_losses(field_id: str) -> tuple[bool, bool]:
        # Only high-confidence development rows may emit loss. The external
        # receipt is additionally field/port scoped; evaluator-only and
        # locked-test rows remain completely loss-closed.
        if role != "development_cv" or not high_confidence:
            return False, False
        ports = authorized_ports_by_field.get(field_id, set())
        return "typed_slot_loss" in ports, "node_localization_loss" in ports

    fields: list[dict[str, object]] = []

    onset = _parse_channel_list(
        target.get("standard19_positive_electrodes"),
        "standard19_positive_electrodes",
    )
    onset_cap = by_id["PRIVATE-ONSET-NODES"]
    if onset:
        typed_loss, node_loss = field_losses("PRIVATE-ONSET-NODES")
        fields.append(
            _provided(
                onset_cap,
                {"values": onset, "semantics": "incomplete_positive"},
                trusted,
                authority="physician",
                quality_tier=quality,
                claim_permission=claim,
                typed_slot_loss=typed_loss,
                node_localization_loss=node_loss,
            )
        )
    else:
        fields.append(_unavailable(onset_cap))

    laterality_cap = by_id["PRIVATE-LATERALITY"]
    hemisphere = source.get("hemisphere")
    if hemisphere in {"L", "R"}:
        typed_loss, _ = field_losses("PRIVATE-LATERALITY")
        fields.append(
            _provided(
                laterality_cap,
                {
                    "value": "left" if hemisphere == "L" else "right",
                    "certainty": "high" if high_confidence else "low",
                },
                trusted,
                authority="physician",
                quality_tier=quality,
                claim_permission=claim,
                typed_slot_loss=typed_loss,
            )
        )
    else:
        fields.append(_unavailable(laterality_cap))

    regions_cap = by_id["PRIVATE-ONSET-REGIONS"]
    regions = sorted(
        {item for item in str(source.get("regions", "")).split(";") if item}
    )
    if any(region not in _ALLOWED_REGIONS for region in regions):
        raise ValueError("private onset region vocabulary drifted")
    if regions:
        typed_loss, _ = field_losses("PRIVATE-ONSET-REGIONS")
        fields.append(
            _provided(
                regions_cap,
                {"values": regions, "semantics": "incomplete_positive"},
                trusted,
                authority="dataset_direct",
                quality_tier="silver" if high_confidence else "uncertain",
                claim_permission=claim,
                typed_slot_loss=typed_loss,
            )
        )
    else:
        fields.append(_unavailable(regions_cap, authority="dataset_direct"))

    spread_cap = by_id["PRIVATE-EARLY-SPREAD-NODES"]
    spread = _parse_channel_list(
        target.get("known_spread_electrodes"),
        "known_spread_electrodes",
    )
    if spread:
        typed_loss, _ = field_losses("PRIVATE-EARLY-SPREAD-NODES")
        fields.append(
            _provided(
                spread_cap,
                {"values": spread, "semantics": "incomplete_positive"},
                trusted,
                authority="physician",
                quality_tier=quality,
                claim_permission=claim,
                typed_slot_loss=typed_loss,
            )
        )
    else:
        fields.append(_unavailable(spread_cap))

    diffuse_cap = by_id["PRIVATE-DIFFUSE-SPREAD"]
    diffuse = target.get("diffuse_spread_present")
    if diffuse == "1":
        typed_loss, _ = field_losses("PRIVATE-DIFFUSE-SPREAD")
        fields.append(
            _provided(
                diffuse_cap,
                {
                    "value": "diffuse_spread_present",
                    "certainty": "high" if high_confidence else "low",
                },
                trusted,
                authority="physician",
                quality_tier=quality,
                claim_permission=claim,
                typed_slot_loss=typed_loss,
            )
        )
    elif diffuse == "0":
        # Lack of a DIFFUSE token is not a physician assertion of absence.
        fields.append(_unavailable(diffuse_cap))
    else:
        raise ValueError("private diffuse-spread flag is not binary")

    for field_id, authority in (
        ("PRIVATE-EVOLUTION", "physician"),
        ("PRIVATE-LOCALIZABILITY", "physician"),
        ("PRIVATE-MORPHOLOGY", "physician"),
        ("PRIVATE-PHYSICIAN-REPORT-TEXT", "physician_authored_text"),
        ("PRIVATE-QUALITY", "dataset_direct"),
    ):
        fields.append(_unavailable(by_id[field_id], authority=authority))
    fields.sort(key=lambda row: str(row["field_id"]))
    return fields, trusted


def _raw_ref(path: Path, *, kind: str, media_type: str) -> dict[str, Any]:
    return build_raw_artifact_ref(
        path.read_bytes(),
        artifact_kind=kind,
        media_type=media_type,
    )


def _manifest_hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def materialize_private_stage0_examples(
    *,
    real_cohort_root: Path,
    split_roster_path: Path,
    signal_roster_path: Path,
    target_ledger_path: Path,
    source_manifest_path: Path,
    output: Path,
    limit: int | None = None,
    private_training_authorization_path: Path | None = None,
) -> dict[str, Any]:
    """Materialize real private field releases and training envelopes.

    Without ``private_training_authorization_path`` this intentionally emits
    an evaluator-only v2 manifest.  With it, the external receipt is replayed
    against every source binding before any field-level loss permission can be
    enabled.  The report-text authorization is a separate contract and is not
    accepted here.
    """

    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    cohort_root = real_cohort_root.resolve(strict=True)
    if cohort_root.is_symlink() or not cohort_root.is_dir():
        raise ValueError("private real Stage-0 cohort root must be a regular directory")
    cohort_manifest_path = cohort_root / "manifest.json"
    cohort = _json(cohort_manifest_path)
    if cohort.get("status") != "completed_real_private_stage0_materialization":
        raise ValueError("private real Stage-0 cohort is incomplete")
    split = _json(split_roster_path.resolve(strict=True))
    signal_rows = _csv(signal_roster_path.resolve(strict=True))
    target_rows = _csv(target_ledger_path.resolve(strict=True))
    source_rows = _csv(source_manifest_path.resolve(strict=True))
    patient_authority = _patient_authority(source_rows, signal_rows)
    groups_by_patient = {
        patient_id: build_private_patient_linkage_group(patient_id)
        for patient_id in patient_authority.values()
    }
    trusted_groups = {
        group["linkage_group_id"]: group for group in groups_by_patient.values()
    }
    split = validate_split_roster(split, trusted_linkage_groups=trusted_groups)
    assignment_by_group = {
        row["linkage_group_id"]: row for row in split["assignments"]
    }
    source_binding_refs: dict[str, object] = {
        "split_roster_ref": build_json_artifact_ref(
            split,
            artifact_kind="split_roster",
            payload_schema_version=SPLIT_ROSTER_SCHEMA_VERSION,
        ),
        "signal_roster_ref": _raw_ref(
            signal_roster_path,
            kind="private_signal_roster",
            media_type="text/csv",
        ),
        "target_ledger_ref": _raw_ref(
            target_ledger_path,
            kind="private_target_ledger",
            media_type="text/csv",
        ),
        "source_manifest_ref": _raw_ref(
            source_manifest_path,
            kind="private_label_authority_manifest",
            media_type="text/csv",
        ),
        "dataset_id": "private",
        "patient_roster_sha256": split["receipt_sha256"],
    }
    private_training_authorization: dict[str, Any] | None = None
    private_training_authorization_ref: dict[str, Any] | None = None
    if private_training_authorization_path is not None:
        authorization_path = private_training_authorization_path.resolve(strict=True)
        if authorization_path.is_symlink() or not authorization_path.is_file():
            raise ValueError("private training authorization must be a regular JSON file")
        authorization_bytes = authorization_path.read_bytes()
        try:
            authorization_value = json.loads(authorization_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("private training authorization is not strict UTF-8 JSON") from exc
        private_training_authorization = validate_private_training_authorization(
            authorization_value,
            expected_bindings=source_binding_refs,
            expected_field_ids={
                str(row["field_id"])
                for row in _dataset_capability(split)["field_roster"]
            },
        )
        private_training_authorization_ref = build_raw_artifact_ref(
            authorization_bytes,
            artifact_kind="private_training_authorization",
            media_type="application/json",
            payload_schema_version=PRIVATE_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
        )
    capability = _dataset_capability(
        split,
        private_training_authorization=private_training_authorization,
    )
    signal_by_event = {row["event_id"]: row for row in signal_rows}
    target_by_event = {row["event_id"]: row for row in target_rows}
    if len(signal_by_event) != len(signal_rows) or len(target_by_event) != len(target_rows):
        raise ValueError("private Stage-0 signal/target ledgers contain duplicate event IDs")
    if set(signal_by_event) != set(target_by_event):
        raise ValueError("private Stage-0 signal/target event ledgers drifted")
    events = list(cohort.get("events", []))
    if not events:
        raise ValueError("private real Stage-0 cohort has no materialized events")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("private Stage-0 example limit must be a positive integer")
        events = events[:limit]
    output.mkdir(parents=True)
    manifest_rows: list[dict[str, object]] = []
    loss_event_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    field_provided_counts: Counter[str] = Counter()
    field_loss_counts: Counter[str] = Counter()
    for cohort_row in events:
        event_id = str(cohort_row["event_id"])
        if event_id not in signal_by_event:
            raise ValueError("real private event is absent from the frozen signal ledger")
        signal = signal_by_event[event_id]
        target = target_by_event[event_id]
        patient_id = str(signal["patient_id"])
        if target["patient_id"] != patient_id or cohort_row["patient_id"] != patient_id:
            raise ValueError("private event patient binding drifted across ledgers")
        source_index = int(signal["source_row"]) - 1
        if source_index < 0 or source_index >= len(source_rows):
            raise ValueError("private event source row is outside the source manifest")
        source = source_rows[source_index]
        expected_patient = patient_authority[str(source["base_patient_id"]).strip()]
        if patient_id != expected_patient:
            raise ValueError("private event source row does not reproduce patient pseudonym")
        group = groups_by_patient[patient_id]
        group_id = group["linkage_group_id"]
        if cohort_row["linkage_group_id"] != group_id:
            raise ValueError("private event linkage group drifted from frozen patient authority")
        assignment = assignment_by_group[group_id]
        if (
            cohort_row["evisoz_role"] != assignment["evisoz_role"]
            or cohort_row["outer_holdout_fold"] != assignment["outer_holdout_fold"]
        ):
            raise ValueError("private event split binding drifted")
        cache_root = cohort_root.joinpath(
            *PurePosixPath(str(cohort_row["relative_cache_path"])).parts
        )
        identity_path = _safe_relative_file(
            cache_root,
            "sidecars/event_identity.json",
        )
        montage_path = _safe_relative_file(
            cache_root,
            "sidecars/montage_receipt.json",
        )
        identity = validate_event_identity(_json(identity_path))
        montage = validate_montage_derivation_receipt(
            _json(montage_path),
            trusted_event_identity=identity,
        )
        if (
            identity["event_id"] != event_id
            or identity["linkage_group_id"] != group_id
            or identity["source_patient_sha256"]
            != group["members"][0]["source_patient_sha256"]
        ):
            raise ValueError("private event identity does not bind its cohort row")
        fields, trusted_values = _direct_fields(
            target=target,
            source=source,
            capability=capability,
            role=str(assignment["evisoz_role"]),
            private_training_authorization=private_training_authorization,
        )
        release = build_field_release(
            dataset_id="private",
            sample_id=identity["sample_id"],
            report_scope="full_soz",
            fields=fields,
            event_identity=identity,
            dataset_capability=capability,
            trusted_values_by_artifact_id=trusted_values,
        )
        example = build_training_example(
            sample_id=identity["sample_id"],
            event_id=event_id,
            dataset_id="private",
            linkage_group_id=group_id,
            anchor_quality=identity["anchor"]["quality"],
            event_identity=identity,
            split_roster=split,
            trusted_linkage_groups=trusted_groups,
            montage_receipt=montage,
            field_release=release,
        )
        event_output = output / "events" / event_id
        event_output.mkdir(parents=True)
        release_path = event_output / "field_release.json"
        example_path = event_output / "training_example.json"
        release_path.write_text(
            json.dumps(release, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        example_path.write_text(
            json.dumps(example, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        provided_ids = sorted(
            row["field_id"] for row in release["fields"] if row["state"] == "provided"
        )
        loss_ids = sorted(
            row["field_id"]
            for row in release["fields"]
            if any(row["loss_permissions"].values())
        )
        for field_id in provided_ids:
            field_provided_counts[field_id] += 1
        for field_id in loss_ids:
            field_loss_counts[field_id] += 1
        for port in example["enabled_loss_ports"]:
            loss_event_counts[port] += 1
        role_counts[str(assignment["evisoz_role"])] += 1
        manifest_rows.append(
            {
                "event_id": event_id,
                "sample_id": identity["sample_id"],
                "linkage_group_id": group_id,
                "evisoz_role": assignment["evisoz_role"],
                "outer_holdout_fold": assignment["outer_holdout_fold"],
                "field_release_ref": build_json_artifact_ref(
                    release,
                    artifact_kind="field_release",
                    payload_schema_version=_FIELD_RELEASE_SCHEMA_VERSION,
                ),
                "training_example_ref": build_json_artifact_ref(
                    example,
                    artifact_kind="training_example",
                    payload_schema_version=TRAINING_EXAMPLE_SCHEMA_VERSION,
                ),
                "provided_field_ids": provided_ids,
                "loss_enabled_field_ids": loss_ids,
                "enabled_loss_ports": example["enabled_loss_ports"],
                "relative_field_release_path": (
                    f"events/{event_id}/field_release.json"
                ),
                "relative_training_example_path": (
                    f"events/{event_id}/training_example.json"
                ),
            }
        )
    manifest_rows.sort(key=lambda row: str(row["event_id"]))
    manifest: dict[str, Any] = {
        "schema_version": PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION,
        "status": "completed_private_real_stage0_field_and_envelope_materialization",
        "source_bindings": {
            "real_cohort_manifest_ref": build_json_artifact_ref(
                cohort,
                artifact_kind="private_real_stage0_cohort_manifest",
                payload_schema_version=str(cohort["schema_version"]),
            ),
            "split_roster_ref": build_json_artifact_ref(
                split,
                artifact_kind="split_roster",
                payload_schema_version=SPLIT_ROSTER_SCHEMA_VERSION,
            ),
            "signal_roster_ref": _raw_ref(
                signal_roster_path,
                kind="private_signal_roster",
                media_type="text/csv",
            ),
            "target_ledger_ref": _raw_ref(
                target_ledger_path,
                kind="private_target_ledger",
                media_type="text/csv",
            ),
            "source_manifest_ref": _raw_ref(
                source_manifest_path,
                kind="private_label_authority_manifest",
                media_type="text/csv",
            ),
            "private_training_authorization_ref": private_training_authorization_ref,
        },
        "events": manifest_rows,
        "counts": {
            "event_count": len(manifest_rows),
            "role_event_counts": dict(sorted(role_counts.items())),
            "enabled_loss_port_event_counts": dict(sorted(loss_event_counts.items())),
            "provided_field_event_counts": dict(sorted(field_provided_counts.items())),
            "loss_enabled_field_event_counts": dict(sorted(field_loss_counts.items())),
            "physician_report_text_provided_count": 0,
            "physician_report_text_training_count": 0,
        },
        "release_policy": {
            "private_training_authority_present": private_training_authorization is not None,
            "private_training_authority_status": (
                "validated_external"
                if private_training_authorization is not None
                else "not_provided"
            ),
            "authorized_field_ids": sorted(
                row["field_id"]
                for row in (
                    private_training_authorization["field_scope"]["field_permissions"]
                    if private_training_authorization is not None
                    else []
                )
            ),
            "authorized_loss_ports": sorted(
                {
                    port
                    for row in (
                        private_training_authorization["field_scope"]["field_permissions"]
                        if private_training_authorization is not None
                        else []
                    )
                    for port in row["loss_ports"]
                }
            ),
            "direct_fields_are_evaluator_only": private_training_authorization is None,
            "development_high_confidence_direct_fields_can_train": private_training_authorization is not None,
            "low_confidence_fields_are_visible_but_loss_disabled": True,
            "locked_test_fields_can_train": False,
            "empty_positive_set_is_not_converted_to_negative": True,
            "missing_diffuse_token_is_not_converted_to_absence": True,
            "morphology_evolution_localizability_quality_unprovided": True,
            "physician_report_text_unprovided_pending_deidentification": True,
            "generated_text_can_supervise_localization": False,
            "report_text_can_supervise_localization": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    manifest["receipt_sha256"] = canonical_json_sha256(
        _manifest_hash_source(manifest)
    )
    return validate_private_stage0_examples_materialization(
        manifest,
        output_root=output,
        split_roster=split,
        trusted_groups=trusted_groups,
        cohort_root=cohort_root,
        private_training_authorization=private_training_authorization,
        private_training_authorization_bytes=(
            authorization_bytes if private_training_authorization_path is not None else None
        ),
        expected_source_bindings={
            **source_binding_refs,
            "real_cohort_manifest_ref": build_json_artifact_ref(
                cohort,
                artifact_kind="private_real_stage0_cohort_manifest",
                payload_schema_version=str(cohort["schema_version"]),
            ),
        },
    )


def validate_private_stage0_examples_materialization(
    value: object,
    *,
    output_root: Path,
    split_roster: Mapping[str, object],
    trusted_groups: Mapping[str, Mapping[str, object]],
    cohort_root: Path,
    private_training_authorization: Mapping[str, object] | None = None,
    private_training_authorization_bytes: bytes | None = None,
    expected_source_bindings: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Reopen and validate every real private field/envelope pair."""

    if type(value) is not dict or set(value) != {
        "schema_version",
        "status",
        "source_bindings",
        "events",
        "counts",
        "release_policy",
        "receipt_sha256",
    }:
        raise ValueError("private Stage-0 example materialization fields drifted")
    data = deepcopy(value)
    schema_version = data["schema_version"]
    if schema_version not in {
        PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION,
        LEGACY_PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION,
    } or data[
        "status"
    ] != "completed_private_real_stage0_field_and_envelope_materialization":
        raise ValueError("private Stage-0 example materialization status drifted")
    is_v2 = schema_version == PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION
    source_bindings = data["source_bindings"]
    expected_source_keys = {
        "real_cohort_manifest_ref",
        "split_roster_ref",
        "signal_roster_ref",
        "target_ledger_ref",
        "source_manifest_ref",
    }
    if is_v2:
        expected_source_keys.add("private_training_authorization_ref")
    if type(source_bindings) is not dict or set(source_bindings) != expected_source_keys:
        raise ValueError("private Stage-0 source bindings drifted")
    if source_bindings["split_roster_ref"] != build_json_artifact_ref(
        split_roster,
        artifact_kind="split_roster",
        payload_schema_version=SPLIT_ROSTER_SCHEMA_VERSION,
    ):
        raise ValueError("private Stage-0 split roster binding drifted")
    for key in (
        "real_cohort_manifest_ref",
        "split_roster_ref",
        "signal_roster_ref",
        "target_ledger_ref",
        "source_manifest_ref",
    ):
        validate_artifact_ref(source_bindings[key])
    if is_v2 and source_bindings["private_training_authorization_ref"] is not None:
        validate_artifact_ref(source_bindings["private_training_authorization_ref"])
    if expected_source_bindings is not None:
        for key, expected in expected_source_bindings.items():
            if key in source_bindings and source_bindings[key] != expected:
                raise ValueError(f"private Stage-0 {key} binding drifted")

    if is_v2:
        authorization_ref = source_bindings["private_training_authorization_ref"]
        if private_training_authorization is None:
            if authorization_ref is not None:
                raise ValueError("private training authorization receipt is required to replay v2")
            if private_training_authorization_bytes is not None:
                raise ValueError("private training authorization bytes supplied without receipt")
        else:
            if authorization_ref is None or private_training_authorization_bytes is None:
                raise ValueError("v2 authorized materialization must bind raw authorization bytes")
            expected_auth_ref = build_raw_artifact_ref(
                private_training_authorization_bytes,
                artifact_kind="private_training_authorization",
                media_type="application/json",
                payload_schema_version=PRIVATE_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
            )
            if authorization_ref != expected_auth_ref:
                raise ValueError("private training authorization artifact reference drifted")
            private_training_authorization = validate_private_training_authorization(
                private_training_authorization,
                expected_bindings={
                    "dataset_id": "private",
                    "patient_roster_sha256": split_roster["receipt_sha256"],
                    "split_roster_ref": source_bindings["split_roster_ref"],
                    "signal_roster_ref": source_bindings["signal_roster_ref"],
                    "target_ledger_ref": source_bindings["target_ledger_ref"],
                    "source_manifest_ref": source_bindings["source_manifest_ref"],
                },
                expected_field_ids={
                    "PRIVATE-DIFFUSE-SPREAD",
                    "PRIVATE-EARLY-SPREAD-NODES",
                    "PRIVATE-EVOLUTION",
                    "PRIVATE-LATERALITY",
                    "PRIVATE-LOCALIZABILITY",
                    "PRIVATE-MORPHOLOGY",
                    "PRIVATE-ONSET-NODES",
                    "PRIVATE-ONSET-REGIONS",
                    "PRIVATE-PHYSICIAN-REPORT-TEXT",
                    "PRIVATE-QUALITY",
                },
            )
    elif private_training_authorization is not None or private_training_authorization_bytes is not None:
        raise ValueError("legacy v1 materialization cannot carry a training authorization")
    roster = validate_split_roster(
        split_roster,
        trusted_linkage_groups=trusted_groups,
    )
    rows = data["events"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("private Stage-0 example materialization has no events")
    if rows != sorted(rows, key=lambda row: row["event_id"]) or len(
        {row["event_id"] for row in rows}
    ) != len(rows):
        raise ValueError("private Stage-0 example rows are not uniquely sorted")
    role_counts: Counter[str] = Counter()
    loss_counts: Counter[str] = Counter()
    provided_counts: Counter[str] = Counter()
    field_loss_counts: Counter[str] = Counter()
    for row in rows:
        if type(row) is not dict or set(row) != {
            "event_id",
            "sample_id",
            "linkage_group_id",
            "evisoz_role",
            "outer_holdout_fold",
            "field_release_ref",
            "training_example_ref",
            "provided_field_ids",
            "loss_enabled_field_ids",
            "enabled_loss_ports",
            "relative_field_release_path",
            "relative_training_example_path",
        }:
            raise ValueError("private Stage-0 example row fields drifted")
        release_path = _safe_relative_file(
            output_root,
            row["relative_field_release_path"],
        )
        example_path = _safe_relative_file(
            output_root,
            row["relative_training_example_path"],
        )
        release = _json(release_path)
        example = _json(example_path)
        if build_json_artifact_ref(
            release,
            artifact_kind="field_release",
            payload_schema_version=_FIELD_RELEASE_SCHEMA_VERSION,
        ) != row["field_release_ref"]:
            raise ValueError("private field release reference drifted on reopen")
        if build_json_artifact_ref(
            example,
            artifact_kind="training_example",
            payload_schema_version=TRAINING_EXAMPLE_SCHEMA_VERSION,
        ) != row["training_example_ref"]:
            raise ValueError("private training example reference drifted on reopen")
        event_id = str(row["event_id"])
        identity = validate_event_identity(
            _json(
                cohort_root
                / "events"
                / event_id
                / "dual_montage"
                / "sidecars"
                / "event_identity.json"
            )
        )
        montage = validate_montage_derivation_receipt(
            _json(
                cohort_root
                / "events"
                / event_id
                / "dual_montage"
                / "sidecars"
                / "montage_receipt.json"
            ),
            trusted_event_identity=identity,
        )
        trusted_values = {
            field["value_ref"]["artifact_id"]: field["value_payload"]
            for field in release["fields"]
            if field["value_ref"] is not None
        }
        from src.evisoz.data.dataset_policy import validate_field_release

        release = validate_field_release(
            release,
            trusted_event_identity=identity,
            trusted_values_by_artifact_id=trusted_values,
        )
        authorized_ports_by_field = {
            str(item["field_id"]): set(item["loss_ports"])
            for item in (
                private_training_authorization["field_scope"]["field_permissions"]
                if private_training_authorization is not None
                else []
            )
        }
        for capability_row in release["dataset_capability"]["field_roster"]:
            expected_ports = authorized_ports_by_field.get(
                str(capability_row["field_id"]), set()
            )
            expected_loss = {
                "typed_slot_loss": "typed_slot_loss" in expected_ports,
                "node_localization_loss": "node_localization_loss" in expected_ports,
                "report_text_loss": False,
            }
            if capability_row["loss_allowed"] != expected_loss:
                raise ValueError(
                    "private field capability loss permission is not bound to authorization"
                )
        example = validate_training_example(
            example,
            split_roster=roster,
            trusted_linkage_groups=trusted_groups,
            event_identity=identity,
            montage_receipt=montage,
            field_release=release,
        )
        expected_provided = sorted(
            field["field_id"]
            for field in release["fields"]
            if field["state"] == "provided"
        )
        expected_loss_fields = sorted(
            field["field_id"]
            for field in release["fields"]
            if any(field["loss_permissions"].values())
        )
        if (
            row["sample_id"] != identity["sample_id"]
            or row["linkage_group_id"] != identity["linkage_group_id"]
            or row["provided_field_ids"] != expected_provided
            or row["loss_enabled_field_ids"] != expected_loss_fields
            or row["enabled_loss_ports"] != example["enabled_loss_ports"]
        ):
            raise ValueError("private Stage-0 example row summary drifted")
        if row["evisoz_role"] != example["split_assignment"]["evisoz_role"]:
            raise ValueError("private Stage-0 example row split role drifted")
        if row["evisoz_role"] != "development_cv" and example["enabled_loss_ports"]:
            raise ValueError("locked private Stage-0 example enabled training loss")
        role_counts[row["evisoz_role"]] += 1
        for port in example["enabled_loss_ports"]:
            loss_counts[port] += 1
        for field_id in expected_provided:
            provided_counts[field_id] += 1
        for field_id in expected_loss_fields:
            field_loss_counts[field_id] += 1
    expected_counts = {
        "event_count": len(rows),
        "role_event_counts": dict(sorted(role_counts.items())),
        "enabled_loss_port_event_counts": dict(sorted(loss_counts.items())),
        "provided_field_event_counts": dict(sorted(provided_counts.items())),
        "loss_enabled_field_event_counts": dict(sorted(field_loss_counts.items())),
        "physician_report_text_provided_count": 0,
        "physician_report_text_training_count": 0,
    }
    if data["counts"] != expected_counts:
        raise ValueError("private Stage-0 example materialization counts drifted")
    expected_authorized_field_ids = sorted(
        authorized_ports_by_field
    ) if private_training_authorization is not None else []
    expected_authorized_loss_ports = sorted(
        {
            port
            for ports in authorized_ports_by_field.values()
            for port in ports
        }
    ) if private_training_authorization is not None else []
    expected_policy = {
        "private_training_authority_present": private_training_authorization is not None,
        "private_training_authority_status": (
            "validated_external"
            if private_training_authorization is not None
            else "not_provided"
        ),
        "authorized_field_ids": expected_authorized_field_ids,
        "authorized_loss_ports": expected_authorized_loss_ports,
        "direct_fields_are_evaluator_only": private_training_authorization is None,
        "development_high_confidence_direct_fields_can_train": private_training_authorization is not None,
        "low_confidence_fields_are_visible_but_loss_disabled": True,
        "locked_test_fields_can_train": False,
        "empty_positive_set_is_not_converted_to_negative": True,
        "missing_diffuse_token_is_not_converted_to_absence": True,
        "morphology_evolution_localizability_quality_unprovided": True,
        "physician_report_text_unprovided_pending_deidentification": True,
        "generated_text_can_supervise_localization": False,
        "report_text_can_supervise_localization": False,
    }
    if data["release_policy"] != expected_policy:
        raise ValueError("private Stage-0 example release policy drifted")
    if data["receipt_sha256"] != canonical_json_sha256(
        _manifest_hash_source(data)
    ):
        raise ValueError("private Stage-0 example materialization hash drifted")
    return data


__all__ = [
    "PRIVATE_STAGE0_EXAMPLES_SCHEMA_VERSION",
    "materialize_private_stage0_examples",
    "validate_private_stage0_examples_materialization",
]
