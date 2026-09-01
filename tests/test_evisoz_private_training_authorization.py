from __future__ import annotations

import pytest

from src.evisoz.data.artifact_ref import build_json_artifact_ref, build_raw_artifact_ref
from src.evisoz.data.private_training_authorization import (
    build_private_training_authorization,
    validate_private_training_authorization,
)


def _bindings() -> dict[str, object]:
    split = {"schema_version": "synthetic_split", "receipt_sha256": "a" * 64}
    return {
        "dataset_id": "private",
        "patient_roster_sha256": "a" * 64,
        "split_roster_ref": build_json_artifact_ref(
            split,
            artifact_kind="split_roster",
            payload_schema_version="evisoz_split_roster_v1",
        ),
        "signal_roster_ref": build_raw_artifact_ref(
            b"signal",
            artifact_kind="private_signal_roster",
            media_type="text/csv",
        ),
        "target_ledger_ref": build_raw_artifact_ref(
            b"target",
            artifact_kind="private_target_ledger",
            media_type="text/csv",
        ),
        "source_manifest_ref": build_raw_artifact_ref(
            b"source",
            artifact_kind="private_label_authority_manifest",
            media_type="text/csv",
        ),
    }


def _authorization(**overrides: object) -> dict[str, object]:
    body = build_private_training_authorization(
        issuer={
            "institution": "SUAT-SYNTH",
            "role": "data_controller",
            "approval_reference": "SYNTH-APPROVAL",
        },
        signature={
            "scheme": "detached_signature",
            "signature_reference": "SYNTH-SIGNATURE",
            "signed_payload_sha256": "b" * 64,
            "verification_receipt_ref": build_raw_artifact_ref(
                b"verified",
                artifact_kind="governance_signature_verification",
                media_type="application/octet-stream",
            ),
            "verification_status": "verified",
        },
        effective_window={
            "effective_from": "2026-01-01T00:00:00Z",
            "effective_until": "2027-01-01T00:00:00Z",
        },
        data_binding=_bindings(),
        field_permissions=[
            {
                "field_id": "PRIVATE-ONSET-NODES",
                "loss_ports": ["node_localization_loss", "typed_slot_loss"],
            }
        ],
    )
    for path, value in overrides.items():
        body[path] = value
    return body


def test_authorization_replays_exact_binding_and_window() -> None:
    auth = _authorization()
    result = validate_private_training_authorization(
        auth,
        expected_bindings=_bindings(),
        expected_field_ids={"PRIVATE-ONSET-NODES"},
        as_of_utc="2026-09-01T00:00:00Z",
    )
    assert result["permissions"]["locked_test_training"] is False


def test_mismatched_source_binding_fails_closed() -> None:
    with pytest.raises(ValueError, match="binding drifted"):
        validate_private_training_authorization(
            _authorization(),
            expected_bindings={**_bindings(), "patient_roster_sha256": "c" * 64},
        )


def test_report_text_field_cannot_be_authorized() -> None:
    with pytest.raises(ValueError, match="report text"):
        validate_private_training_authorization(
            _authorization(
                field_scope={
                    "allowed_evisoz_roles": ["development_cv"],
                    "field_permissions": [
                        {
                            "field_id": "PRIVATE-PHYSICIAN-REPORT-TEXT",
                            "loss_ports": ["typed_slot_loss"],
                        }
                    ],
                    "locked_test_training_allowed": False,
                    "report_text_loss_allowed": False,
                    "prompt_or_rag_allowed": False,
                    "report_text_can_supervise_localization": False,
                }
            )
        )


def test_unverified_signature_fails_closed() -> None:
    auth = _authorization()
    auth["signature"] = {
        **auth["signature"],
        "verification_status": "pending",
    }
    with pytest.raises(ValueError, match="verification"):
        validate_private_training_authorization(auth)
