#!/usr/bin/env python3
"""Materialize a privacy-safe Stage-0 blocker remediation packet.

The packet is an evidence request and review workspace.  It never promotes a
user claim to institutional authorization, guesses a report-to-EDF mapping,
releases report text, or treats an opaque TUEV evaluation identity as closed.
It contains hashes/counts and blank receipt fields so an authorised controller
can complete the remaining closures without copying raw EEG or PHI into the
repository.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.artifact_ref import canonical_json_bytes, canonical_json_sha256


_HASH_PLACEHOLDER = "0" * 64


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    body["receipt_sha256"] = canonical_json_sha256(body)
    return body


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_gate(root: Path) -> Path:
    candidates = sorted(
        (p / "gate.json" for p in (root / "outputs").glob("evisoz_stage0_gate_v1_*")),
        key=lambda p: p.parent.name,
    )
    if not candidates:
        raise FileNotFoundError("no Stage-0 gate output found")
    return candidates[-1]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _build_teacher_inventory(root: Path, gate: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = Path("/mnt/hd1/dyf/dataset/CerebraGloss_ICLR2026/DataEngine/checkpoints/model.pkl")
    audit_path = root / "configs/cerebragloss_positive_box_external_audit_v22.json"
    audit = _load(audit_path)
    cg_manifests = sorted(
        p for p in (root / "outputs").glob("evisoz_stage0_cerebragloss_candidates_v*/manifest.json")
        if p.is_file() and not p.is_symlink()
    )
    cg_materialization = _load(cg_manifests[-1]) if cg_manifests else None
    cerebragloss: dict[str, Any] = {
        "teacher_id": "cerebragloss",
        "source_artifact_present": checkpoint.is_file() and not checkpoint.is_symlink(),
        "source_artifact_sha256": _sha256_file(checkpoint) if checkpoint.is_file() else None,
        "source_artifact_label": "CerebraGloss DataEngine model.pkl",
        "candidate_cache_materialized": cg_materialization is not None,
        "candidate_event_count": cg_materialization.get("counts", {}).get("event_count", 0) if cg_materialization else 0,
        "candidate_count": cg_materialization.get("counts", {}).get("candidate_count", 0) if cg_materialization else 0,
        "admission_status": "materialized_candidate_only_uncalibrated" if cg_materialization else "not_materialized",
        "blocking_evidence": {
            "external_audit_status": audit.get("status"),
            "blocking_mismatches": list(audit.get("blocking_mismatches", [])),
            "training_allowed": audit.get("training_allowed"),
        },
        "next_action": "complete remaining development events and retain candidate-only status; then fit fold-local calibration with a separate receipt",
    }
    # Prefer the dedicated read-only discovery receipt.  Its conservative
    # path-hint policy avoids classifying generic ``eeg-language`` reports as
    # an ELM checkpoint.  Discovery remains inventory-only and cannot grant
    # admission or calibration authority.
    discovery_candidates = sorted(
        (root / "outputs" / "evisoz_teacher_artifact_discovery_v1_20260901").glob(
            "elm*.json"
        )
    )
    discovery = _load(discovery_candidates[-1]) if discovery_candidates else None
    elm_candidates = []
    if discovery is None:
        elm_candidates = sorted(
            p for p in root.rglob("*")
            if p.is_file()
            and "elm" in p.name.casefold()
            and p.suffix.casefold() in {".bin", ".ckpt", ".h5", ".pkl", ".pt", ".pth", ".safetensors"}
        )
    elm: dict[str, Any] = {
        "teacher_id": "elm",
        "source_artifact_present": bool(discovery and discovery["status"] == "found_unvalidated") or bool(elm_candidates),
        "source_artifact_sha256": None,
        "source_artifact_label": "ELM checkpoint/manifest",
        "candidate_cache_materialized": False,
        "admission_status": "found_unvalidated" if discovery and discovery["status"] == "found_unvalidated" else "missing",
        "searched_repository_relative_matches": [
            row["relative_path"] for row in (discovery or {}).get("candidates", [])[:20]
        ] or [str(p.relative_to(root)) for p in elm_candidates[:20]],
        "discovery_receipt_sha256": discovery.get("receipt_sha256") if discovery else None,
        "discovery_status": discovery.get("status") if discovery else ("found_unvalidated" if elm_candidates else "missing"),
        "next_action": "provide an ELM checkpoint plus preprocessing/exposure manifest, then run the audited importer",
    }
    deterministic = gate["checks"][-2]
    return _seal(
        {
            "schema_version": "evisoz_stage0_teacher_artifact_inventory_v1",
            "status": "evidence_inventory_only_no_teacher_outputs_promoted",
            "gate_status_at_materialization": gate["status"],
            "teachers": [cerebragloss, elm],
            "fold_local_calibration": {
                "status": "missing",
                "current_receipt_count": deterministic["facts"].get("fold_local_calibration_receipt_count", 0),
                "required_scope": "one receipt per admitted development fold and candidate/calibrator version",
                "locked_test_included": False,
                "calibration_must_be_fit_on_outer_fold_training_patients_only": True,
                "next_action": "fit and sign fold-local calibrators from development data; do not fabricate a receipt from uncalibrated candidates",
            },
        }
    )


def _build_mapping_packet(root: Path, inventory: Mapping[str, Any]) -> dict[str, Any]:
    unresolved = [
        row for row in inventory["reports"]
        if row["association"]["status"] == "unresolved"
    ]
    expected = {row["document_ref"]["content_hash"]["sha256"] for row in unresolved}
    found = 0
    scanned_docx = 0
    # The controlled report source directory is outside this repository.  We
    # hash it read-only and retain no paths or document names in the packet.
    # Do not recursively walk the whole repository: this workspace contains
    # large model/checkpoint trees and generated outputs.  The repository's
    # controlled report locations are narrow; the external report directory
    # is the authoritative source that must be hash-matched.
    search_roots = [
        root / "reports",
        root / "inputs",
        Path("/mnt/hd1/dyf/dataset/EEG_Reports/Reports"),
    ]
    seen_paths: set[Path] = set()
    for search_root in search_roots:
        if not search_root.is_dir() or search_root.is_symlink():
            continue
        for path in search_root.rglob("*.docx"):
            resolved = path.resolve()
            if resolved in seen_paths or path.is_symlink() or not path.is_file():
                continue
            seen_paths.add(resolved)
            scanned_docx += 1
            if _sha256_file(path) in expected:
                found += 1
    return _seal(
        {
            "schema_version": "evisoz_stage0_private_report_mapping_resolution_packet_v1",
            "status": "awaiting_controller_mapping_or_explicit_exclusion",
            "unresolved_reports": [
                {
                    "report_id": row["report_id"],
                    "document_sha256": row["document_ref"]["content_hash"]["sha256"],
                    "proposed_linkage_group_id": None,
                    "authoritative_mapping_status": "pending",
                    "exclusion_status": "pending",
                    "controller_receipt_reference": None,
                }
                for row in unresolved
            ],
            "controlled_source_exact_sha256_search": {
                "scanned_regular_docx_count": scanned_docx,
                "matching_unresolved_source_count": found,
                "raw_report_bytes_copied_to_repository": False,
                "raw_report_bytes_available_in_controlled_source": found == len(unresolved),
                "interpretation": "exact source bytes are available in the controlled source root, but no patient/EDF linkage was inferred; an authoritative crosswalk or explicit exclusion is still required",
            },
            "required_controller_action": [
                "For each report, provide a content-addressed report→patient/EDF crosswalk, or sign an explicit report_missing/unresolved_exclusion receipt.",
                "Do not map by filename similarity, date proximity, clinical text similarity, or model output.",
            ],
        }
    )


def _build_review_matrix(root: Path, deid: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for candidate in deid["candidates"]:
        rows.append(
            {
                "report_id": candidate["report_id"],
                "candidate_id": candidate["candidate_id"],
                "split_role": (
                    candidate["association"]["split_assignment"]["evisoz_role"]
                    if candidate["association"]["split_assignment"] is not None
                    else "unresolved"
                ),
                "candidate_text_sha256": candidate["text_ref"]["content_hash"]["sha256"],
                "automated_phi_scan": candidate["automated_phi_scan"]["automated_scan_status"],
                "manual_review_status": "pending",
                "indirect_identifier_review": "pending",
                "reviewer_name": "",
                "reviewer_role": "",
                "review_timestamp_utc": "",
                "release_receipt_id": "",
                "development_qwen_training_release": "false",
                "locked_language_evaluation_release": "false",
            }
        )
    rows.sort(key=lambda row: row["candidate_id"])
    matrix = _seal(
        {
            "schema_version": "evisoz_stage0_private_report_manual_review_matrix_v1",
            "status": "technical_screen_complete_institutional_review_pending",
            "candidate_count": len(rows),
            "automated_phi_scan_pass_count": sum(row["automated_phi_scan"] == "pass" for row in rows),
            "manual_review_pass_count": 0,
            "development_qwen_training_release_count": 0,
            "locked_language_evaluation_release_count": 0,
            "review_scope": [
                "direct identifiers",
                "rare dates and locations",
                "clinician names and institutions",
                "procedures and combinations of facts",
                "split role and intended release lane",
            ],
            "release_rule": "Only an authorised institutional reviewer can change pending rows and issue a signed release receipt; automated scan is not approval.",
            "rows": rows,
        }
    )
    return matrix, rows


def _build_public_packet(root: Path, exposure: Mapping[str, Any], field_release: Mapping[str, Any], crosswalk: Mapping[str, Any]) -> dict[str, Any]:
    registry_path = next((root / "outputs").glob("clinical_eeg_full_stack_nested_exposure_graph*/exposure_registry.json"), None)
    registry = _load(registry_path) if registry_path else {}
    tuev = next((d for d in registry.get("dataset_registry", []) if d.get("dataset_id") == "TUEV"), {})
    return _seal(
        {
            "schema_version": "evisoz_stage0_public_overlap_audit_request_v1",
            "status": "audit_inputs_missing_training_remains_disabled",
            "known_closed_inputs": {
                "public_v29_tusz_crosswalk_status": crosswalk.get("status"),
                "public_v29_tusz_crosswalk_patient_count": crosswalk.get("counts", {}).get("v29_patient_count"),
                "public_field_release_status": field_release.get("status"),
                "tusz_source_train_patient_count": exposure.get("counts", {}).get("tusz_source_train_patient_count"),
                "deepsoz_exact_overlap_patient_count": exposure.get("counts", {}).get("deepsoz_source_train_overlap_patient_count"),
                "tuev_train_visible_overlap_patient_count": exposure.get("counts", {}).get("tuev_train_visible_overlap_patient_count"),
            },
            "remaining_audit_requests": [
                {
                    "request_id": "tusz_tuev_decoded_near_partial_overlap",
                    "status": "pending",
                    "required_evidence": "decoded waveform/content fingerprints with crop shift/resample tolerance and patient/session exclusion result",
                },
                {
                    "request_id": "tuev_eval_session_patient_identity",
                    "status": "pending",
                    "current_registry_status": tuev.get("patient_identity_status", "train_visible_eval_opaque"),
                    "required_evidence": "dataset-authoritative eval-session→patient crosswalk and signed fold exposure receipt",
                },
                {
                    "request_id": "tuev_label_fold_receipt",
                    "status": "pending",
                    "required_evidence": "authorized label manifest plus fold-scoped exposure/permission receipt",
                },
            ],
            "safety_boundary": "Opaque TUEV eval sessions are excluded from training and cannot be assigned patient identities by inference or filename heuristics.",
        }
    )


def _build_blocker_rows(blocking_check_ids: Sequence[str]) -> list[dict[str, str]]:
    """Return the user-actionable remediation rows for a source gate.

    The source gate is authoritative.  This helper only projects known
    blocker IDs and never invents a closure or authorization receipt.
    """

    rows = [
        {
            "check_id": "clean_freeze_audit",
            "local_action": "perform clean freeze audit",
            "closure_state": "clean Git snapshot and required contract hashes required",
        },
        {
            "check_id": "offline_teacher_and_derived_candidates",
            "local_action": "inventory and calibration plan prepared",
            "closure_state": "external teacher outputs/calibration receipts required",
        },
        {
            "check_id": "private_field_envelopes",
            "local_action": "user claim retained as non-authorizing attestation",
            "closure_state": "controller signature/approval reference required",
        },
        {
            "check_id": "private_report_linkage",
            "local_action": "three-row mapping/exclusion packet prepared",
            "closure_state": "authoritative crosswalk or explicit exclusion required",
        },
        {
            "check_id": "private_report_text_release",
            "local_action": "43-row manual review matrix prepared",
            "closure_state": "institutional review and release receipts required",
        },
        {
            "check_id": "public_auxiliary_patient_exposure_ledger",
            "local_action": "overlap audit request prepared",
            "closure_state": "decoded overlap and TUEV identity receipts required",
        },
    ]
    source_blockers = set(blocking_check_ids)
    return [row for row in rows if row["check_id"] in source_blockers]


def _write_readme(path: Path) -> None:
    path.write_text(
        """# EviSOZ Stage-0 remediation packet

This packet records what can be prepared locally without promoting
unverified authority or copying private EEG/report content. The gate
must remain `NO_GO` until the external receipts listed below are
returned and the normal materializers are replayed.

* `teacher_artifact_inventory.json`: CerebraGloss checkpoint inventory,
  ELM search result, and fold-local calibration requirements.
* `private_report_mapping_resolution_packet.json`: the three exact
  unresolved report IDs and a no-guess mapping/exclusion form.
* `private_report_manual_review_matrix.json/.csv`: all 43 candidate
  rows with blank institutional reviewer/release fields.
* `public_overlap_audit_request.json`: decoded near/partial overlap and
  TUEV eval identity receipts still required.

No file in this packet authorizes training, Qwen use, language
evaluation, or release of report text.
""",
        encoding="utf-8",
    )


def build_packet(root: Path, output: Path, gate_path: Path | None = None) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    gate_path = gate_path or _latest_gate(root)
    gate = _load(gate_path)
    inventory = _load(root / "outputs/evisoz_stage0_private_physician_report_inventory_v1_20260831/inventory.json")
    deid = _load(root / "outputs/evisoz_stage0_private_report_deid_candidates_v1_20260831/manifest.json")
    exposure = _load(root / "outputs/evisoz_public_auxiliary_exposure_projection_v1_20260831/projection.json")
    field_release = _load(root / "outputs/evisoz_public_auxiliary_field_release_v1_20260831/field_release.json")
    crosswalk = _load(root / "outputs/evisoz_public_v29_tusz_crosswalk_v1_20260831/crosswalk.json")
    output.mkdir(parents=True)
    _write_json(output / "teacher_artifact_inventory.json", _build_teacher_inventory(root, gate))
    _write_json(output / "private_report_mapping_resolution_packet.json", _build_mapping_packet(root, inventory))
    matrix, rows = _build_review_matrix(root, deid)
    _write_json(output / "private_report_manual_review_matrix.json", matrix)
    with (output / "private_report_manual_review_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(output / "public_overlap_audit_request.json", _build_public_packet(root, exposure, field_release, crosswalk))
    # Keep the remediation roster synchronized with the source gate.  The
    # clean-freeze audit is an independent blocker: even when all data
    # receipts are present, a dirty worktree is not a reproducible snapshot.
    blocker_rows = _build_blocker_rows(gate["blocking_check_ids"])
    remediation = _seal(
        {
            "schema_version": "evisoz_stage0_remediation_packet_v1",
            "status": "prepared_external_evidence_required",
            "source_gate": {"path": str(gate_path.relative_to(root)), "status": gate["status"], "blocking_check_ids": gate["blocking_check_ids"]},
            "blockers": blocker_rows,
        }
    )
    _write_json(output / "remediation.json", remediation)
    _write_readme(output / "README.md")
    print(json.dumps({"output": str(output), "status": remediation["status"], "source_gate": str(gate_path), "blocking_check_ids": gate["blocking_check_ids"]}, ensure_ascii=False, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evisoz_stage0_remediation_packet_v1_20260901")
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args(argv)
    build_packet(ROOT, args.output.resolve(), args.gate.resolve() if args.gate else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
