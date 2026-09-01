"""Load and validate the de-identified clinical EEG writing style profile.

The style profile controls headings, ordering and lexical preferences only.  It
must never carry patient facts or be treated as clinical evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


STYLE_SCHEMA = "clinical_eeg_report_style_zh_v1"


@dataclass(frozen=True)
class ClinicalEEGStyleProfile:
    payload: Mapping[str, Any]
    sha256: str

    @property
    def profile_id(self) -> str:
        return str(self.payload["profile_id"])

    @property
    def section_headings_zh(self) -> Mapping[str, str]:
        value = self.payload["section_headings_zh"]
        assert isinstance(value, Mapping)
        return value

    def prompt_payload(self) -> dict[str, Any]:
        """Return style-only fields that are safe to expose to the LLM."""

        narrator_headings = {
            key: str(self.section_headings_zh[key])
            for key in (
                "background",
                "interictal",
                "eeg_events",
                "impression",
                "interictal_impression",
                "ictal_impression",
            )
        }
        return {
            "profile_id": self.profile_id,
            "document_structure": ["background", "interictal", "eeg_events", "impression"],
            "section_headings_zh": narrator_headings,
            "event_table_columns_zh": list(self.payload["event_table_columns_zh"]),
            "style_rules": dict(self.payload["style_rules"]),
            "electrode_display_policy": dict(self.payload["electrode_display_policy"]),
        }


def _require_keys(value: Mapping[str, Any], required: set[str], *, name: str) -> None:
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{name} is missing required keys: {sorted(missing)}")


def load_style_profile(path: Path) -> ClinicalEEGStyleProfile:
    raw = path.resolve(strict=True).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("clinical EEG style profile must be a JSON object")
    _require_keys(
        payload,
        {
            "schema_version",
            "profile_id",
            "origin",
            "document_structure",
            "section_headings_zh",
            "omitted_sections",
            "event_table_columns_zh",
            "style_rules",
            "electrode_display_policy",
            "rendering",
        },
        name="style profile",
    )
    if payload["schema_version"] != STYLE_SCHEMA:
        raise ValueError("unsupported clinical EEG style profile schema")
    origin = payload["origin"]
    if not isinstance(origin, dict):
        raise TypeError("style origin must be an object")
    if origin.get("patient_facts_retained") is not False:
        raise ValueError("style profile must not retain patient facts")
    if origin.get("document_instructions_trusted") is not False:
        raise ValueError("sample document instructions must remain untrusted")
    expected_structure = [
        "identity_and_recording_table",
        "background",
        "interictal",
        "eeg_events",
        "impression",
        "review_and_signature",
    ]
    if payload["document_structure"] != expected_structure:
        raise ValueError(
            "style document structure must omit unsupported clinical, activation, "
            "and sleep sections"
        )
    headings = payload["section_headings_zh"]
    if not isinstance(headings, dict) or not headings:
        raise TypeError("style headings must be a non-empty object")
    if not all(isinstance(key, str) and isinstance(value, str) and value for key, value in headings.items()):
        raise TypeError("style headings must contain non-empty strings")
    forbidden_headings = {"manual_information", "activation", "sleep"}
    if forbidden_headings.intersection(headings):
        raise ValueError("style headings must omit unsupported report sections")
    omitted_sections = payload["omitted_sections"]
    if omitted_sections != ["clinical_information", "activation", "sleep"]:
        raise ValueError(
            "style profile must omit exactly the unsupported clinical, activation, "
            "and sleep sections"
        )
    if "fixed_placeholders_zh" in payload:
        raise ValueError("legacy unsupported-section placeholders are forbidden")
    columns = payload["event_table_columns_zh"]
    if not isinstance(columns, list) or len(columns) != 4 or not all(
        isinstance(item, str) and item for item in columns
    ):
        raise ValueError("event table must define exactly four headings")
    rendering = payload["rendering"]
    if not isinstance(rendering, dict):
        raise TypeError("rendering style must be an object")
    style_rules = payload["style_rules"]
    if not isinstance(style_rules, dict):
        raise TypeError("style rules must be an object")
    if style_rules.get("forbid_non_eeg_context") is not True:
        raise ValueError("style rules must forbid non-EEG context")
    if style_rules.get("forbid_sleep_and_activation_generation") is not True:
        raise ValueError("style rules must forbid sleep and activation generation")
    if style_rules.get("unsupported_sections_omitted_from_report") is not True:
        raise ValueError("style rules must omit unsupported report sections")
    for key in (
        "llm_controls_layout",
        "identity_fields_are_deterministic",
        "event_counts_are_deterministic",
        "unsupported_sections_are_omitted",
        "signature_fields_are_deterministic",
    ):
        expected = key != "llm_controls_layout"
        if rendering.get(key) is not expected:
            raise ValueError(f"unsafe rendering style flag: {key}")
    return ClinicalEEGStyleProfile(
        payload=payload,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "STYLE_SCHEMA",
    "ClinicalEEGStyleProfile",
    "load_style_profile",
]
