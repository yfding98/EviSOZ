"""Auditable language-quality evaluation for clinical EEG report drafts.

The module deliberately separates three questions which must not be collapsed
into one score:

* paired report-generation quality, computed only against a deidentified,
  complete physician report for the *same* recording;
* reference-free grounding and writing checks against a frozen EEG fact
  ledger; and
* post-freeze agreement with the closed-vocabulary doctor-onset projection.

The last item is an outcome-consistency analysis, not a language reference.
Neither doctor labels nor EDF annotations are accepted by the candidate input
contract or exposed to a report generator through this module.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


CANDIDATE_SCHEMA = "clinical_eeg_language_quality_input_v1"
REFERENCE_SCHEMA = "clinical_eeg_complete_reference_corpus_v1"
OUTPUT_SCHEMA = "clinical_eeg_language_quality_evaluation_v1"
TOKENIZER_ID = "mixed_zh_clinical_character_tokenizer_v1"
BLEU_POLICY_ID = "corpus_bleu_modified_precision_epsilon_v1"
ROUGE_POLICY_ID = "rouge_l_lcs_best_reference_v1"
REFERENCE_FREE_POLICY_ID = "eeg_fact_grounded_language_audit_v1"
DOCTOR_ONSET_POLICY_ID = "postfreeze_structured_onset_consistency_summary_v1"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TOKEN_RE = re.compile(
    r"[A-Za-z]+[A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"
    r"|[A-Za-z]+[A-Za-z0-9]*"
    r"|[0-9]+(?:\.[0-9]+)?"
    r"|[αβγδθμΩ]+"
    r"|[\u3400-\u4dbf\u4e00-\u9fff]"
)
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
_NORMALIZED_SENTENCE_RE = re.compile(r"[^0-9a-zαβγδθμΩ\u3400-\u9fff]+")

# Terms below require an explicit, fact-bound authorization.  Detection is a
# surface audit only; it does not independently diagnose the EEG pattern.
TERM_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "spike": re.compile(r"棘波|spikes?", re.IGNORECASE),
    "sharp_wave": re.compile(r"尖波|sharp\s*waves?", re.IGNORECASE),
    "epileptiform_discharge": re.compile(r"癫痫样放电|epileptiform", re.IGNORECASE),
    "electrographic_seizure": re.compile(r"电图发作|electrographic\s+seizure", re.IGNORECASE),
    "low_voltage_fast_activity": re.compile(r"低电压快活动|\bLVFA\b", re.IGNORECASE),
    "electrodecrement": re.compile(r"电压递减|electrodecrement", re.IGNORECASE),
    "ictal_onset": re.compile(r"发作(?:期脑电)?起始|脑电起始|ictal\s+onset", re.IGNORECASE),
    "ictal_evolution": re.compile(r"发作(?:期脑电)?演变|ictal\s+evolution", re.IGNORECASE),
    "ictal_spread": re.compile(r"发作(?:期)?扩(?:散|展)|传播|ictal\s+spread", re.IGNORECASE),
    "postictal": re.compile(r"发作后|postictal", re.IGNORECASE),
    "diffuse_or_generalized": re.compile(r"弥漫(?:性)?|广泛性|generalized|diffuse", re.IGNORECASE),
    "soz": re.compile(r"(?:^|[^A-Za-z0-9])SOZ(?:$|[^A-Za-z0-9])|致痫区|发作起始区", re.IGNORECASE),
}

# A match means that the prohibited topic appears on the generated report
# surface, including in a disclaimer.  This is intentionally stricter than
# checking only affirmative assertions because unsupported boilerplate is also
# undesirable in the EEG-only report requested for this pipeline.
FORBIDDEN_SURFACE_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "sleep_eeg": re.compile(r"睡眠(?:脑电|分期|期)?|睡期|\b(?:N[123]|REM)\b", re.IGNORECASE),
    "activation_experiment": re.compile(r"诱发试验|过度换气|闪光刺激|睁闭眼"),
    "ecg_emg_eog": re.compile(r"心电|心率|肌电|眼电|\b(?:ECG|EKG|EMG|EOG)\b", re.IGNORECASE),
    "patient_or_clinical_context": re.compile(r"病史|症状|意识|行为表现|临床资料|既往诊断"),
    "medication_or_treatment": re.compile(r"用药|服药|药物|治疗建议|手术方案"),
    "imaging": re.compile(r"影像资料|头颅影像|\b(?:MRI|CT)\b", re.IGNORECASE),
    "direct_demographics": re.compile(r"(?:姓名|性别|年龄|住院号|门诊号)\s*[:：]"),
    "annotation_or_excel": re.compile(r"EDF\s*annotation|Excel|医生标注|起始字段", re.IGNORECASE),
}

BOILERPLATE_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "missing_structured_fact": re.compile(r"未提供[^。；]*结构化事实"),
    "awaiting_manual_completion": re.compile(r"待[^。；]*医师[^。；]*(?:补充|审核|确认)"),
    "generic_pending_review": re.compile(r"(?:尚待|有待)[^。；]*(?:审核|复核|确认)"),
    "scope_disclaimer": re.compile(r"不在本报告生成范围|未使用\s*(?:EDF|Excel|病史|临床资料)"),
    "quantitative_only_disclaimer": re.compile(r"仅表示量化头皮信号变化"),
}

_CLAUSE_BOUNDARY_CHARACTERS = "。；;\n！？!?"
_ADVERSATIVE_RE = re.compile(r"(?:但是|但|然而|不过|却)")
_NEGATIVE_TERM_PREFIX_RE = re.compile(
    r"(?:"
    r"未(?:见|发现|检出|形成|确认|支持|提示|达到|通过|能|可|明确|提供|获得)"
    r"|尚无|没有|缺乏"
    r"|无(?:明确|足够|充分|可用)?"
    r"|不(?:能|可|予|足以|支持|提示|代表|等同于|直接等同于)"
    r"|无法|不可|难以"
    r")[^。；;\n！？!?]{0,32}$"
)
_NEGATIVE_TERM_SUFFIX_RE = re.compile(
    r"^[^。；;\n！？!?]{0,20}(?:"
    r"证据(?:不足|不充分)"
    r"|定位(?:证据)?(?:不足|不充分|不明确)"
    r"|无法(?:判断|确定|评估|定位|确认)"
    r"|不能(?:判断|确定|评估|定位|确认)"
    r"|不可(?:判断|确定|评估|定位|确认)"
    r"|不(?:明确|支持|成立|进入)"
    r"|未(?:明确|确认|成立)"
    r")"
)
_NEGATED_EXTERNAL_SOURCE_PREFIX_RE = re.compile(
    r"(?:未|没有|不曾|不)(?:实际)?"
    r"(?:使用|读取|加载|访问|调用|纳入|输入|发送|接触)"
    r"[^。；;\n！？!?]{0,64}$",
    re.IGNORECASE,
)
_NEGATED_EXTERNAL_SOURCE_SUFFIX_RE = re.compile(
    r"^[^。；;\n！？!?]{0,8}(?:未|没有|不曾|不)"
    r"(?:实际)?(?:使用|读取|加载|访问|调用|纳入|输入|发送|接触)",
    re.IGNORECASE,
)


def _strict_object(
    value: object,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    context: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    keys = set(value)
    required_set = set(required)
    allowed = required_set.union(optional)
    missing = required_set.difference(keys)
    extra = keys.difference(allowed)
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return deepcopy(value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise TypeError(f"{context} must be a safe identifier")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be non-empty text")
    result = value.strip()
    if len(result) > 500_000:
        raise ValueError(f"{context} is too long")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in result):
        raise ValueError(f"{context} contains control characters")
    return result


def _bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{context} must be boolean")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{context} must be an integer >= {minimum}")
    return value


def _unique_identifiers(value: object, context: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    result = [_identifier(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicates")
    return result


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def tokenize_clinical_zh(text: str) -> list[str]:
    """Tokenize Chinese EEG prose without an external mutable dictionary.

    Han text is evaluated at character level.  Decimal values, Latin clinical
    terms and bipolar derivations such as ``F7-T7`` remain single tokens.
    Punctuation and whitespace are excluded.  NFKC and Latin lower-casing make
    the operation deterministic across presentation variants.
    """

    normalized = unicodedata.normalize("NFKC", _text(text, "report text"))
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(normalized)]


def _validate_claim(value: object, context: str) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("claim_id", "text_zh", "fact_ids", "term_codes"),
        context=context,
    )
    claim_id = _identifier(data["claim_id"], f"{context}.claim_id")
    text = _text(data["text_zh"], f"{context}.text_zh")
    fact_ids = _unique_identifiers(data["fact_ids"], f"{context}.fact_ids")
    term_codes = _unique_identifiers(data["term_codes"], f"{context}.term_codes")
    unsupported = set(term_codes).difference(TERM_PATTERNS)
    if unsupported:
        raise ValueError(f"{context}.term_codes contains unsupported codes: {sorted(unsupported)}")
    return {
        "claim_id": claim_id,
        "text_zh": text,
        "fact_ids": fact_ids,
        "term_codes": term_codes,
    }


def _validate_evidence(value: object, context: str) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=(
            "source_scope",
            "reportable_fact_ids",
            "authorized_terms",
            "doctor_labels_used",
            "edf_annotations_used",
            "excel_fields_used",
        ),
        context=context,
    )
    if data["source_scope"] != "frozen_eeg_fact_ledger_only":
        raise ValueError(f"{context}.source_scope is not EEG-fact-ledger-only")
    for field in ("doctor_labels_used", "edf_annotations_used", "excel_fields_used"):
        if _bool(data[field], f"{context}.{field}") is not False:
            raise ValueError(f"{context}.{field} must be false")
    reportable = _unique_identifiers(
        data["reportable_fact_ids"], f"{context}.reportable_fact_ids"
    )
    raw_terms = data["authorized_terms"]
    if not isinstance(raw_terms, list):
        raise TypeError(f"{context}.authorized_terms must be an array")
    authorized: list[dict[str, Any]] = []
    seen: set[str] = set()
    reportable_set = set(reportable)
    for index, raw in enumerate(raw_terms):
        item_context = f"{context}.authorized_terms[{index}]"
        item = _strict_object(
            raw,
            required=("term_code", "fact_ids"),
            context=item_context,
        )
        code = _identifier(item["term_code"], f"{item_context}.term_code")
        if code not in TERM_PATTERNS:
            raise ValueError(f"{item_context}.term_code is unsupported")
        if code in seen:
            raise ValueError(f"{context}.authorized_terms contains duplicate term codes")
        seen.add(code)
        fact_ids = _unique_identifiers(item["fact_ids"], f"{item_context}.fact_ids")
        if not fact_ids or not set(fact_ids).issubset(reportable_set):
            raise ValueError(
                f"{item_context}.fact_ids must be a non-empty subset of reportable facts"
            )
        authorized.append({"term_code": code, "fact_ids": fact_ids})
    return {
        "source_scope": "frozen_eeg_fact_ledger_only",
        "reportable_fact_ids": reportable,
        "authorized_terms": authorized,
        "doctor_labels_used": False,
        "edf_annotations_used": False,
        "excel_fields_used": False,
    }


def validate_candidate_manifest(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("schema_version", "cohort_id", "records"),
        context="language-quality candidate manifest",
    )
    if data["schema_version"] != CANDIDATE_SCHEMA:
        raise ValueError("candidate manifest schema_version is unsupported")
    cohort_id = _identifier(data["cohort_id"], "candidate cohort_id")
    if not isinstance(data["records"], list) or not data["records"]:
        raise TypeError("candidate manifest records must be a non-empty array")
    records: list[dict[str, Any]] = []
    recording_ids: set[str] = set()
    for index, raw in enumerate(data["records"]):
        context = f"candidate records[{index}]"
        item = _strict_object(
            raw,
            required=(
                "recording_id",
                "report_kind",
                "report_text_zh",
                "sections",
                "required_sections",
                "event_count",
                "claims",
                "evidence",
            ),
            context=context,
        )
        recording_id = _identifier(item["recording_id"], f"{context}.recording_id")
        if recording_id in recording_ids:
            raise ValueError("candidate recording IDs must be unique")
        recording_ids.add(recording_id)
        report_kind = item["report_kind"]
        if report_kind not in {"eeg_report", "technical_unassessable_report"}:
            raise ValueError(f"{context}.report_kind is unsupported")
        report_text = _text(item["report_text_zh"], f"{context}.report_text_zh")
        if type(item["sections"]) is not dict:
            raise TypeError(f"{context}.sections must be an object")
        sections: dict[str, str] = {}
        for raw_key, raw_text in item["sections"].items():
            key = _identifier(raw_key, f"{context}.sections key")
            sections[key] = _text(raw_text, f"{context}.sections.{key}")
        required_sections = _unique_identifiers(
            item["required_sections"], f"{context}.required_sections"
        )
        event_count = _integer(item["event_count"], f"{context}.event_count")
        if not isinstance(item["claims"], list):
            raise TypeError(f"{context}.claims must be an array")
        claims = [
            _validate_claim(claim, f"{context}.claims[{claim_index}]")
            for claim_index, claim in enumerate(item["claims"])
        ]
        if report_kind == "eeg_report" and not claims:
            raise ValueError(f"{context}.claims must enumerate EEG-report assertions")
        for claim in claims:
            if claim["text_zh"] not in report_text:
                raise ValueError(
                    f"{context} claim {claim['claim_id']!r} is not present on the report surface"
                )
        claim_ids = [claim["claim_id"] for claim in claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(f"{context}.claims has duplicate claim IDs")
        evidence = _validate_evidence(item["evidence"], f"{context}.evidence")
        records.append(
            {
                "recording_id": recording_id,
                "report_kind": report_kind,
                "report_text_zh": report_text,
                "sections": sections,
                "required_sections": required_sections,
                "event_count": event_count,
                "claims": claims,
                "evidence": evidence,
            }
        )
    return {
        "schema_version": CANDIDATE_SCHEMA,
        "cohort_id": cohort_id,
        "records": records,
    }


def validate_reference_manifest(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=("schema_version", "cohort_id", "records"),
        context="complete-reference manifest",
    )
    if data["schema_version"] != REFERENCE_SCHEMA:
        raise ValueError("reference manifest schema_version is unsupported")
    cohort_id = _identifier(data["cohort_id"], "reference cohort_id")
    if not isinstance(data["records"], list) or not data["records"]:
        raise TypeError("reference records must be a non-empty array")
    records: list[dict[str, Any]] = []
    recording_ids: set[str] = set()
    reference_ids: set[str] = set()
    text_owner: dict[str, str] = {}
    for index, raw in enumerate(data["records"]):
        context = f"reference records[{index}]"
        item = _strict_object(
            raw,
            required=("recording_id", "reports"),
            context=context,
        )
        recording_id = _identifier(item["recording_id"], f"{context}.recording_id")
        if recording_id in recording_ids:
            raise ValueError("reference recording IDs must be unique")
        recording_ids.add(recording_id)
        if not isinstance(item["reports"], list) or not item["reports"]:
            raise TypeError(f"{context}.reports must be a non-empty array")
        reports: list[dict[str, Any]] = []
        local_hashes: set[str] = set()
        for report_index, raw_report in enumerate(item["reports"]):
            report_context = f"{context}.reports[{report_index}]"
            report = _strict_object(
                raw_report,
                required=(
                    "reference_id",
                    "text_zh",
                    "reference_type",
                    "same_recording",
                    "deidentified",
                ),
                context=report_context,
            )
            reference_id = _identifier(
                report["reference_id"], f"{report_context}.reference_id"
            )
            if reference_id in reference_ids:
                raise ValueError("reference IDs must be globally unique")
            reference_ids.add(reference_id)
            if report["reference_type"] != "complete_physician_eeg_report":
                raise ValueError(
                    f"{report_context} is not a complete physician EEG report"
                )
            if _bool(report["same_recording"], f"{report_context}.same_recording") is not True:
                raise ValueError(f"{report_context} is not paired to the same recording")
            if _bool(report["deidentified"], f"{report_context}.deidentified") is not True:
                raise ValueError(f"{report_context} must be deidentified")
            text = _text(report["text_zh"], f"{report_context}.text_zh")
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in local_hashes:
                raise ValueError(f"{context} contains duplicate reference texts")
            local_hashes.add(text_hash)
            previous_owner = text_owner.get(text_hash)
            if previous_owner is not None and previous_owner != recording_id:
                raise ValueError(
                    "one reference report text must not be reused as gold for different recordings"
                )
            text_owner[text_hash] = recording_id
            reports.append(
                {
                    "reference_id": reference_id,
                    "text_zh": text,
                    "reference_type": "complete_physician_eeg_report",
                    "same_recording": True,
                    "deidentified": True,
                }
            )
        records.append({"recording_id": recording_id, "reports": reports})
    return {
        "schema_version": REFERENCE_SCHEMA,
        "cohort_id": cohort_id,
        "records": records,
    }


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _closest_reference_length(candidate_length: int, references: Sequence[Sequence[str]]) -> int:
    return min((len(item) for item in references), key=lambda length: (abs(length - candidate_length), length))


def corpus_bleu(
    pairs: Sequence[tuple[Sequence[str], Sequence[Sequence[str]]]],
    *,
    epsilon: float = 0.1,
) -> dict[str, Any]:
    """Compute corpus BLEU-1..4 with inspectable aggregate counts."""

    if not pairs:
        return {
            "status": "not_computed_no_paired_complete_references",
            "policy_id": BLEU_POLICY_ID,
            "tokenizer_id": TOKENIZER_ID,
            "smoothing": {"method": "epsilon_for_zero_matched_ngrams", "epsilon": epsilon},
            "bleu_1": None,
            "bleu_2": None,
            "bleu_3": None,
            "bleu_4": None,
        }
    if epsilon <= 0:
        raise ValueError("BLEU epsilon must be positive")
    matches = [0, 0, 0, 0]
    totals = [0, 0, 0, 0]
    candidate_length = 0
    reference_length = 0
    for candidate, references in pairs:
        if not references:
            raise ValueError("BLEU pairs require at least one reference")
        candidate_length += len(candidate)
        reference_length += _closest_reference_length(len(candidate), references)
        for order in range(1, 5):
            candidate_counts = _ngrams(candidate, order)
            maximum_reference_counts: Counter[tuple[str, ...]] = Counter()
            for reference in references:
                reference_counts = _ngrams(reference, order)
                for ngram, count in reference_counts.items():
                    maximum_reference_counts[ngram] = max(
                        maximum_reference_counts[ngram], count
                    )
            matches[order - 1] += sum(
                min(count, maximum_reference_counts[ngram])
                for ngram, count in candidate_counts.items()
            )
            totals[order - 1] += sum(candidate_counts.values())
    if candidate_length == 0:
        brevity_penalty = None
    elif candidate_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - reference_length / candidate_length)
    precisions: list[dict[str, Any]] = []
    smoothed_values: list[float | None] = []
    for order, (matched, total) in enumerate(zip(matches, totals), start=1):
        unsmoothed = matched / total if total else None
        smoothed = ((matched if matched else epsilon) / total) if total else None
        precisions.append(
            {
                "order": order,
                "matched_ngrams": matched,
                "candidate_ngrams": total,
                "unsmoothed_precision": unsmoothed,
                "smoothed_precision": smoothed,
                "smoothing_applied": bool(total and matched == 0),
            }
        )
        smoothed_values.append(smoothed)
    scores: dict[str, float | None] = {}
    for maximum_order in range(1, 5):
        selected = smoothed_values[:maximum_order]
        if brevity_penalty is None or any(value is None for value in selected):
            score = None
        else:
            score = brevity_penalty * math.exp(
                sum(math.log(float(value)) for value in selected) / maximum_order
            )
        scores[f"bleu_{maximum_order}"] = score
    return {
        "status": "computed",
        "policy_id": BLEU_POLICY_ID,
        "tokenizer_id": TOKENIZER_ID,
        "score_scale": "0_to_1",
        "pair_count": len(pairs),
        "candidate_token_count": candidate_length,
        "effective_reference_token_count": reference_length,
        "brevity_penalty": brevity_penalty,
        "smoothing": {"method": "epsilon_for_zero_matched_ngrams", "epsilon": epsilon},
        "modified_precisions": precisions,
        **scores,
    }


def _lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    if len(first) > len(second):
        first, second = second, first
    previous = [0] * (len(first) + 1)
    for right in second:
        current = [0]
        for index, left in enumerate(first, start=1):
            if left == right:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l(pairs: Sequence[tuple[Sequence[str], Sequence[Sequence[str]]]]) -> dict[str, Any]:
    if not pairs:
        return {
            "status": "not_computed_no_paired_complete_references",
            "policy_id": ROUGE_POLICY_ID,
            "tokenizer_id": TOKENIZER_ID,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    per_pair: list[dict[str, float | int]] = []
    micro_lcs = 0
    micro_candidate = 0
    micro_reference = 0
    for candidate, references in pairs:
        options: list[tuple[float, float, float, int, int]] = []
        for reference in references:
            lcs = _lcs_length(candidate, reference)
            precision = lcs / len(candidate) if candidate else 0.0
            recall = lcs / len(reference) if reference else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            options.append((f1, precision, recall, lcs, len(reference)))
        best_f1, best_precision, best_recall, best_lcs, best_reference_length = max(
            options, key=lambda item: (item[0], item[3], -item[4])
        )
        per_pair.append(
            {
                "precision": best_precision,
                "recall": best_recall,
                "f1": best_f1,
                "lcs_token_count": best_lcs,
            }
        )
        micro_lcs += best_lcs
        micro_candidate += len(candidate)
        micro_reference += best_reference_length
    micro_precision = micro_lcs / micro_candidate if micro_candidate else 0.0
    micro_recall = micro_lcs / micro_reference if micro_reference else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "status": "computed",
        "policy_id": ROUGE_POLICY_ID,
        "tokenizer_id": TOKENIZER_ID,
        "score_scale": "0_to_1",
        "pair_count": len(pairs),
        "macro": {
            "precision": statistics.fmean(float(item["precision"]) for item in per_pair),
            "recall": statistics.fmean(float(item["recall"]) for item in per_pair),
            "f1": statistics.fmean(float(item["f1"]) for item in per_pair),
        },
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "lcs_token_count": micro_lcs,
        },
    }


def _sentence_units(text: str) -> list[str]:
    return [item.strip() for item in _SENTENCE_SPLIT_RE.split(text) if item.strip()]


def _normalized_sentence(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return _NORMALIZED_SENTENCE_RE.sub("", normalized)


def _term_local_segments(
    text: str, start: int, stop: int
) -> tuple[str, str, str]:
    """Return local prefix/surface/suffix without crossing a contrast clause."""

    left = max((text.rfind(char, 0, start) for char in _CLAUSE_BOUNDARY_CHARACTERS), default=-1) + 1
    right_candidates = [
        position
        for char in _CLAUSE_BOUNDARY_CHARACTERS
        if (position := text.find(char, stop)) >= 0
    ]
    right = min(right_candidates) if right_candidates else len(text)
    for adversative in _ADVERSATIVE_RE.finditer(text, left, start):
        left = adversative.end()
    following_adversative = _ADVERSATIVE_RE.search(text, stop, right)
    if following_adversative is not None:
        right = following_adversative.start()
    return text[left:start], text[start:stop], text[stop:right]


def _is_soz_structural_heading(text: str, start: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_stop = text.find("\n", start)
    if line_stop < 0:
        line_stop = len(text)
    line = text[line_start:line_stop]
    heading = re.match(
        r"^\s*(?:[一二三四五六七八九十]+[、.]?|[0-9]+[、.)．]?)?"
        r"\s*SOZ\s*定位结论\s*[:：]",
        line,
        re.IGNORECASE,
    )
    return heading is not None and start < line_start + heading.end()


def _term_occurrence_non_positive_reason(
    text: str,
    *,
    term_code: str,
    start: int,
    stop: int,
) -> str | None:
    """Classify headings and explicit abstentions before qualification.

    The qualification gate is intended to block unsupported *positive*
    clinical assertions.  A controlled term can also occur in a section title
    or in an explicit negative/insufficient-evidence statement.  Those uses
    remain counted for auditability but are not promoted to positive claims.
    """

    if term_code == "soz" and _is_soz_structural_heading(text, start):
        return "structural_heading"
    prefix, _, suffix = _term_local_segments(text, start, stop)
    if _NEGATIVE_TERM_PREFIX_RE.search(prefix):
        return "explicit_negative_or_absent"
    if _NEGATIVE_TERM_SUFFIX_RE.search(suffix):
        return "explicit_insufficient_or_indeterminate"
    return None


def _semantic_term_span(
    term_code: str, match: re.Match[str]
) -> tuple[int, int]:
    """Remove regex boundary characters from the clinical term span."""

    if term_code != "soz":
        return match.start(), match.end()
    core = re.search(r"SOZ|致痫区|发作起始区", match.group(0), re.IGNORECASE)
    if core is None:  # defensive closure over the frozen TERM_PATTERNS entry
        return match.start(), match.end()
    return match.start() + core.start(), match.start() + core.end()


def _forbidden_occurrence_exclusion_reason(
    text: str,
    *,
    category_code: str,
    start: int,
    stop: int,
    report_kind: str,
) -> str | None:
    """Exclude only explicit non-use receipts in technical status reports."""

    if (
        category_code != "annotation_or_excel"
        or report_kind != "technical_unassessable_report"
    ):
        return None
    prefix, _, suffix = _term_local_segments(text, start, stop)
    if _NEGATED_EXTERNAL_SOURCE_PREFIX_RE.search(prefix) or (
        _NEGATED_EXTERNAL_SOURCE_SUFFIX_RE.search(suffix)
    ):
        return "technical_report_explicit_non_use_receipt"
    return None


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
        "status": "computed" if denominator else "not_applicable_zero_denominator",
    }


def _claim_character_coverage(text: str, claims: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    relevant = [not char.isspace() for char in text]
    covered = [False] * len(text)
    for claim in claims:
        claim_text = str(claim["text_zh"])
        start = 0
        while True:
            position = text.find(claim_text, start)
            if position < 0:
                break
            for index in range(position, position + len(claim_text)):
                if relevant[index]:
                    covered[index] = True
            start = position + max(1, len(claim_text))
    denominator = sum(relevant)
    numerator = sum(1 for index, flag in enumerate(covered) if flag and relevant[index])
    return _ratio(numerator, denominator)


def evaluate_reference_free_record(record: Mapping[str, Any]) -> dict[str, Any]:
    text = str(record["report_text_zh"])
    evidence = record["evidence"]
    reportable = set(evidence["reportable_fact_ids"])
    claims = list(record["claims"])
    known_citations: set[str] = set()
    grounded_count = 0
    claims_without_fact_ids = 0
    claims_with_unknown_facts = 0
    claim_receipts: list[dict[str, Any]] = []
    authorization = {
        item["term_code"]: set(item["fact_ids"])
        for item in evidence["authorized_terms"]
    }
    qualified_term_codes: set[str] = set()
    qualified_terms_by_claim: dict[str, set[str]] = {}
    for claim in claims:
        cited = set(claim["fact_ids"])
        known = cited.intersection(reportable)
        unknown = cited.difference(reportable)
        known_citations.update(known)
        grounded = bool(cited) and not unknown
        if grounded:
            grounded_count += 1
        if not cited:
            claims_without_fact_ids += 1
        if unknown:
            claims_with_unknown_facts += 1
        term_bindings: list[dict[str, Any]] = []
        for code in claim["term_codes"]:
            authorized_facts = authorization.get(code, set())
            qualified = bool(known.intersection(authorized_facts))
            if qualified:
                qualified_term_codes.add(code)
            term_bindings.append({"term_code": code, "qualified": qualified})
        qualified_terms_by_claim[str(claim["claim_id"])] = {
            item["term_code"] for item in term_bindings if item["qualified"]
        }
        claim_receipts.append(
            {
                "claim_id": claim["claim_id"],
                "grounded": grounded,
                "cited_fact_count": len(cited),
                "known_fact_count": len(known),
                "unknown_fact_count": len(unknown),
                "term_bindings": term_bindings,
            }
        )
    claim_spans: list[tuple[int, int, str]] = []
    for claim in claims:
        claim_text = str(claim["text_zh"])
        start = 0
        while True:
            position = text.find(claim_text, start)
            if position < 0:
                break
            claim_spans.append(
                (position, position + len(claim_text), str(claim["claim_id"]))
            )
            start = position + max(1, len(claim_text))
    term_occurrence_receipts: list[dict[str, Any]] = []
    unqualified_terms: list[str] = []
    positive_term_occurrences = 0
    qualified_term_occurrences = 0
    for code, pattern in TERM_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        qualified_occurrences = 0
        non_positive_reasons: Counter[str] = Counter()
        for match in matches:
            term_start, term_stop = _semantic_term_span(code, match)
            non_positive_reason = _term_occurrence_non_positive_reason(
                text,
                term_code=code,
                start=term_start,
                stop=term_stop,
            )
            if non_positive_reason is not None:
                non_positive_reasons[non_positive_reason] += 1
                continue
            if any(
                span_start <= term_start
                and term_stop <= span_stop
                and code in qualified_terms_by_claim.get(claim_id, set())
                for span_start, span_stop, claim_id in claim_spans
            ):
                qualified_occurrences += 1
        occurrence_count = len(matches)
        non_positive_occurrence_count = sum(non_positive_reasons.values())
        positive_occurrence_count = occurrence_count - non_positive_occurrence_count
        positive_term_occurrences += positive_occurrence_count
        qualified_term_occurrences += qualified_occurrences
        if qualified_occurrences != positive_occurrence_count:
            unqualified_terms.append(code)
        term_occurrence_receipts.append(
            {
                "term_code": code,
                "occurrence_count": occurrence_count,
                "positive_occurrence_count": positive_occurrence_count,
                "non_positive_occurrence_count": non_positive_occurrence_count,
                "non_positive_reason_counts": dict(sorted(non_positive_reasons.items())),
                "qualified_occurrence_count": qualified_occurrences,
                "unqualified_occurrence_count": positive_occurrence_count
                - qualified_occurrences,
            }
        )
    detected_terms = [item["term_code"] for item in term_occurrence_receipts]
    positive_detected_terms = [
        item["term_code"]
        for item in term_occurrence_receipts
        if item["positive_occurrence_count"] > 0
    ]
    non_positive_only_terms = [
        item["term_code"]
        for item in term_occurrence_receipts
        if item["positive_occurrence_count"] == 0
    ]
    unqualified_terms.sort()
    forbidden_matches: list[dict[str, Any]] = []
    forbidden_non_assertive_exclusions: list[dict[str, Any]] = []
    for code, pattern in FORBIDDEN_SURFACE_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        excluded_reasons: Counter[str] = Counter()
        assertive_count = 0
        for match in matches:
            reason = _forbidden_occurrence_exclusion_reason(
                text,
                category_code=code,
                start=match.start(),
                stop=match.end(),
                report_kind=str(record["report_kind"]),
            )
            if reason is None:
                assertive_count += 1
            else:
                excluded_reasons[reason] += 1
        if assertive_count:
            forbidden_matches.append(
                {
                    "code": code,
                    "occurrence_count": assertive_count,
                    "total_surface_occurrence_count": len(matches),
                }
            )
        if excluded_reasons:
            forbidden_non_assertive_exclusions.append(
                {
                    "code": code,
                    "excluded_occurrence_count": sum(excluded_reasons.values()),
                    "reason_counts": dict(sorted(excluded_reasons.items())),
                }
            )
    sections = record["sections"]
    required_sections = record["required_sections"]
    missing_sections = sorted(
        section for section in required_sections if not str(sections.get(section, "")).strip()
    )
    sentences = _sentence_units(text)
    normalized_sentences = [_normalized_sentence(item) for item in sentences]
    normalized_sentences = [item for item in normalized_sentences if item]
    sentence_counts = Counter(normalized_sentences)
    duplicate_sentence_count = sum(count - 1 for count in sentence_counts.values() if count > 1)
    boilerplate_counts = {
        code: sum(1 for sentence in sentences if pattern.search(sentence))
        for code, pattern in BOILERPLATE_PATTERNS.items()
    }
    boilerplate_sentence_indices = {
        index
        for index, sentence in enumerate(sentences)
        if any(pattern.search(sentence) for pattern in BOILERPLATE_PATTERNS.values())
    }
    tokens = tokenize_clinical_zh(text)
    fourgrams = _ngrams(tokens, 4)
    repeated_fourgram_occurrences = sum(count - 1 for count in fourgrams.values() if count > 1)
    sentence_lengths = [len(tokenize_clinical_zh(sentence)) for sentence in sentences]
    long_sentence_count = sum(length > 80 for length in sentence_lengths)
    fact_coverage = _ratio(len(known_citations), len(reportable))
    grounding_precision = _ratio(grounded_count, len(claims))
    ungrounded_rate = _ratio(len(claims) - grounded_count, len(claims))
    term_qualification = _ratio(
        qualified_term_occurrences, positive_term_occurrences
    )
    structure_completeness = _ratio(
        len(required_sections) - len(missing_sections), len(required_sections)
    )
    duplicate_rate = _ratio(duplicate_sentence_count, len(normalized_sentences))
    boilerplate_rate = _ratio(len(boilerplate_sentence_indices), len(sentences))
    repeated_fourgram_rate = _ratio(repeated_fourgram_occurrences, sum(fourgrams.values()))
    safety_gate = (
        (not claims or grounded_count == len(claims))
        and not unqualified_terms
        and not forbidden_matches
        and not missing_sections
    )
    return {
        "recording_id": record["recording_id"],
        "report_kind": record["report_kind"],
        "report_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_count": len(tokens),
        "fact_grounding": {
            "policy_id": REFERENCE_FREE_POLICY_ID,
            "reportable_fact_count": len(reportable),
            "cited_known_fact_count": len(known_citations),
            "fact_coverage": fact_coverage,
            "claim_count": len(claims),
            "grounded_claim_count": grounded_count,
            "claims_without_fact_ids": claims_without_fact_ids,
            "claims_with_unknown_fact_ids": claims_with_unknown_facts,
            "claim_grounding_precision": grounding_precision,
            "ungrounded_claim_rate": ungrounded_rate,
            "claim_surface_character_coverage": _claim_character_coverage(text, claims),
            "claim_receipts": claim_receipts,
            "interpretation_boundary": (
                "ledger-level grounding audit; not an independent clinical truth adjudication"
            ),
        },
        "term_qualification": {
            "detected_term_codes": detected_terms,
            "positive_detected_term_codes": positive_detected_terms,
            "non_positive_only_term_codes": non_positive_only_terms,
            "qualified_detected_term_codes": sorted(
                item["term_code"]
                for item in term_occurrence_receipts
                if item["positive_occurrence_count"] > 0
                and item["qualified_occurrence_count"]
                == item["positive_occurrence_count"]
            ),
            "unqualified_term_codes": unqualified_terms,
            "occurrence_receipts": term_occurrence_receipts,
            "qualification_rate": term_qualification,
        },
        "forbidden_surface_audit": {
            "matched_category_count": len(forbidden_matches),
            "matches": forbidden_matches,
            "non_assertive_exclusions": forbidden_non_assertive_exclusions,
            "passed": not forbidden_matches,
            "affirmation_not_inferred": True,
        },
        "structure": {
            "required_sections": list(required_sections),
            "present_sections": sorted(sections),
            "missing_sections": missing_sections,
            "completeness": structure_completeness,
            "passed": not missing_sections,
        },
        "redundancy": {
            "sentence_count": len(sentences),
            "duplicate_sentence_count": duplicate_sentence_count,
            "duplicate_sentence_rate": duplicate_rate,
            "boilerplate_sentence_count": len(boilerplate_sentence_indices),
            "boilerplate_sentence_rate": boilerplate_rate,
            "boilerplate_pattern_counts": boilerplate_counts,
            "repeated_fourgram_occurrences": repeated_fourgram_occurrences,
            "repeated_fourgram_rate": repeated_fourgram_rate,
        },
        "readability_descriptive": {
            "method": "mixed-token sentence length; descriptive, not a validated clinical readability score",
            "sentence_count": len(sentence_lengths),
            "mean_sentence_tokens": statistics.fmean(sentence_lengths) if sentence_lengths else None,
            "median_sentence_tokens": statistics.median(sentence_lengths) if sentence_lengths else None,
            "maximum_sentence_tokens": max(sentence_lengths) if sentence_lengths else None,
            "over_80_token_sentence_count": long_sentence_count,
            "over_80_token_sentence_rate": _ratio(long_sentence_count, len(sentence_lengths)),
        },
        "quality_gate": {
            "passed": safety_gate,
            "gate_scope": "grounding_term_scope_and_structure_only",
            "boilerplate_and_readability_are_diagnostic_not_hard_failures": True,
        },
    }


def _aggregate_reference_free(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_facts = sum(item["fact_grounding"]["reportable_fact_count"] for item in records)
    cited_facts = sum(item["fact_grounding"]["cited_known_fact_count"] for item in records)
    total_claims = sum(item["fact_grounding"]["claim_count"] for item in records)
    grounded_claims = sum(item["fact_grounding"]["grounded_claim_count"] for item in records)
    sentences = sum(item["redundancy"]["sentence_count"] for item in records)
    boilerplate = sum(item["redundancy"]["boilerplate_sentence_count"] for item in records)
    duplicate = sum(item["redundancy"]["duplicate_sentence_count"] for item in records)
    return {
        "policy_id": REFERENCE_FREE_POLICY_ID,
        "record_count": len(records),
        "eeg_report_count": sum(item["report_kind"] == "eeg_report" for item in records),
        "technical_report_count": sum(
            item["report_kind"] == "technical_unassessable_report" for item in records
        ),
        "quality_gate_pass_record_count": sum(item["quality_gate"]["passed"] for item in records),
        "forbidden_surface_match_record_count": sum(
            not item["forbidden_surface_audit"]["passed"] for item in records
        ),
        "unqualified_term_record_count": sum(
            bool(item["term_qualification"]["unqualified_term_codes"]) for item in records
        ),
        "structure_incomplete_record_count": sum(
            not item["structure"]["passed"] for item in records
        ),
        "micro_fact_coverage": _ratio(cited_facts, total_facts),
        "micro_claim_grounding_precision": _ratio(grounded_claims, total_claims),
        "micro_ungrounded_claim_rate": _ratio(total_claims - grounded_claims, total_claims),
        "micro_duplicate_sentence_rate": _ratio(duplicate, sentences),
        "micro_boilerplate_sentence_rate": _ratio(boilerplate, sentences),
        "no_composite_language_score_reported": True,
    }


def _optional_meteor(
    texts: Sequence[tuple[Sequence[str], Sequence[Sequence[str]]]], requested: bool
) -> dict[str, Any]:
    if not requested:
        return {"status": "not_requested", "score": None}
    if not texts:
        return {"status": "not_computed_no_paired_complete_references", "score": None}
    try:
        import nltk  # type: ignore
        from nltk.translate.meteor_score import meteor_score  # type: ignore
    except ImportError:
        return {
            "status": "unavailable_dependency_not_installed",
            "score": None,
            "network_download_attempted": False,
        }
    try:
        values = [
            float(meteor_score([list(reference) for reference in references], list(candidate)))
            for candidate, references in texts
        ]
    except LookupError:
        return {
            "status": "unavailable_local_nltk_resource",
            "score": None,
            "network_download_attempted": False,
        }
    except Exception as exc:  # dependency-specific errors must not abort the core audit
        return {
            "status": "unavailable_backend_error",
            "error_type": type(exc).__name__,
            "score": None,
            "network_download_attempted": False,
        }
    return {
        "status": "computed",
        "backend": "nltk.translate.meteor_score",
        "backend_version": getattr(nltk, "__version__", "unknown"),
        "tokenizer_id": TOKENIZER_ID,
        "score": statistics.fmean(values),
        "pair_count": len(values),
        "network_download_attempted": False,
    }


def _tree_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = 0
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise ValueError("local BERTScore model tree must not contain symlinks")
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files += 1
    if files == 0:
        raise ValueError("local BERTScore model directory contains no files")
    return digest.hexdigest(), files


def _optional_bertscore(
    raw_pairs: Sequence[tuple[str, Sequence[str]]],
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config is None:
        return {"status": "not_requested", "f1": None}
    if not raw_pairs:
        return {"status": "not_computed_no_paired_complete_references", "f1": None}
    model_path_raw = config.get("model_path")
    num_layers = config.get("num_layers")
    domain = config.get("model_domain")
    device = config.get("device", "cpu")
    if not isinstance(model_path_raw, (str, Path)):
        raise TypeError("BERTScore model_path must be a local directory")
    unresolved_model_path = Path(model_path_raw)
    if unresolved_model_path.is_symlink():
        raise ValueError("BERTScore model_path must not be a symlink")
    model_path = unresolved_model_path.resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError("BERTScore model_path must be a non-symlink local directory")
    if type(num_layers) is not int or num_layers <= 0:
        raise TypeError("BERTScore num_layers must be a positive integer")
    if domain not in {"general", "medical_eeg"}:
        raise ValueError("BERTScore model_domain must be general or medical_eeg")
    if not isinstance(device, str) or not device:
        raise TypeError("BERTScore device must be a non-empty string")
    model_hash, model_file_count = _tree_sha256(model_path)
    try:
        import bert_score  # type: ignore
    except ImportError:
        return {
            "status": "unavailable_dependency_not_installed",
            "f1": None,
            "model_tree_sha256": model_hash,
            "model_file_count": model_file_count,
            "network_download_attempted": False,
        }
    candidates = [candidate for candidate, _ in raw_pairs]
    references: list[list[str]] = [list(items) for _, items in raw_pairs]
    try:
        precision, recall, f1 = bert_score.score(
            candidates,
            references,
            model_type=str(model_path),
            num_layers=num_layers,
            device=device,
            verbose=False,
            idf=False,
            rescale_with_baseline=False,
        )
        precision_value = float(precision.mean().item())
        recall_value = float(recall.mean().item())
        f1_value = float(f1.mean().item())
    except Exception as exc:
        return {
            "status": "unavailable_backend_error",
            "error_type": type(exc).__name__,
            "precision": None,
            "recall": None,
            "f1": None,
            "model_tree_sha256": model_hash,
            "model_file_count": model_file_count,
            "network_download_attempted": False,
        }
    return {
        "status": "computed",
        "backend": "bert-score",
        "backend_version": getattr(bert_score, "__version__", "unknown"),
        "model_tree_sha256": model_hash,
        "model_file_count": model_file_count,
        "model_domain_operator_declaration": domain,
        "medical_domain_status": (
            "operator_declared_not_independently_verified"
            if domain == "medical_eeg"
            else "not_medical_specific"
        ),
        "num_layers": num_layers,
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_value,
        "pair_count": len(raw_pairs),
        "network_download_attempted": False,
    }


def _paired_reference_metrics(
    candidates: Mapping[str, Mapping[str, Any]],
    reference_manifest: Mapping[str, Any] | None,
    *,
    meteor_requested: bool,
    bertscore_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if reference_manifest is None:
        empty_pairs: list[tuple[Sequence[str], Sequence[Sequence[str]]]] = []
        return {
            "status": "not_computed_no_complete_paired_reference_manifest",
            "candidate_record_count": len(candidates),
            "paired_record_count": 0,
            "pair_coverage": 0.0,
            "bleu": corpus_bleu(empty_pairs),
            "rouge_l": rouge_l(empty_pairs),
            "meteor": _optional_meteor(empty_pairs, meteor_requested),
            "bertscore": _optional_bertscore([], bertscore_config),
            "excel_onset_is_not_a_reference_report": True,
            "single_style_example_is_not_a_cohort_gold_reference": True,
        }
    references_by_id = {
        item["recording_id"]: item for item in reference_manifest["records"]
    }
    paired_ids = sorted(set(candidates).intersection(references_by_id))
    token_pairs: list[tuple[Sequence[str], Sequence[Sequence[str]]]] = []
    raw_pairs: list[tuple[str, Sequence[str]]] = []
    per_record: list[dict[str, Any]] = []
    for recording_id in paired_ids:
        candidate_text = str(candidates[recording_id]["report_text_zh"])
        reference_texts = [
            str(item["text_zh"]) for item in references_by_id[recording_id]["reports"]
        ]
        candidate_tokens = tokenize_clinical_zh(candidate_text)
        reference_tokens = [tokenize_clinical_zh(text) for text in reference_texts]
        token_pairs.append((candidate_tokens, reference_tokens))
        raw_pairs.append((candidate_text, reference_texts))
        pair_rouge = rouge_l([(candidate_tokens, reference_tokens)])
        per_record.append(
            {
                "recording_id": recording_id,
                "candidate_token_count": len(candidate_tokens),
                "reference_count": len(reference_tokens),
                "reference_token_counts": [len(item) for item in reference_tokens],
                "rouge_l_best_reference": pair_rouge["macro"],
            }
        )
    return {
        "status": "computed" if paired_ids else "not_computed_no_matching_recording_ids",
        "candidate_record_count": len(candidates),
        "reference_record_count": len(references_by_id),
        "paired_record_count": len(paired_ids),
        "pair_coverage": len(paired_ids) / len(candidates) if candidates else 0.0,
        "unpaired_candidate_recording_ids": sorted(set(candidates).difference(references_by_id)),
        "unpaired_reference_recording_ids": sorted(set(references_by_id).difference(candidates)),
        "bleu": corpus_bleu(token_pairs),
        "rouge_l": rouge_l(token_pairs),
        "meteor": _optional_meteor(token_pairs, meteor_requested),
        "bertscore": _optional_bertscore(raw_pairs, bertscore_config),
        "per_record": per_record,
        "excel_onset_is_not_a_reference_report": True,
        "single_style_example_is_not_a_cohort_gold_reference": True,
    }


def _summarize_validated_doctor_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    record_dispositions: Counter[str] = Counter()
    overall_statuses: Counter[str] = Counter()
    uncertainty_statuses: Counter[str] = Counter()
    spatial_statuses: Counter[str] = Counter()
    label_count = 0
    eligible_label_count = 0
    for record in bundle["records"]:
        record_dispositions[str(record["record_consistency_disposition"])] += 1
        for label in record["doctor_labels"]:
            label_count += 1
            if label.get("evaluation_eligible") is not True:
                continue
            eligible_label_count += 1
            comparison = label["fact_consistency"]
            overall_statuses[str(comparison["overall_status"])] += 1
            uncertainty_statuses[str(comparison["onset_uncertainty"]["status"])] += 1
            for field in comparison["spatial_fields"]:
                spatial_statuses[str(field["status"])] += 1
    overall_comparable = sum(
        overall_statuses[code] for code in ("match", "partial_match", "mismatch")
    )
    spatial_comparable = sum(
        spatial_statuses[code] for code in ("match", "partial_match", "mismatch")
    )
    return {
        "status": "computed_from_validated_postfreeze_structured_sidecar",
        "policy_id": DOCTOR_ONSET_POLICY_ID,
        "label_release_id": bundle.get("label_release_id"),
        "record_count": len(bundle["records"]),
        "doctor_label_count": label_count,
        "evaluation_eligible_label_count": eligible_label_count,
        "record_consistency_disposition_counts": dict(sorted(record_dispositions.items())),
        "label_overall_status_counts": dict(sorted(overall_statuses.items())),
        "onset_uncertainty_status_counts": dict(sorted(uncertainty_statuses.items())),
        "spatial_field_status_counts": dict(sorted(spatial_statuses.items())),
        "selective_overall_exact_match": _ratio(overall_statuses["match"], overall_comparable),
        "selective_overall_compatible_match": _ratio(
            overall_statuses["match"] + overall_statuses["partial_match"],
            overall_comparable,
        ),
        "selective_spatial_compatible_match": _ratio(
            spatial_statuses["match"] + spatial_statuses["partial_match"],
            spatial_comparable,
        ),
        "spatially_comparable_field_count": spatial_comparable,
        "overall_match_may_reflect_uncertainty_alignment": True,
        "overall_match_is_not_soz_localization_accuracy": True,
        "not_a_language_reference": True,
        "excluded_from_bleu_rouge_meteor_bertscore": True,
        "raw_excel_text_consumed_by_this_evaluator": False,
        "edf_annotations_consumed": False,
    }


def summarize_postfreeze_doctor_onset(value: object) -> dict[str, Any]:
    """Validate and summarize the already-published structured sidecar."""

    from src.clinical_eeg_long_recording.postfreeze_doctor_label_bundle import (
        validate_postfreeze_doctor_label_bundle,
    )

    validated = validate_postfreeze_doctor_label_bundle(value)
    return _summarize_validated_doctor_bundle(validated)


def evaluate_language_quality(
    candidates: object,
    *,
    references: object | None = None,
    doctor_label_bundle: object | None = None,
    meteor_requested: bool = False,
    bertscore_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_manifest = validate_candidate_manifest(candidates)
    reference_manifest = (
        validate_reference_manifest(references) if references is not None else None
    )
    if (
        reference_manifest is not None
        and reference_manifest["cohort_id"] != candidate_manifest["cohort_id"]
    ):
        raise ValueError("candidate and reference cohort IDs do not match")
    record_metrics = [
        evaluate_reference_free_record(record) for record in candidate_manifest["records"]
    ]
    candidates_by_id = {
        record["recording_id"]: record for record in candidate_manifest["records"]
    }
    paired = _paired_reference_metrics(
        candidates_by_id,
        reference_manifest,
        meteor_requested=meteor_requested,
        bertscore_config=bertscore_config,
    )
    if doctor_label_bundle is None:
        doctor_summary = {
            "status": "not_provided",
            "not_a_language_reference": True,
            "excluded_from_bleu_rouge_meteor_bertscore": True,
        }
    else:
        doctor_summary = summarize_postfreeze_doctor_onset(doctor_label_bundle)
    source_receipts = {
        "candidate_manifest_canonical_sha256": canonical_sha256(candidate_manifest),
        "reference_manifest_canonical_sha256": (
            canonical_sha256(reference_manifest) if reference_manifest is not None else None
        ),
        "postfreeze_doctor_label_bundle_canonical_sha256": (
            canonical_sha256(doctor_label_bundle) if doctor_label_bundle is not None else None
        ),
        "edf_annotation_loaded": False,
        "excel_workbook_loaded": False,
        "doctor_label_available_to_generation": False,
    }
    body = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "completed",
        "cohort_id": candidate_manifest["cohort_id"],
        "source_receipts": source_receipts,
        "method_receipt": {
            "tokenizer_id": TOKENIZER_ID,
            "paired_reference_policy": (
                "same-recording_deidentified_complete_physician_report_only"
            ),
            "reference_free_policy_id": REFERENCE_FREE_POLICY_ID,
            "doctor_onset_policy_id": DOCTOR_ONSET_POLICY_ID,
            "single_example_broadcast_as_gold_forbidden": True,
            "no_unvalidated_composite_quality_score": True,
            "optional_metric_failure_blocks_core_audit": False,
        },
        "reference_free_cohort_summary": _aggregate_reference_free(record_metrics),
        "paired_complete_reference_metrics": paired,
        "postfreeze_doctor_onset_consistency": doctor_summary,
        "records": record_metrics,
    }
    return {
        **body,
        "evaluation_id": "CEGLQE-" + canonical_sha256(body)[:24],
    }


__all__ = [
    "BLEU_POLICY_ID",
    "CANDIDATE_SCHEMA",
    "DOCTOR_ONSET_POLICY_ID",
    "OUTPUT_SCHEMA",
    "REFERENCE_FREE_POLICY_ID",
    "REFERENCE_SCHEMA",
    "ROUGE_POLICY_ID",
    "TOKENIZER_ID",
    "canonical_sha256",
    "corpus_bleu",
    "evaluate_language_quality",
    "evaluate_reference_free_record",
    "rouge_l",
    "summarize_postfreeze_doctor_onset",
    "tokenize_clinical_zh",
    "validate_candidate_manifest",
    "validate_reference_manifest",
]
