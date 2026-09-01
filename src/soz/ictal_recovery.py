"""Development-only LaBraM recovery decisions for native ictal involvement.

The formal-v5 result remains immutable.  This module applies the same numeric
checks to one prespecified long-context candidate, while making it impossible
to mistake an already-open development result for formal promotion.
"""

from __future__ import annotations

from src.soz.concept_metrics import IctalConceptMetrics
from src.soz.ictal_v5 import decide_v5_i_dev


LABRAM_LONG_CONTEXT_HEAD = "labram_temporal_residual_k31"


def decide_labram_long_context_development(
    *,
    independent_metrics: IctalConceptMetrics,
    long_context_metrics: IctalConceptMetrics,
    time_only_metrics: IctalConceptMetrics,
    mask_only_metrics: IctalConceptMetrics,
    prevalence_metrics: IctalConceptMetrics,
) -> dict[str, object]:
    """Apply unchanged v5 thresholds without granting formal promotion."""

    source = decide_v5_i_dev(
        independent_metrics=independent_metrics,
        temporal_metrics=long_context_metrics,
        time_only_metrics=time_only_metrics,
        mask_only_metrics=mask_only_metrics,
        prevalence_metrics=prevalence_metrics,
    )
    qualified = bool(source["passed"])
    return {
        "schema_version": "soz_labram_ictal_long_context_dev_decision_v1",
        "candidate": LABRAM_LONG_CONTEXT_HEAD,
        "development_qualified": qualified,
        "formal_promotion": False,
        "selected_head": LABRAM_LONG_CONTEXT_HEAD if qualified else None,
        "checks": source["checks"],
        "thresholds": source["thresholds"],
        "metrics": {
            "independent": source["metrics"]["independent"],
            "long_context": source["metrics"]["temporal"],
            "time_only": source["metrics"]["time_only"],
            "mask_only": source["metrics"]["mask_only"],
            "prevalence": source["metrics"]["prevalence"],
        },
        "formal_v5_negative_preserved": True,
        "i_gate_open_authorized": False,
        "next_action": (
            "requires_new_independent_protocol_before_any_gate_open"
            if qualified
            else "stop_k31_candidate"
        ),
    }


__all__ = [
    "LABRAM_LONG_CONTEXT_HEAD",
    "decide_labram_long_context_development",
]
