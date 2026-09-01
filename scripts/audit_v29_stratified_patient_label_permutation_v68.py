#!/usr/bin/env python3
"""Expanded conditional patient-label falsification for frozen v29.

The experiment preserves each permuted patient's documented positive-set
cardinality bucket and laterality stratum inside every outer-training fold.
It therefore tests patient-reference correspondence beyond these coarse label
priors.  Formal v29 is never replaced, selected, or ensembled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from scripts import audit_v29_patient_label_permutation_v61 as v61


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "trustworthy_soz_v29_stratified_patient_label_permutation_v68"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_stratified_patient_label_permutation_v68_20260816"
REPETITIONS = 99


def build_parser():
    parser = v61.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_directory=DEFAULT_OUTPUT,
        repetitions=REPETITIONS,
        permutation_mode="cardinality_laterality_stratified",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = build_parser().parse_args(argv)
    if args.permutation_mode != "cardinality_laterality_stratified":
        raise ValueError("v68 requires cardinality/laterality-stratified permutation")
    result, tensors = v61.run(args)
    result["schema_version"] = SCHEMA
    result["status"] = "completed_public_stratified_patient_label_permutation_falsification"
    result["extension_of"] = "trustworthy_soz_v29_patient_label_permutation_v61"
    result["interpretation_boundary"]["conditional_null_preserves_all_spatial_or_channel_priors"] = False
    result["interpretation_boundary"]["allowed_claim"] = (
        "patient-reference correspondence beyond coarse positive-set cardinality and laterality strata is required for the audited H/D head pipeline to retain formal public OOF performance"
    )
    output = v61.publish(output=args.output_directory, result=result, tensors=tensors)
    print(json.dumps({
        "output": str(output),
        "status": result["status"],
        "repetitions": result["repetitions"],
        "mode": result["permutation_mode"],
        "strict_tail": result["descriptive_empirical_tail_probability_null_ge_formal"]["strict"],
        "private_loaded": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
