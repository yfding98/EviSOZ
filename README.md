# EviSOZ

This is the auditable EviSOZ-LM clean worktree. It contains source code,
schemas, configuration, knowledge rules, tests and review tooling. It
intentionally does **not** contain raw EDF/DOCX, patient-name mappings,
checkpoints, or generated model/output bundles.

The Git project root is `/mnt/hd1/dyf/workspace/laptop/EviSOZ`. Controlled
inputs and historical receipts remain outside this repository in the original
workspace. A receipt copied from that workspace is evidence for replay only;
it is never a new training authorization.

## Controlled data locations

The defaults below make the boundary explicit. Override them with CLI options
or environment variables when the controlled artifact store changes:

```bash
EVISOZ_ROOT=/mnt/hd1/dyf/workspace/laptop/EviSOZ
CONTROLLED_WORKSPACE=/mnt/hd1/dyf/workspace/laptop/EEG_Seizure
REPORT_ROOT=/mnt/hd1/dyf/dataset/EEG_Reports/Reports
```

Keep raw reports, EDF files, de-identified signal bundles and model outputs
outside this Git project, and grant access only to the authorized controller.

## Start the review service

Run from the clean worktree. The service reads report bytes only in memory and
returns conservative de-identified intervals; it never writes report text or
source paths into the repository:

```bash
cd /mnt/hd1/dyf/workspace/laptop/EviSOZ
CONTROLLED_WORKSPACE=/mnt/hd1/dyf/workspace/laptop/EEG_Seizure
PYTHONPATH=. python3 scripts/serve_evisoz_private_report_review_v1.py \
  --bundle-root "$CONTROLLED_WORKSPACE/outputs/private_public_mapping_split_deid_v1_20260901_r4" \
  --source-manifest "$CONTROLLED_WORKSPACE/outputs/soz_pre/private_edf_soz_manifest.csv" \
  --exclusion "$CONTROLLED_WORKSPACE/outputs/private_public_mapping_split_deid_v1_20260901_r4/private_reports/exclusion_manifest.json"
```

It binds loopback `127.0.0.1:8791` by default. Three explicitly excluded
unresolved reports are not served. A passed review is still only a local draft;
an institutionally issued release receipt is required before any text enters
training or evaluation.

## Stage-0 and continuation

The latest known Stage-0 replay is `NO_GO`; see
[`SESSION_HANDOFF.md`](SESSION_HANDOFF.md) for the blocker list and the
external receipt paths. Until a newly replayed gate is `GO`, only read-only
audits, schema materialization, loader replay and shadow smoke are permitted.

To continue in a fresh Codex session:

```bash
cd /mnt/hd1/dyf/workspace/laptop/EviSOZ
codex
```

The first message should ask Codex to read `AGENTS.md`, `README.md` and
`SESSION_HANDOFF.md`, verify `pwd` and `git status`, then inspect the latest
Stage-0 gate before doing any training work.
