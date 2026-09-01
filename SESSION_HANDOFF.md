# EviSOZ session handoff

Updated: 2026-09-01 (Asia/Shanghai)

## Current state

- Stage-0 replay: `NO_GO` at
  `/mnt/hd1/dyf/workspace/laptop/EEG_Seizure/outputs/evisoz_stage0_gate_v1_20260901_r68/gate.json`.
- `private_report_linkage`: `GO` after an explicit operational quarantine receipt.
- Independent EviSOZ clean-freeze audit: `GO` at local ignored
  `outputs/clean_freeze_audit_v7.json` (audit ID
  `EVISOZ-FREEZE-bcd37f977aad23c7d812fe13`).
- Remaining independent blockers are unchanged: private governance training
  authority, report manual review/release, public overlap/TUEV identity,
  CerebraGloss/ELM candidates and fold-local calibration, plus the parent
  worktree clean-freeze audit.

## Three excluded reports

Receipt (external to this clean worktree):
`/mnt/hd1/dyf/workspace/laptop/EEG_Seizure/outputs/private_public_mapping_split_deid_v1_20260901_r4/private_reports/exclusion_manifest.json`

The receipt binds these report IDs and exact DOCX hashes, but stores no names,
paths or report text:

- `EVISOZ-PRPT-32efa0b02b8149ca70779b11`
- `EVISOZ-PRPT-3754cbc80cd1f59d67031247`
- `EVISOZ-PRPT-4e79c3dac42502339e5787e5`

All six downstream policy flags are false for each entry: no linkage, no split,
no signal preprocessing, no event training, no Qwen training and no language
evaluation. The operational receipt is not an institutional training
authorization.

The active candidate root used for downstream work is
`/mnt/hd1/dyf/workspace/laptop/EEG_Seizure/outputs/evisoz_private_report_active_candidates_v1_20260901`; it contains
40 candidate files. The original r4 candidate root is retained only as an
immutable audit source and is not the downstream input.

## Review service

The service is running in the original workspace at:
`http://127.0.0.1:8791/`

Health endpoint: `http://127.0.0.1:8791/api/health` reports 40 reviewable and 3
hidden/excluded reports. Review submissions are written only to the ignored
local output `outputs/evisoz_private_report_manual_review_service_v1_20260901/reviews.json`
under whichever controlled run root is supplied. The server re-reads raw DOCX bytes in memory and returns only the conservative
de-identified clinical interval; it never writes raw text or source paths.

## Migration boundary

Copied into this project: `src/evisoz`, its runtime dependencies (`src/soz`,
`src/clinical_eeg_report`, `src/clinical_eeg_long_recording`), the report viewer,
schemas/configuration/knowledge, the Stage-0 scripts, and the dashboard
template. Not copied: raw EDF/DOCX, internal path maps, private reports,
de-identified EDF bundles and model/output artifacts. The parent workspace
remains the source of those controlled inputs. This clean worktree is
`/mnt/hd1/dyf/workspace/laptop/EviSOZ`; its own `.git` history is independent
of the parent repository. Generated outputs are intentionally not migrated.
Recreate target-relative audit receipts after migration; do not copy old
receipts into this repository and do not treat them as authorization.

## Fresh-session checks

```bash
cd /mnt/hd1/dyf/workspace/laptop/EviSOZ
pwd
git status --short --branch
PYTHONPATH=. python3 scripts/materialize_evisoz_schema_registry_v1.py
PYTHONPATH=. python3 scripts/materialize_evisoz_clean_freeze_audit_v1.py \
  --output outputs/clean_freeze_audit_target_v1.json
```

The last command writes an ignored local artifact. The Stage-0 gate must be
replayed with explicit external artifact paths before any training loader,
optimizer, Qwen SFT, alignment or non-zero residual is constructed.
