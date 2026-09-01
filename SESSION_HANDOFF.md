# EviSOZ session handoff

Updated: 2026-09-01 16:27 CST (Asia/Shanghai)

## Current state

- Stage-0 replay: `NO_GO` at the current clean-worktree gate
  `outputs/evisoz_stage0_gate_v1_20260901_r72/gate.json` (blocking checks:
  offline teacher/calibration, private governance authority, report text
  release, and public auxiliary exposure ledger).
- `private_report_linkage`: `GO` after an explicit operational quarantine receipt.
- Independent EviSOZ clean-freeze audit: `GO` at the target worktree's ignored
  `outputs/clean_freeze_audit_target_v1_20260901_r3.json` (receipt SHA-256
  `ec4e26785a55383a57bc2e80fdc14b454c38cb0082094a6caab58dc2a92653b0`). The
  receipt is regenerated after each committed change; inspect its audit ID
  and SHA-256 in the local file rather than treating this handoff line as an
  authorization.
- Remaining independent blockers are unchanged: private governance training
  authority, report manual review/release, public overlap/TUEV identity,
  CerebraGloss/ELM candidates and fold-local calibration, plus the parent
  worktree clean-freeze audit.

- Remediation packet r4 was materialized in this clean worktree using explicit
  controlled external JSON inputs (no private/raw bytes copied):
  `outputs/evisoz_stage0_remediation_packet_v1_20260901_r4/`. The materializer
  now accepts `--inventory`, `--deid`, `--exposure`, `--field-release`,
  `--crosswalk`, `--cerebragloss-audit`, and `--cerebragloss-manifest` so a
  clean migration can replay external inputs without assuming they exist under
  repository `outputs/`. The packet remains evidence-request-only and does
  not authorize training or report release.

- Latest clean-worktree replay receipts (2026-09-01): aggregate gate
  `outputs/evisoz_stage0_gate_v1_20260901_r73/gate.json` remains `NO_GO`;
  bound-evidence loader replay is
  `outputs/evisoz_stage0_bound_evidence_loader_replay_v1_20260901_r1` (88
  events/31 patients); full structural shadow smoke is
  `outputs/evisoz_stage0_shadow_inference_smoke_v1_20260901_r14/` (88 events,
  31 patient packets, all structural metrics 1.0). These receipts are
  non-authorizing and do not expose physician report text.

- The remediation dashboard builder now defaults to the clean-worktree
  template `docs/evisoz_stage0_remediation_dashboard.html` rather than the
  absent migration-era `code/data_preprocess` path. A default invocation was
  replayed successfully; embedded JavaScript and PHI/path absence checks pass.

- Stage-1 guard-only replay against gate r74 is recorded at
  `outputs/evisoz_stage1_evidence_training_v1_20260901_r8/receipt.json` with
  status `blocked_before_model_or_loader_construction`. The receipt validates
  that model, optimizer, training loader, teacher runtime, Qwen generation and
  residual were all kept closed under `NO_GO`.

- Qwen connector synthetic smoke is recorded at
  `outputs/evisoz_qwen_connector_synthetic_smoke_v1_20260901_r1.json` with
  status `synthetic_qwen_connector_smoke_pass`. It verifies the 32x5120
  evidence-token contract, embedding insertion, multi-positive MIL and
  evidence-guided masking without importing a Qwen runtime or loading a
  checkpoint. The 88-event real shadow smoke contains 119 event/patient Qwen
  packets; all remain `shadow_input_no_generation` with patient-fact,
  localization-change and generation permissions disabled.

- The private report review service defaults now resolve its inventory/bundle
  and patient-name manifest from the controlled external
  `EEG_Seizure/outputs` store rather than absent repository paths. A read-only
  `_build_dataset` replay found 43 inventory reports, 40 reviewable reports
  and 3 explicitly excluded reports; raw report text/bytes were not persisted
  or copied.

- A clean-worktree review service instance is currently running loopback-only
  at `http://127.0.0.1:8792/` (session bound outside the restricted shell).
  Controlled API checks report 40 reviewable and 3 explicitly excluded
  reports, with `raw_report_text_persisted=false` and
  `raw_report_bytes_copied=false`. This service only stores local review
  drafts; it never issues an institutional release receipt or training
  authorization.

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
schemas/configuration/knowledge, the Stage-0 scripts, the dashboard template,
the controlled method references under `docs/method/reference`, and the
canonical v29 H/D source/checkpoint bundle (`third_party/labram`,
`models/canonical_v29_h_d`, plus local ignored v29 state artifacts). Not copied:
raw EDF/DOCX, internal path maps, private reports, de-identified EDF bundles,
Qwen/CerebraGloss/ELM models and private prediction caches. The parent workspace
remains the source of those controlled inputs. This clean worktree is
`/mnt/hd1/dyf/workspace/laptop/EviSOZ`; its own `.git` history is independent
of the parent repository. Generated outputs are intentionally not migrated.
Recreate target-relative audit receipts after migration; do not copy old
receipts into this repository and do not treat them as authorization.

## Method documents

The current implementation contract and its Evidence JSON runtime map are
maintained in [`docs/method/reference/`](docs/method/reference/). The primary
design document is
[`evisoz_lm_repository_aligned_design_v1_20260830_zh.md`](docs/method/reference/evisoz_lm_repository_aligned_design_v1_20260830_zh.md);
the companion runtime map is
[`evisoz_evidence_json_runtime_usage_v1_20260901_zh.md`](docs/method/reference/evisoz_evidence_json_runtime_usage_v1_20260901_zh.md).
The reference directory also contains the v29 protocol, SOZ target and
reporting contracts needed for a clean-worktree replay.

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

## Verification in this handoff

- `python3 -m py_compile scripts/materialize_evisoz_stage0_remediation_packet_v1.py`: pass.
- `python3 -m compileall -q src scripts`: pass.
- `python3 scripts/materialize_evisoz_schema_registry_v1.py --check`: pass.
- Remediation packet r4 privacy scan: pass; no controlled source path, DOCX
  suffix, report text, or patient-name token is present in the packet.
