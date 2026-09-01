# EviSOZ session handoff

Updated: 2026-09-01 18:45 CST (Asia/Shanghai)

## Current-turn update (ELM runtime probe and worktree alignment)

- All work for this turn remains rooted at
  `/mnt/hd1/dyf/workspace/laptop/EviSOZ`; no code, output, or model bytes were
  written to `EEG_Seizure`.
- Added the schema/validator pair for
  `evisoz_elm_runtime_probe_v1` and registered it in the clean-worktree schema
  registry. The probe pins ELM source commit
  `fcd929a57ce3dc9a409be37a71f4ee80ee59979d` and reads only the external public
  artifact root
  `/mnt/hd1/dyf/workspace/laptop/EviSOZ_artifacts/elm_public_artifacts_v1_20260901_r1`.
- `outputs/evisoz_elm_runtime_probe_v1_20260901_r1/receipt.json` is a
  `synthetic_forward_pass_only` receipt (CPU zeros, strict 5s/60s checkpoint
  load, finite and bitwise-repeat checks). It does not open patient EEG,
  reports, labels, Qwen, or a teacher candidate cache, and does not authorize
  inference, calibration or training.
- Appended the ELM probe boundary to both method references. The probe remains
  weaker than ELM candidate admission: preprocessing/exposure provenance,
  fold-local calibration, and the current aggregate Stage-0 gate are still
  required before any candidate lane can be materialized.
- The repository currently has uncommitted schema/runtime/doc changes. After
  validation, commit them, regenerate a new clean-freeze audit, and replay
  Stage-0 with the new registry/freeze; historical r85/r22 receipts must remain
  untouched.

## Stage-0 replay after the ELM contract commit (r86; pre-handoff commit)

- Commit `b5923bb7c66891a95c5dcd0277e21d694288a5d1` was the clean
  snapshot. The post-commit audit is
  `outputs/clean_freeze_audit_target_v1_20260901_r23.json` with
  `status=GO`, `git_clean=true`, `training_authorized=false`, and receipt
  SHA-256 `7853d156b1196346dff31d1e3a0654d66cbbd97eaf6a3cd10b40960b12834ded`.
- Schema registry check and `python3 -m compileall -q src scripts code` both
  pass. The registry now includes the 56th
  `evisoz_elm_runtime_probe_v1` binding.
- A fresh ELM synthetic replay is
  `outputs/evisoz_elm_runtime_probe_v1_20260901_r2/receipt.json`; it has the
  same content receipt SHA-256 as r1
  (`e6db2f0e6e3040ede7770d527194954746a6a195cc43300f8317f3f084e2e576`) and
  remains `synthetic_forward_pass_only`.
- Stage-0 was replayed with the new registry and r23 freeze, explicit external
  private/public inputs, the active 40-row report candidate bundle and the
  SUAT-signed report-release receipt. The new aggregate gate is
  `outputs/evisoz_stage0_gate_v1_20260901_r86/gate.json` with
  `status=NO_GO`, gate ID `EVISOZ-STAGE0-77db9b2db92c34bfe79e964a`, and
  receipt SHA-256
  `1a9c87c0f04a7050d25b55b3f84cb55235257e77f4eee2647b2550a03d82307f`.
- r86 checks for schema, canonical v29 reference, dual-montage qualification,
  report linkage/release, knowledge, Findings/claim/report, bound evidence and
  clean freeze are green or qualified. Remaining blockers are exactly:
  `private_data_governance_training_authority_missing`,
  `near_or_partial_overlap_closure_incomplete`,
  `tuev_eval_patient_identity_opaque`,
  `elm_candidate_artifact_missing`, and
  `fold_local_calibration_receipts_missing`. Formal DataLoader, optimizer,
  private-label training, Qwen SFT/alignment, non-zero residual and large-scale
  teacher inference remain closed.

The handoff-only commit that records r86 is now `b1ea372`; r23/r86 are
historical and must not be treated as the current freeze/gate. The current
post-commit audit is
`outputs/clean_freeze_audit_target_v1_20260901_r24.json` (`status=GO`,
`git_clean=true`, `training_authorized=false`, receipt SHA-256
`9e35fbb785b725ebfb4854d2428a5ac1924f1c734f6ac0d2354ec0c118cd4723`). A
fresh gate replay against r24 is required before any stage transition.

## Current clean snapshot replay (r87)

- The handoff correction is committed as `00c0a25`. Its current freeze audit is
  `outputs/clean_freeze_audit_target_v1_20260901_r25.json` with
  `status=GO`, `git_clean=true`, `training_authorized=false`, and receipt
  SHA-256 `368a2e1af201d00adb78a8f5f68c7f92e0941fb0754ea2bf796507f625e9ecc5`.
- Gate r87 is
  `outputs/evisoz_stage0_gate_v1_20260901_r87/gate.json`, status `NO_GO`, gate
  ID `EVISOZ-STAGE0-e4469db1f3e9b45e2eb5bcf0`, receipt SHA-256
  `7a904fbc129b103c9e31081e13945503e3e58506daf8163ac099eda171fb0363`.
  The blocking checks are `offline_teacher_and_derived_candidates`,
  `private_field_envelopes`, and `public_auxiliary_patient_exposure_ledger`;
  their blocker codes are respectively ELM candidate/fold-local calibration,
  missing private training governance authority, and near/partial overlap plus
  opaque TUEV evaluation identity.
- r87 confirms `private_report_linkage=GO` through explicit quarantine of the
  three unresolved reports and `private_report_text_release=GO` through the
  externally authorized SUAT release. These statuses do not open localization
  supervision, Qwen generation, or any formal training lane.

## Current-turn update

- Updated `scripts/materialize_evisoz_qwen_patient_shadow_v1.py` so its
  bound-evidence, private examples, Findings/claim/report, dual-montage and
  split-roster inputs are explicit CLI parameters. Defaults now point to the
  current repository bound-evidence r51 and the controlled read-only
  `EEG_Seizure/outputs` roots; the script never copies raw EDF/DOCX or report
  text and remains no-generation.
- Replayed the current 88-event bound evidence with the deterministic shadow
  smoke at `outputs/evisoz_stage0_shadow_inference_smoke_v1_20260901_r17/`
  (88 events, 31 patients), then materialized the patient-level Qwen shadow at
  `outputs/evisoz_stage0_qwen_patient_shadow_v1_20260901_r1/` (31 packets;
  23 development and 8 locked-test patients). Patient graph/report/selection
  replay rates are all 1.0. Runtime policy remains
  `qwen_generation_allowed=false`, `training_allowed=false`,
  `embeddings_materialized=false`, and `patient_fact_creation_allowed=false`.
- The new patient-shadow materialization receipt is
  `9191f6c84435a19e19c2fd70125b0fa06754b876b6e4c911c2a22cf4eed48cd4` and its
  patient evaluation receipt is
  `ef05f837d8c9d4ffbc7b05c8e0dd0b6ca016fb672af62107255b1fc3b35dc052`.
- The input-boundary change is committed as `f0342e1` and this handoff update
  as `2c71d61`; a post-commit
  clean-freeze audit at
  `outputs/clean_freeze_audit_target_v1_20260901_r20.json` was `GO` with
  `git_clean=true` (receipt SHA-256
  `5c0b8a8b4d60242a1cabfcbd2a609ff584f249b6138e165ebe2a22d492470d0b`).
- A new execution-plan replay against Stage-0 gate r85 is
  `outputs/evisoz_execution_plan_v1_20260901_r44/plan.json` with status
  `STAGE0_NO_GO` and receipt SHA-256
  `c3c7d52e4ad053596dffba8d8a4b42f72f898e76a568b3247b4b2e2b8ff00615`.
- The subsequent handoff-only commit is `9a595e3`; regenerate the
  non-authorizing clean-freeze audit after this commit before treating r20 as
  current.

- All commands for this turn were run from the clean worktree
  `/mnt/hd1/dyf/workspace/laptop/EviSOZ`; `EEG_Seizure` was used only through
  explicit read-only external paths. No raw EDF/DOCX, patient mapping or
  checkpoint bytes were copied into this repository.
- Added an optional `--private-report-release` input to the Stage-0 gate. The
  gate now validates an externally authorized
  `evisoz_private_physician_report_release_v1` against the exact candidate
  manifest and candidate text bytes before it can mark
  `private_report_text_release=GO`; absent or incomplete release remains
  `NO_GO`. This change does not authorize EEG training, localization loss or
  Qwen generation.
- The external SUAT release replay was independently validated for 40 rows
  (33 development Qwen-training rows and 7 locked language-evaluation rows),
  while the controller authorization remains outside the repository. A fresh
  aggregate gate must be generated after the code commit and clean-freeze.

## Current state

- Stage-0 replay: `NO_GO` at the current clean-worktree gate
  `outputs/evisoz_stage0_gate_v1_20260901_r78/gate.json` (blocking checks:
  offline teacher/calibration, private governance authority, report text
  release, and public auxiliary exposure ledger).
- `private_report_linkage`: `GO` after an explicit operational quarantine receipt.
- The latest clean-freeze audit before this handoff update was `GO` at
  `outputs/clean_freeze_audit_target_v1_20260901_r13.json` (receipt SHA-256
  `97708ed0d4e3dba35ec8a235085aae0456b1d1ef1a427cc5c05da9383b185286`). The
  receipt is regenerated after each committed change; inspect its audit ID
  and SHA-256 in the local file rather than treating this handoff line as an
  authorization.
- Remaining independent blockers are unchanged: private governance training
  authority, report manual review/release, public overlap/TUEV identity,
  CerebraGloss/ELM candidates and fold-local calibration, plus the parent
  worktree clean-freeze audit.

- The clean-worktree ELM discovery replay is
  `outputs/evisoz_teacher_artifact_discovery_v1_20260901_evisoz_artifacts_r1.json`.
  It scans only the external
  `/mnt/hd1/dyf/workspace/laptop/EviSOZ_artifacts/elm_public_artifacts_v1_20260901_r1`
  root, records four full SHA-256 hashes, and remains
  `found_unvalidated` with training/inference disabled. A new remediation
  packet `outputs/evisoz_stage0_remediation_packet_v1_20260901_r6/` binds this
  discovery receipt explicitly via `--elm-discovery`; it does not promote ELM
  candidates or copy model bytes into the repository.

- Aggregate Stage-0 replay r78 was executed with explicit external private
  inputs, the r13 clean-freeze receipt, and the ELM discovery receipt above.
  It remains `NO_GO` with blockers limited to offline teacher/calibration,
  private governance authority, report text release, and public overlap/TUEV
  identity. No model, optimizer, loader, teacher runtime, Qwen generation or
  residual was opened.

- The post-change clean-freeze audit
  `outputs/clean_freeze_audit_target_v1_20260901_r12.json` was `GO` at commit
  `0e71671` (`tracked_modified=0`, no untracked files). It is retained as a
  historical pre-handoff snapshot; after any subsequent commit the audit must
  be regenerated before claiming the current clean freeze. All such audits
  remain non-authorizing for training.

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
