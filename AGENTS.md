# EviSOZ clean-worktree instructions

This repository is the independent clean worktree for EviSOZ-LM. Read this
file, `README.md` and `SESSION_HANDOFF.md` before changing code or running a
pipeline.

## Safety gate

- Treat the newest Stage-0 gate as authoritative for the current artifact
  set. A `NO_GO` or missing gate permits only read-only audits,
  schema/receipt materializers, bound-evidence loader replay and shadow smoke.
- Do not construct a training `DataLoader`, optimizer, Qwen SFT job,
  EEG-to-Qwen alignment job, private-label fit, or non-zero residual while the
  gate is not `GO`.
- Never overwrite historical receipts. Every rerun writes a new
  content-addressed/versioned output under an ignored output root.

## Data and provenance

- Raw EDF/DOCX, patient-name mappings, checkpoints and generated bundles stay
  outside this repository. Do not copy them here.
- `knowledge/eeg` supplies terminology, rules, differential considerations
  and reporting boundaries only. It cannot create patient facts or labels.
- CerebraGloss, ELM and deterministic signal features are offline candidate
  teachers. They are not hard SOZ truth.
- TCP22 is a signed bipolar-edge view for phase reversal, morphology and
  propagation. It must not be expanded into Standard19 SOZ endpoint labels.
- The frozen SOZ reference is canonical v29 H/D, Standard19 + CAR. Missing
  channels use explicit masks; spherical interpolation is an independent
  ablation only.
- Physician-authored reports remain `physician_authored`; they are not
  `generated_text` and require manual review/release before training or
  evaluation.

## Worktree and paths

- Run commands from `/mnt/hd1/dyf/workspace/laptop/EviSOZ`.
- Controlled raw inputs and historical artifacts are normally under
  `/mnt/hd1/dyf/workspace/laptop/EEG_Seizure/outputs` and
  `/mnt/hd1/dyf/dataset`; pass those locations explicitly rather than relying
  on `../outputs`.
- Keep generated artifacts in ignored `outputs/` or an external artifact
  store. Do not add them to Git.
- Legacy launcher scripts that reference components not present in this
  clean worktree must not be used as evidence of a runnable EviSOZ stage;
  first audit their dependencies and record a new receipt.

## Required verification

After a code or schema change, run the smallest relevant tests and update the
handoff with a new timestamp/status. Before declaring a stage complete, verify
patient-level split isolation, schema registry consistency, provenance and
the corresponding evaluation receipt.
