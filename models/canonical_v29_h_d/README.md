# Canonical v29 H/D model artifacts

This directory contains the local, byte-verified LaBraM foundation checkpoint
used by the canonical v29 H/D reference route.  The checkpoint is ignored by
Git because it is a large binary artifact.  Its exact SHA-256 and provenance
are recorded in [`artifact_manifest.json`](artifact_manifest.json).

The v29 H/D reasoners are the small, frozen state containers under the
repository's ignored `outputs/` artifact root:

- D: `outputs/labram_rank1_direct_token_oof_v28_20260815/model_and_oof.safetensors`
- H: `outputs/labram_identity_recovery_closed_replay_v16_replay_20260815/outer_fold_states.safetensors`

The public freeze and roster manifests required by the fail-closed registry
are copied beside them under `outputs/`.  They remain artifacts rather than
source-of-truth code.  No private prediction, patient mapping, raw EDF/DOCX,
or Qwen/CerebraGloss/ELM model is included here.

`third_party/labram/modeling_finetune.py` is the MIT-licensed upstream model
definition pinned by the same manifest.  The v29 loader checks both source and
checkpoint hashes before constructing the encoder.
