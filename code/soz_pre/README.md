# SOZ Pretraining Pipeline

This package implements a heterogeneous SOZ localization pipeline:

- TUSZ weak scalp-onset labels -> region-level pretraining.
- Private doctor significant electrodes -> strong SOZ supervision.
- Private doctor early spread/diffuse labels -> propagation or SOZ-loss masks.
- Optional SPHL/SPHR input features -> left/right temporal priors without forcing
  channel-level labels when the source annotations do not provide them.

## 1. Build Private Manifest

```bash
python3 code/soz_pre/build_private_edf_soz_manifest.py \
  --eeg_root /mnt/hd1/dyf/dataset/EEG \
  --doctor_summary /mnt/hd1/dyf/dataset/EEG/发作起始通道汇总.csv \
  --edf_annotations /mnt/hd1/dyf/dataset/EEG/edf_annotations.csv \
  --output outputs/soz_pre/private_edf_soz_manifest.csv
```

The output contains canonical channel labels, label masks, propagation labels,
region labels, hemisphere labels, review flags, and original raw text.

## 2. Merge With TUSZ

```bash
python3 code/soz_pre/build_unified_region_manifest.py \
  --tusz_manifest outputs/deepsoz/tusz_v203_manifest_vote_v1.csv \
  --private_manifest outputs/soz_pre/private_edf_soz_manifest.csv \
  --output outputs/soz_pre/unified_region_soz_manifest.csv
```

By default, generalized TUSZ seizure types are kept for temporal seizure
training but masked out of spatial SOZ loss.

## 3. Preprocess EDF Signals

Small smoke:

```bash
python3 code/soz_pre/preprocess_unified_soz.py \
  --manifest outputs/soz_pre/unified_region_soz_manifest.csv \
  --output_dir outputs/soz_pre/preprocessed_smoke \
  --max_rows 4 \
  --roles onset \
  --include_sph
```

Full run:

```bash
python3 code/soz_pre/preprocess_unified_soz.py \
  --manifest outputs/soz_pre/unified_region_soz_manifest.csv \
  --output_dir outputs/soz_pre/preprocessed \
  --tusz_root /mnt/hd1/dyf/dataset/TUSZ \
  --private_root /mnt/hd1/dyf/dataset/EEG \
  --roles onset,early_ictal,propagation,background \
  --include_sph
```

NPZ shape is `[T, C, W]`, where `C=22` or `C=24` when SPHL/SPHR are included.
The channel SOZ labels remain the canonical 22 TCP channels.

The same command now resolves both common TUSZ roots:
`/mnt/hd1/dyf/dataset/TUSZ` and
`/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf`. It also resolves private EDF paths
relative to `/mnt/hd1/dyf/dataset/EEG`.

If the conservative QC pipeline has already produced window-level reports, pass
the QC output root to align ArtifactScore with every SOZ sample:

```bash
python3 code/soz_pre/preprocess_unified_soz.py \
  --manifest outputs/soz_pre/unified_region_soz_manifest.csv \
  --output_dir outputs/soz_pre/preprocessed \
  --tusz_root /mnt/hd1/dyf/dataset/TUSZ \
  --private_root /mnt/hd1/dyf/dataset/EEG \
  --qc_root outputs \
  --roles onset,early_ictal,propagation,background \
  --include_sph
```

Each NPZ contains `artifact_score` and `artifact_mask` with shape `[T, C]`.
For TCP bipolar channels, the score is the max score of the two endpoint
electrodes in the QC report; SPHL/SPHR use their own QC channel when present.
By default these scores are metadata only. Use `--artifact_weight_mode
downweight` only for an explicit experiment that lowers `sample_weight` by mean
ArtifactScore.

## 4. Train TUSZ Region Pretraining

```bash
python3 code/soz_pre/train_region_soz.py \
  --preprocessed_dir outputs/soz_pre/preprocessed \
  --output_dir outputs/soz_pre/tusz_region_pretrain \
  --train_splits train \
  --val_splits dev \
  --train_sources tusz \
  --val_sources tusz \
  --epochs 40
```

## 5. Private LOPO Fine-Tuning/Evaluation

From scratch on private:

```bash
python3 code/soz_pre/run_private_lopo.py \
  --preprocessed_dir outputs/soz_pre/preprocessed \
  --output_dir outputs/soz_pre/private_lopo \
  --epochs 40
```

Fine-tune from TUSZ pretraining:

```bash
python3 code/soz_pre/run_private_lopo.py \
  --preprocessed_dir outputs/soz_pre/preprocessed \
  --output_dir outputs/soz_pre/private_lopo_from_tusz \
  --init_checkpoint outputs/soz_pre/tusz_region_pretrain/best_model.pt \
  --epochs 40
```

## 6. Report

```bash
python3 code/soz_pre/report_soz_results.py \
  --run_dir outputs/soz_pre/private_lopo_from_tusz
```

The report includes sample-level and patient-macro region top-1 hit,
hemisphere accuracy, and channel top-k hit.
