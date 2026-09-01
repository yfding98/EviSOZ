# Private SOZ annotation and adjudication manual

**Status:** clinician sign-off required before any private-label evaluation  
**Scope:** `EEG-fMRI颞叶癫痫(1).xls`, `头皮扩散.xlsx`, and their exact
event-to-EDF crosswalk  
**Model role:** locked post-training clinical transfer evaluation only

## 1. Clinical construct to confirm

For each seizure event, `significant electrode` is interpreted as the set of
recorded scalp electrodes that the reviewing clinician judges to provide the
strongest support for the event-level clinical SOZ hypothesis after reviewing
the complete seizure EEG and the available clinical manifestations.

It is not automatically equivalent to:

- the first scalp-visible change;
- the maximum-amplitude channel;
- every channel involved during the seizure;
- a cortical, SEEG/ECoG, resection, or epileptogenic-zone target; or
- the model prediction.

Clinician confirmation:

- [ ] This definition matches the original meaning of `significant`.
- [ ] This definition does not match. Correct definition: ________________

## 2. Information used to create the reference

Complete for the cohort and, when it varies, for each event:

- EEG reviewed: [ ] full seizure [ ] selected onset clip [ ] report only
- Clinical manifestations/semiology: [ ] available and used [ ] unavailable
- Imaging: [ ] used [ ] not used [ ] unknown
- Invasive EEG: [ ] used [ ] not used [ ] unknown
- Prior diagnostic report: [ ] used [ ] not used [ ] unknown
- Reviewer was blinded to model output: [ ] yes [ ] no

Each evaluable event must receive one `localization_basis`:

```text
EEG_primary
EEG_and_semiology_concordant
semiology_resolved_ambiguous_EEG
other_privileged_information_used
unknown
```

The last three states remain evaluable for descriptive transfer, but must be
reported separately because the EEG-only model cannot observe all information
used by the reference.

## 3. Exhaustiveness of the significant set

Choose one cohort-level policy, or annotate this field per event:

- [ ] **Exhaustive:** every recorded standard-19 electrode was considered;
  listed electrodes are positive and every reviewed, unlisted standard-19
  electrode is an operational reference negative.
- [ ] **Positive-only:** listed electrodes are positive; unlisted electrodes
  are `not_marked/unknown`, not confirmed negative.
- [ ] **Mixed:** exhaustiveness is recorded per event as
  `exhaustive | positive_only | unknown`.

Consequences are fixed:

- exhaustive cases: ranking plus AP/AUROC/F1/Jaccard and calibration are
  allowed;
- positive-only or unknown cases: Hit@K, MRR/positive recall, laterality,
  region agreement, and significant-versus-known-spread ranking only;
- missing or blank cells never become an all-negative target.

## 4. Meaning of source values

The clinician must confirm the following mappings:

| Source state | Proposed interpretation | Confirm/correct |
|---|---|---|
| named electrode(s) in `significant` | event-level clinical SOZ-reference positive | ____ |
| blank `significant` | missing/unknown | ____ |
| `无` in `significant` | no localizable positive stated; not automatically 19 negatives | ____ |
| named electrode(s) in `early spread` | known clinical spread reference, separate from SOZ positive | ____ |
| `弥漫/全导/覆盖全导` | diffuse involvement descriptor; creates neither 19 SOZ positives nor 19 spread electrodes | ____ |
| blank spread | missing/unknown spread | ____ |
| `无` in spread | no spread stated under the review protocol | ____ |

## 5. Significant versus spread

`spread` is never merged into the SOZ-positive set. It is not assumed to be an
absolute biological non-SOZ electrode. Where both sets are available, it may
support the directional clinical test:

```text
score(significant electrode) > score(known spread electrode)
```

If one electrode appears in both fields for the same event, the event is
quarantined until the clinician selects exactly one action:

- [ ] retain as significant only;
- [ ] retain as spread only;
- [ ] retain in both because the fields encode different time/context states;
  exclude that electrode pair from significant-versus-spread ranking;
- [ ] mark the event reference unresolved and exclude it.

No programmatic priority rule is permitted.

## 6. Electrode coordinate policy

The model output space is:

```text
FP1 FP2 F7 F3 FZ F4 F8 T7 C3 CZ C4 T8 P7 P3 PZ P4 P8 O1 O2
```

Only identity aliases are allowed: `T3→T7`, `T4→T8`, `T5→P7`, `T6→P8`.

- `SPHL/SPHR`, `A1/A2`, and `OZ` remain outside the standard-19 head.
- They are not reassigned to a neighboring scalp electrode or region.
- An event with both in-head and outside-head positives is evaluated on the
  in-head subset and carries an outside-head coverage flag.
- An event with only outside-head positives is not converted into an all-zero
  standard-19 target; it enters coverage/abstention analysis only.

Clinician confirmation of outside-head handling: [ ] confirmed [ ] corrected
policy: _________________________________________________________________

## 7. Duplicate-event adjudication

The two source workbooks contain three duplicate patient-event keys whose
onset, significant, and spread fields conflict. For each duplicate key, a
clinician must review the original cells and linked EDF and record:

| Pseudonymous event key | chosen onset/source text | final significant | final spread | decision basis | reviewer/date |
|---|---|---|---|---|---|
| pending-1 | | | | | |
| pending-2 | | | | | |
| pending-3 | | | | | |

Allowed outcomes are `adjudicated` or `exclude_unresolved`; workbook order,
latest-file priority, union, intersection, and majority are forbidden.

## 8. Event eligibility record

Every private event opened for frozen evaluation must contain:

```text
pseudonymous_patient_id
event_id
linked_edf_and_event_anchor
significant_electrodes_native
spread_electrodes_native
standard19_significant_subset
standard19_spread_subset
outside_head_positive_flag
significant_set_exhaustiveness
localization_basis
duplicate_adjudication_state
significant_spread_overlap_state
reviewer_id_and_review_date
```

Events missing the required decision fields remain locked. Patients, not
events, are the bootstrap unit.

## 9. Reader-study and reference quality

Record the number and qualification of annotators, whether annotations were
independent, the disagreement/adjudication procedure, and inter-rater
agreement when more than one reviewer is available. A single-expert reference
is acceptable for an exploratory internal retrospective study if disclosed as
such; it must not be described as a universal clinical gold standard.

## 10. Sign-off

By signing, the clinical reviewer confirms the construct definition, source
value interpretation, exhaustiveness policy, spread separation, and
outside-head handling above. This sign-off authorizes only locked evaluation;
it does not authorize training, calibration, threshold selection, report
editing, or model selection on private labels.

```text
Clinician name/ID:
Qualification:
Institution:
Signature:
Date:
Protocol version:
```
