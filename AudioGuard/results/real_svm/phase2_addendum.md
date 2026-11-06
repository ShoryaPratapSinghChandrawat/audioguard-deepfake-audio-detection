# AudioGuard Phase 2.5: independent SVM verification

## Data and validation

Independent recomputation reads only saved score CSVs for scores/labels, the saved TRAIN artifact for the full-precision frozen threshold, and official protocol files for trial metadata validation. The validation pass ran before any metric calculation.

DEV: 24,844/24,844 protocol trials, 24,844 unique IDs; target 0 bona fide=2,548, target 1 spoof=22,296; no blank values, nonfinite scores, duplicate IDs, or metadata mismatches.
EVAL: 71,237/71,237 protocol trials, 71,237 unique IDs; target 0 bona fide=7,355, target 1 spoof=63,882; no blank values, nonfinite scores, duplicate IDs, or metadata mismatches.
Frozen DEV threshold loaded from training.json without retyping: 0.8227455542399777.

## Independent pooled metrics

EER nearest = mean(FPR,FNR) at the finite sklearn ROC point minimizing |FPR-FNR|. Interpolated EER is the linear interpolation between adjacent ROC points at which FPR-FNR crosses zero. Pooled confusion uses scores >= the saved threshold as spoof.

| Split | AUC | Nearest EER | Interpolated EER | TN / FP / FN / TP |
|---|---:|---:|---:|---|
| DEV | 97.192246% | 8.242695% | 8.241758% | 2,338 / 210 / 1,838 / 20,458 |
| EVAL | 93.899198% | 12.782465% | 12.784509% | 7,028 / 327 / 13,400 / 50,482 |

## Recomputed vs saved Phase 2 values

AudioGuard Phase 2 used the same nearest finite empirical ROC point and mean(FPR,FNR). AUC and confusion matrices should match directly. Interpolated EER is a separate added estimate; differences from saved EER are expected because discrete ROC thresholds need not hit equality exactly.

| Split | AUC delta | Nearest EER delta | Interpolated EER minus saved nearest | Confusion match |
|---|---:|---:|---:|---|
| DEV | 0 | 2.77555756156e-17 | -9.3645142084e-06 | True |
| EVAL | 0 | 2.77555756156e-17 | 2.04372818502e-05 | True |

Fixed-threshold metric comparison to saved Phase 2 values (delta = independent minus saved):

| Split | Accuracy delta | Spoof F1 delta | Macro-F1 delta | Balanced accuracy delta | Spoof miss delta | Bona-fide error delta |
|---|---:|---:|---:|---:|---:|---:|
| DEV | 0 | 0 | 0 | 0 | 0 | 0 |
| EVAL | 0 | 0 | 0 | 0 | 0 | 0 |

Per-attack EVAL differences to saved Phase 2 metrics (delta = independent minus saved):

| Attack | Nearest EER delta | Interpolated EER minus saved nearest | Miss-rate delta |
|---|---:|---:|---:|
| A07 | 1.38777878078e-17 | 2.82078120284e-05 | 0 |
| A08 | -4.16333634234e-17 | 1.53697298496e-05 | 0 |
| A09 | -2.08166817117e-17 | -3.66604717387e-06 | 0 |
| A10 | -1.38777878078e-17 | 4.42692488919e-05 | 0 |
| A11 | -2.08166817117e-17 | -2.59251788824e-05 | 0 |
| A12 | -2.08166817117e-17 | -2.59251788824e-05 | 0 |
| A13 | -2.08166817117e-17 | -2.59251788824e-05 | 0 |
| A14 | 4.16333634234e-17 | -3.14311667132e-05 | 0 |
| A15 | -2.77555756156e-17 | -2.78896268019e-05 | 0 |
| A16 | 0 | 1.83994065706e-06 | 0 |
| A17 | 0 | 3.37276339996e-05 | 0 |
| A18 | -5.55111512313e-17 | 4.13087428723e-05 | 0 |
| A19 | 0 | 3.75181884359e-05 | 0 |

The JSON sidecar records unrounded values and confusion matrices. Recomputed global AUC, nearest EER, confusion matrices, and all thresholded metrics agree with saved Phase 2 values apart from floating-point round-off. Linear-interpolated EER is intentionally distinct. DEV per-attack EERs were not saved in Phase 2, so no prior DEV per-attack value exists to compare.

## Per-attack DEV and EVAL

DEV attack EER uses that attack and the shared 2,548 bona-fide DEV pool. EVAL attack EER uses that attack and the shared 7,355 bona-fide EVAL pool. Miss rate is FN divided by spoof attack clips at the frozen DEV threshold.

| Split | Attack | Bona fide pool | Spoof clips | Nearest EER | Interpolated EER | Frozen-threshold miss |
|---|---|---:|---:|---:|---:|---:|
| DEV | A01 | 2,548 | 3,716 | 5.492140% | 5.494505% | 0.430571% |
| DEV | A02 | 2,548 | 3,716 | 3.800655% | 3.794403% | 1.157158% |
| DEV | A03 | 2,548 | 3,716 | 6.560180% | 6.566200% | 5.247578% |
| DEV | A04 | 2,548 | 3,716 | 10.234689% | 10.226050% | 12.567277% |
| DEV | A05 | 2,548 | 3,716 | 8.078985% | 8.073197% | 7.857912% |
| DEV | A06 | 2,548 | 3,716 | 13.969187% | 13.971743% | 22.201292% |
| EVAL | A07 | 7,355 | 4,914 | 5.721176% | 5.723997% | 10.175010% |
| EVAL | A08 | 7,355 | 4,914 | 4.539592% | 4.541128% | 5.168905% |
| EVAL | A09 | 7,355 | 4,914 | 2.787586% | 2.787220% | 0.000000% |
| EVAL | A10 | 7,355 | 4,914 | 4.074431% | 4.078858% | 3.052503% |
| EVAL | A11 | 7,355 | 4,914 | 3.863911% | 3.861319% | 0.101750% |
| EVAL | A12 | 7,355 | 4,914 | 3.863911% | 3.861319% | 2.787953% |
| EVAL | A13 | 7,355 | 4,914 | 3.863911% | 3.861319% | 0.061050% |
| EVAL | A14 | 7,355 | 4,914 | 2.892846% | 2.889703% | 0.997151% |
| EVAL | A15 | 7,355 | 4,914 | 2.566892% | 2.564103% | 0.936101% |
| EVAL | A16 | 7,355 | 4,914 | 9.544344% | 9.544528% | 18.315018% |
| EVAL | A17 | 7,355 | 4,914 | 34.354207% | 34.357580% | 81.725682% |
| EVAL | A18 | 7,355 | 4,914 | 22.551953% | 22.556084% | 69.515670% |
| EVAL | A19 | 7,355 | 4,914 | 28.453080% | 28.456832% | 79.853480% |

### Seen/unseen attack comparisons

| Comparison | DEV attack nearest / interpolated EER | DEV miss | EVAL attack nearest / interpolated EER | EVAL miss |
|---|---:|---:|---:|---:|
| A04 vs A16 | 10.234689% / 10.226050% | 12.567277% | 9.544344% / 9.544528% | 18.315018% |
| A06 vs A19 | 13.969187% / 13.971743% | 22.201292% | 28.453080% / 28.456832% | 79.853480% |

These are measured results only; no similarity is inferred from attack-family metadata.

## EVAL always-spoof baseline

| System | Accuracy | Spoof F1 | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Always predicts spoof | 89.675309% | 94.556650% | 47.278325% | 50.000000% |
| MFCC + SVM | 80.730519% | 88.031319% | 69.311704% | 87.288935% |

## Published context and polarity

Published ASVspoof 2019 LA evaluation EERs were 8.09% for LFCC-GMM and 9.57% for CQCC-GMM. These are contextual only; they use different features/classifiers and evaluation conventions, so this is not a like-for-like comparison. The official evaluation plan treats bona fide as the positive class and expects higher scores to indicate bona fide. AudioGuard encodes spoof as 1 and uses higher scores to indicate spoof; its internal mapping and score polarity checks are consistent. See the [ASVspoof 2019 baseline results](https://www.apsipa.org/proceedings/2019/pdfs/246.pdf) and [official ASVspoof 2019 evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf).

## Unresolved extra audio

All 116 unmatched extra audio files remain UNRESOLVED. Filename patterns only:
- LA_D_A*: 20
- LA_E_A*: 96

No enrollment-file attribution is made. Exact stems and comparison provenance remain in extra_files_comparison.json.

## Reproduction and artifacts

Rerun from the AudioGuard project root with: `.venv/Scripts/python.exe results/real_svm/recompute_phase2.py`.

The script validates both CSVs before calculating metrics and writes phase2_score_csv_validation.json, phase2_independent_metrics.json, and this addendum, plus phase2_per_attack.csv. It does not modify Phase 2 score files, model artifacts, or core research code.
