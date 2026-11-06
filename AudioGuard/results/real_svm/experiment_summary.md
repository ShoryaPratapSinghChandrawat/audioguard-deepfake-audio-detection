# AudioGuard Phase 2: ASVspoof 2019 LA MFCC + SVM

## Scope and commands

PHASE 2 STATUS: PASS. Trained the full TRAIN partition and scored complete clean DEV/EVAL. No CNN, parameter search, or robustness condition was run. Per the user's clean-only selection, the repository evaluate command was not invoked because its documented path also runs trim, noise, and MP3 conditions.

Commands (exit code 0):
- .venv/Scripts/python.exe scripts/experiment.py train --config configs/research.local.yaml --model svm_mfcc
- .venv/Scripts/python.exe scripts/clean_evaluate.py --config configs/research.local.yaml --model svm_mfcc
- .venv/Scripts/python.exe scripts/extra_metrics.py --config configs/research.local.yaml --model svm_mfcc
- .venv/Scripts/python.exe scripts/verify_research.py --config configs/research.local.yaml
- .venv/Scripts/python.exe -m pytest -q — 24 passed in 30.28s

Environment: Python 3.11.9; torch 2.5.1+cu121; CUDA available on NVIDIA GeForce RTX 2050. The baseline itself uses scikit-learn SVC.

## Dataset and validation

Dataset root: C:/Games/projects/LA/LA

| Split | Total | Bona fide | Spoof | Speakers | Attack IDs |
|---|---:|---:|---:|---:|---|
| Train | 25,380 | 2,580 | 22,800 | 20 | A01–A06 |
| Dev | 24,844 | 2,548 | 22,296 | 20 | A01–A06 |
| Eval | 71,237 | 7,355 | 63,882 | 67 | A07–A19 |

Loader counts match the complete expected partitions. Phase 1 audit found zero malformed protocol rows, missing audio, speaker overlap, and utterance overlap. Full training used all 25,380 TRAIN rows. Cache scan found all 121,461 expected feature matrices valid, finite, float32, 60-by-T, with pooled dimension 120; zero invalid matrices.

## Features and resolved SVM

Audio is mono, 16 kHz; the reader rejects sample-rate/channel mismatches and does not resample. Frontend: 20 MFCCs, 64 HTK mel filters, 512-point FFT, 512-sample Hann window, 160-sample hop, 9-frame delta and delta-delta. Per-coefficient mean concatenated with standard deviation gives 120 pooled dimensions from 60 float32 frame coefficients. StandardScaler and SVC were fit on TRAIN only.

Effective settings saved before training: C=1.0, RBF kernel, gamma=scale, class_weight=balanced, probability=False, random_state=2026. Scores are continuous SVC decision_function values. The existing nearest-finite empirical DEV ROC EER threshold was frozen at 0.822745554240 before EVAL scoring.

## DEV and clean EVAL

Confusion matrix order is bona fide then spoof; cells reported TN / FP / FN / TP. Spoof is the positive class.

| Split | N | Bona precision | Bona recall | Spoof precision | Spoof recall | Spoof F1 | Macro F1 | Balanced accuracy | ROC-AUC | Pooled EER | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DEV | 24,844 | 55.99% | 91.76% | 98.98% | 91.76% | 95.23% | 82.39% | 91.76% | 97.19% | 8.24% | 2,338 / 210 / 1,838 / 20,458 |
| EVAL clean | 71,237 | 34.40% | 95.55% | 99.36% | 79.02% | 88.03% | 69.31% | 87.29% | 93.90% | 12.78% | 7,028 / 327 / 13,400 / 50,482 |

Definitions: TP = spoof correctly classified spoof; FN = spoof classified bona fide; FP = bona fide classified spoof; TN = bona fide correctly classified bona fide. Requested spoof false-accept rate = FN/(TP+FN), which under this requested formula equals conventional FNR = FN/(FN+TP), denominator all spoof trials. Requested bona-fide false-reject rate = FP/(TN+FP), which equals conventional FPR = FP/(FP+TN), denominator all bona-fide trials.

- DEV: spoof FN rate 1,838/22,296 = 8.24%; bona-fide FP rate 210/2,548 = 8.24%.
- EVAL: spoof FN rate 13,400/63,882 = 20.98%; bona-fide FP rate 327/7,355 = 4.45%.

## Score polarity

SVC classes_ = [0, 1]; decision_function positive side is class 1 (spoof). AudioGuard predicts spoof at score >= threshold. DEV AUC was 0.9719224580 on original scores and 0.0280775420 on negated scores; clean EVAL AUC was 0.9389919798 on original scores and 0.0610080202 on negated scores. Polarity agrees with high-score-means-spoof; scores were not reversed.

## EVAL per-attack diagnostics

Each attack EER is diagnostic only, computed against the same shared 7,355 bona-fide EVAL pool; it is not a deployment threshold. Fixed-threshold spoof miss = FN/(TP+FN), using the single frozen DEV threshold. Bona-fide FP rate is shared across attacks: 327/7,355 = 4.45%.

| Attack | Spoof clips | EER diagnostic | Spoof FN rate at frozen threshold | TP | FN |
|---|---:|---:|---:|---:|---:|
| A07 | 4,914 | 5.72% | 10.18% | 4,414 | 500 |
| A08 | 4,914 | 4.54% | 5.17% | 4,660 | 254 |
| A09 | 4,914 | 2.79% | 0.00% | 4,914 | 0 |
| A10 | 4,914 | 4.07% | 3.05% | 4,764 | 150 |
| A11 | 4,914 | 3.86% | 0.10% | 4,909 | 5 |
| A12 | 4,914 | 3.86% | 2.79% | 4,777 | 137 |
| A13 | 4,914 | 3.86% | 0.06% | 4,911 | 3 |
| A14 | 4,914 | 2.89% | 1.00% | 4,865 | 49 |
| A15 | 4,914 | 2.57% | 0.94% | 4,868 | 46 |
| A16 | 4,914 | 9.54% | 18.32% | 4,014 | 900 |
| A17 | 4,914 | 34.35% | 81.73% | 898 | 4,016 |
| A18 | 4,914 | 22.55% | 69.52% | 1,498 | 3,416 |
| A19 | 4,914 | 28.45% | 79.85% | 990 | 3,924 |

A17, A19, and A18 had the highest observed attack EERs and frozen-threshold spoof miss rates. Train/DEV contain A01–A06; EVAL contains A07–A19, IDs absent from TRAIN/DEV protocols. Unseen attack ID does not establish an unseen synthesis family. The ASVspoof 2019 database paper documents algorithm reuse: A16 uses the same TTS algorithm as A04, and A19 the same VC algorithm as A06 ([Wang et al., 2020](https://doi.org/10.1016/j.csl.2020.101114)); these are algorithm-reuse cases, not 13 entirely new algorithms.

## Extra audio files

Compared exact filename stems for all extra DEV/EVAL audio against 10 ASV protocol files, including enrollment lists.

| Split | Extra stems | Matched | Unmatched |
|---|---:|---:|---:|
| Dev | 142 | 122 | 20 |
| Eval | 696 | 600 | 96 |
| Total | 838 | 722 | 116 |

All 722 matches were found in ASV enrollment .trn files; none were found only in trial .trl files. The enrollment explanation applies to those matched files only; 116 remain unmatched.

## Runtime and verification

Initial free space: 133.68 GiB. Final feature cache: 121,461 files, 8.765 GiB. Final C: free space: 125.447 GiB. A reproducible 500-clip TRAIN-only DEBUG ONLY extraction took 4.070 seconds; estimated full cache 9.379 GiB and extraction 988.6 seconds. This timing is not a performance result. Full training took 469.653 seconds; clean EVAL took 612.13 seconds; combined 1,081.783 seconds. Feature-cache integrity scan: 150.948 seconds, all matrices valid.

Research verification passed for svm_mfcc. Raw command logs are under results/real_svm/logs. Model, scaler, full per-clip scores/predictions, metrics, and attack diagnostics are under results/real_svm/svm_mfcc.

The results are being promoted to a new phase-2-real-svm branch created from the supplied GitHub repository main branch. Local-only configs and bulk artifacts remain ignored. The report and selected small evidence files are staged explicitly; no per-clip score CSV, model weight, dataset, venv, or feature cache is included.



## Phase 2.5 staging rule

When the Phase 2.5 report is generated, include results/real_svm/phase2_addendum.md in the small, explicitly staged result-file list and force-stage that named file if needed because results are ignored by default. Never commit per-clip score CSVs, including dev_scores.csv and eval_clean_scores.csv, regardless of file size; retain the saved CSVs locally for verification. Do not stage a wildcard under results/real_svm.
