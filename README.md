# AudioGuard

AudioGuard is a reproducible research workspace for detecting spoofed speech in
the Logical Access (LA) partition of ASVspoof 2019. The repository includes
reports from experiments on the official train, development, and evaluation
partitions. These findings apply to the reported ASVspoof 2019 LA setup; they do
not establish universal deepfake detection.

## Dataset and provenance

The official ASVspoof 2019 LA audio and protocols are **not redistributed** in
this repository. Obtain them from the [ASVspoof 2019 challenge page](https://www.asvspoof.org/index2019.html)
or [University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336).
The dataset's own terms apply.

This project was imported from an AudioGuard source archive; it was not written
from scratch. Under the owner's specification, AI coding agents (Codex,
Antigravity, and Arena) wrote most of the code. The owner specified, ran, and
verified the experiments. This provenance statement is an owner-provided
disclosure.

## Research reports

| Phase | Evidence | Scope |
| --- | --- | --- |
| Phase 2 | [MFCC + SVM report](AudioGuard/results/real_svm/experiment_summary.md) | Full official LA partitions; clean evaluation |
| Phase 2.5 | [Independent SVM verification](AudioGuard/results/real_svm/phase2_addendum.md) | Saved score CSV validation and independent metric recomputation |
| Phase 3 | [Log-Mel CNN report](AudioGuard/results/real_cnn_logmel/phase3_summary.md) | One-seed full official LA training and clean evaluation |
| Phase 4A | [Error and shortcut analysis](AudioGuard/results/phase4a/phase4a_summary.md) | Saved clean EVAL scores, speaker spread, nuisance-cue analysis |
| Phase 4B | [Verified robustness report](AudioGuard/results/phase4b/optimized_results/phase4b_summary_verified.md) | Fixed 10,000-clip **SUBSET RESULTS**; noise, MP3, and trimming |
| Phase 5.1 | [Resampling diagnostics](AudioGuard/results/phase5_diagnostics/summary.md) | Small diagnostic sample; simulated resampling sensitivity |

Phase 4B is explicitly a fixed subset and is not a full-EVAL robustness result.
The older [synthetic smoke report](AudioGuard/reports/run.md) is software-test
evidence only, not ASVspoof benchmark evidence. See the [project README](AudioGuard/README.md)
for environment setup, data mapping, reproduction commands, limitations, and
the demo.

## Clean EVAL headline results

| Model | Pooled EER | ROC-AUC | Balanced accuracy |
| --- | ---: | ---: | ---: |
| MFCC SVM | 12.782465% | 0.938992 | 0.872889 |
| Log-Mel CNN | 8.456524% | 0.969237 | 0.884696 |

Observed failure cases: A17 had the highest attack EER (30.2221% CNN; 34.3542% SVM); on the Phase 4B subset, CNN spoof recall fell near zero under noise at its frozen threshold, while SVM decisions approached all-spoof at 10 and 0 dB. These claims and the precise operating-point values are traced in the [final research report](AudioGuard/reports/final_research_report.md).

## Contents and artifacts

Committed artifacts include the CNN `model.pt`, `scaler.npz`, and training
metadata needed for the saved CNN run. This repository does not include the
ASVspoof dataset, feature caches, per-clip score CSVs, local dataset configs,
or SVM model/scaler files. Small aggregate reports and metrics are included;
large local experiment outputs remain outside version control.

## Demo and video

Follow [the local demo instructions](AudioGuard/demo/README.md).

demo video: pending

The final demo was manually tested through a physical browser microphone
(owner-reported). Live human speech returned bona fide. A separate uploaded
audio sample containing ChatGPT-generated speech returned spoof. These are two
manual smoke tests (one sample each), not independently labelled benchmark
trials and not evidence of accuracy. Earlier microphone-preview issues occurred
during development. The automated fake-microphone browser regression test is
not part of the default test run.

## License status

N## Copyright and permissions

Copyright © 2026 [Shorya Pratap Singh Chandrawat]. All rights reserved.

No license is granted to use, modify, or redistribute this project. For
permission, contact [your contact information].

This notice does not override the permissions GitHub grants for viewing or
forking a public repository.
