# AudioGuard

AudioGuard evaluates spoofed-speech countermeasures on the Logical Access (LA)
track of ASVspoof 2019. The repository contains full-partition clean-evaluation
reports for an MFCC + SVM and a Log-Mel CNN, plus saved-score analysis,
fixed-subset robustness results, and resampling diagnostics. The work compares
these two complete pipelines; it is not an architecture-only ablation.

## Dataset

The official ASVspoof 2019 LA dataset is not redistributed here. Obtain the
audio and protocols from the official [ASVspoof 2019 page](https://www.asvspoof.org/index2019.html)
or [University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336).
The dataset's own terms apply. No license file is included with this project;
the repository is submitted for evaluation.

The experiments used official LA train, development, and evaluation partitions.
The saved audit reports 25,380 train, 24,844 development, and 71,237 evaluation
trials, with no malformed protocol rows, missing protocol audio, speaker
overlap, or utterance overlap. Target 0 is bona fide; target 1 is spoof.

The extracted dataset root should contain the official partition audio folders
and the `ASVspoof2019_LA_cm_protocols` folder, including the trial protocol
files. Do not move or rename the distributed data to match an assumed layout.
First run `inspect` and use its actual paths and file names to fill in configs.

## Results

| Phase | Report | What it contains |
| --- | --- | --- |
| 2 | [MFCC + SVM](results/real_svm/experiment_summary.md) | Full train/dev/eval audit, clean metrics, attacks, frozen DEV threshold |
| 2.5 | [Independent SVM verification](results/real_svm/phase2_addendum.md) | CSV validation, independent pooled and attack metrics, DEV/EVAL comparisons |
| 3 | [Log-Mel CNN](results/real_cnn_logmel/phase3_summary.md) | CNN configuration, one-seed training, clean metrics, attack comparison |
| 4A | [Saved-score analysis](results/phase4a/phase4a_summary.md) | Speaker error spread, attack groups, duration correlations, nuisance-cue baseline |
| 4B | [Verified fixed-subset results](results/phase4b/optimized_results/phase4b_summary_verified.md) | **SUBSET RESULTS** on the fixed 10,000-clip subset: noise, MP3, and trimming |
| 5.1 | [Resampling sensitivity](results/phase5_diagnostics/summary.md) | Small sample of simulated resampling paths; raw-logit comparisons |

All reported thresholds were selected using clean DEV data and then frozen for
EVAL and perturbed conditions. Phase 4B uses only its preselected 10,000-clip
subset; do not describe those results as full-EVAL robustness. EER is a ranking
diagnostic and does not replace metrics at the frozen decision threshold.

## Environment setup (Windows)

Use Python 3.11 and the pinned requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

From the `AudioGuard` folder, create `.venv` with `py -3.11 -m venv .venv`
before running these commands.

`requirements.txt` pins `torch==2.5.1`, `gradio==6.31.0`, and
`huggingface-hub==1.16.4`. The plain Torch pin does not select a CUDA wheel.
For GPU training, install the compatible CUDA-enabled PyTorch 2.5.1 wheel for
the machine using the official PyTorch installation instructions; confirm
`torch.cuda.is_available()` before training. The saved Phase 3 run used
PyTorch 2.5.1+cu121 on an NVIDIA GeForce RTX 2050.

### Test environments

This final submission pass ran `python -m pytest -q` in two environments from
the Python project root. The existing project environment used Python 3.11.9,
Gradio 5.49.1, and `huggingface-hub` 0.36.0: **48 passed, 1 skipped, 14
warnings**. A clean venv installed from `requirements.txt` used Python 3.11.9,
Gradio 6.31.0, `huggingface-hub` 1.16.4, and Torch 2.5.1+cpu: **48 passed,
1 skipped, 16 warnings**. The clean venv matches the pinned requirements; the
existing environment does not. The skipped test is the optional browser
microphone test, which runs only when `AUDIOGUARD_DEMO_URL` points to a running
demo. Warnings were third-party deprecations.

## Local dataset configuration

Do not commit machine-specific paths. Copy `configs/baseline.yaml` to the
git-ignored `configs/baseline.local.yaml`; copy `configs/research.yaml` to
`configs/research.local.yaml` if running research commands. Set
`dataset.root`, each `dataset.protocols` entry, and each `dataset.audio_roots`
entry using the actual directory layout found by `inspect`:

```powershell
.\.venv\Scripts\python.exe scripts\audioguard.py inspect --config configs\baseline.local.yaml
.\.venv\Scripts\python.exe scripts\audioguard.py audit --config configs\baseline.local.yaml
```

Use forward slashes in absolute paths if preferred, for example
`C:/datasets/LA`. Map the protocols and audio roots from inspection output;
do not infer or relocate dataset files. The audit rejects speaker and utterance
overlap and missing audio before model fitting.

## Reproduction commands

Run commands from the `AudioGuard` Python project root with the local configs
and required dataset available. Training is not required to inspect the saved
reports or use the saved CNN demo.

```powershell
# Reproduce the clean SVM training and evaluation pipeline
.\.venv\Scripts\python.exe scripts\experiment.py train --config configs\research.local.yaml --model svm_mfcc
.\.venv\Scripts\python.exe scripts\clean_evaluate.py --config configs\research.local.yaml --model svm_mfcc
.\.venv\Scripts\python.exe scripts\extra_metrics.py --config configs\research.local.yaml --model svm_mfcc

# Train and clean-evaluate the Log-Mel CNN with the configured CUDA-capable Torch
.\.venv\Scripts\python.exe scripts\experiment.py train --config configs\research.local.yaml --model cnn_logmel
.\.venv\Scripts\python.exe scripts\clean_evaluate.py --config configs\research.local.yaml --model cnn_logmel

# Verify saved research artifacts
.\.venv\Scripts\python.exe scripts\verify_research.py --config configs\research.local.yaml
.\.venv\Scripts\python.exe -m pytest -q
```

Use a fresh local config/output directory when reproducing training; the listed
commands can create experiment outputs. Preserve the reported train/dev-only
selection policy. Do not use EVAL to tune models or thresholds. The checked-in
clean results and Phase 4B subset report are already saved; rerunning training
or robustness experiments is not necessary for review.

## Demo

Install the pinned requirements and prepare the ignored `configs/baseline.local.yaml`
as described above. Launch the saved CNN and optional locally available SVM
baseline with:

```powershell
.\.venv\Scripts\python.exe demo\app.py --run results\real_cnn_logmel\cnn_logmel --dataset-config configs\baseline.local.yaml --port 7860
```

The demo binds to `127.0.0.1` with sharing disabled. It supports WAV, FLAC,
MP3, WebM, and Opus uploads, and browser microphone input. Limits and
preprocessing are documented in [demo/README.md](demo/README.md).

The final demo was manually tested through a physical browser microphone
(owner-reported). Live human speech returned bona fide. A separate uploaded
audio sample containing ChatGPT-generated speech returned spoof. These are two
manual smoke tests (one sample each), not independently labelled benchmark
trials and not evidence of accuracy. Earlier microphone-preview issues occurred
during development. The automated fake-microphone browser regression test is
not part of the default test run.

demo video: pending

## What is included

Committed model artifacts are the CNN `model.pt`, `scaler.npz`, and its
training metadata. The repository does not redistribute the dataset, feature
caches, per-clip score CSVs, ignored local configs, or the SVM model/scaler
files. Small aggregate metrics, plots, and reports are committed. Do not treat
per-clip local score files as repository artifacts.

## Repository structure

```text
AudioGuard/
  configs/       portable YAML defaults; local configs are ignored
  src/           data loading, feature extraction, models, evaluation
  scripts/       inspection, audit, experiment, verification, packaging
  tests/         unit and regression tests (mostly fixture-based)
  demo/          local Gradio inference app and instructions
  docs/          research guide and glossary
  reports/       smoke-test report and report figures
  results/       committed aggregate reports and selected artifacts
  data/          local dataset and feature caches (not redistributed)
```

## Limitations

- The clean benchmark is one run/seed on ASVspoof 2019 LA. It does not
  establish performance on other datasets, languages, codecs, or future
  generators.
- The SVM and CNN differ in features, pooling, crop behavior, optimizer, and
  model; the comparison is not an architecture-only ablation.
- Phase 4B measures a fixed 10,000-clip subset with one seed and selected
  synthetic noise, MP3, and trimming conditions. Masking or simulated
  perturbations are not equivalent to natural recording conditions.
- Phase 5.1 examines a small diagnostic set and several resampling paths; its
  sensitivity is not a universal effect.
- Phase 4A shortcut and duration analyses show dataset-level predictive
  information and score associations. They do not prove that either detector
  relies on those cues.
- Official protocol IDs alone do not identify synthesis algorithms. Algorithm
  reuse statements rely on the cited ASVspoof database paper.
- Results use spoof as the positive class, while the official challenge
  convention treats bona fide as positive. Comparisons to published numbers
  are context only unless evaluation conventions and pipelines match.
- The Phase 4B noise, MP3, and trimming files are **SUBSET RESULTS**, not
  full-EVAL estimates. No claim of universal deepfake detection is made.
- The demo is an exploratory demonstration, not a forensic or production
  security tool. Predictions for individual uploaded clips are not validated
  benchmark trials.
  

## License status

## Copyright and permissions

Copyright © 2026 [SHorya Pratap Singh Chandrawat]. All rights reserved.

No license is granted to use, modify, or redistribute this project. For
permission, contact [your contact information].

This notice does not override the permissions GitHub grants for viewing or
forking a public repository.
