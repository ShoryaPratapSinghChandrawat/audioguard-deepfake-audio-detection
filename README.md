# audioguard-deepfake-audio-detection

Research workspace for **AudioGuard**: reproducible ASVspoof Logical Access (LA)
deepfake-audio detection research — can detectors trained on known synthesis attacks
recognize genuine and spoofed speech, including attacks absent from training, and
withstand noise and lossy compression?

> **Research status: no benchmark evidence yet.**
> The official ASVspoof LA dataset has **not** been supplied to this repository.
> Every number produced so far comes from **synthetic tone fixtures** and validates
> software wiring only. Nothing here is a real-speech deepfake-detection result, and
> no recruitment-project conclusion is claimed.

## Layout

The project lives in [`AudioGuard/`](AudioGuard/README.md), preserving its original
workspace structure. Read [`AudioGuard/README.md`](AudioGuard/README.md),
[`AudioGuard/docs/glossary.md`](AudioGuard/docs/glossary.md),
[`AudioGuard/docs/research_guide.md`](AudioGuard/docs/research_guide.md) and the phase
notes before defending or extending the work.

```text
AudioGuard/
  configs/            # YAML hyperparameters, fixed seeds, synthetic smoke configs
  src/
    data/             # protocol parsing, partition loading, leakage checks
    features/         # log-Mel, MFCC (+ delta/delta-delta), disk caching
    models/           # SVM+MFCC, CNN, shortcut classifier
    evaluation/       # metrics, per-attack/per-speaker analysis, plots
    robustness/       # silence trim, noise, MP3 transforms, shortcut probes
  scripts/            # audioguard.py, experiment.py, smoke.py, research_smoke.py,
                      # verify_results.py, verify_research.py, package.py, ...
  tests/              # pytest suite (synthetic fixtures)
  demo/               # local Gradio demo (displays saved metrics only)
  notebooks/          # Colab quickstart
  docs/               # glossary, research guide, phase notes, saved validation JSON
  data/ models/ results/ reports/figures/   # gitignored working dirs
```

`src/data/` and `src/models/` are present and populated.

## Provenance

Imported verbatim from a source archive, not rebuilt from scratch:

- Archive: `AudioGuard-source.zip`
- Source repo: `ShoryaPratapSinghChandrawat/qff2026-phase1-s1-ising_chain`
- Commit: `a18a7626cfa5e243deb441116cb4c6a388af28d1`
- Git blob SHA-1 of the archive: `7b86e3bb113f5f72f18afb4c6a63a9a5c5028de0` (matches GitHub)
- Archive SHA-256: `411b13597d2eb6f2ba14c59143c186a520afbe2e87a6238554d7dbfeb350e11c`
- Integrity: all **67/67** files in `AudioGuard/MANIFEST.sha256.json` verified with zero
  mismatches and no unmanifested files.

The archive is not committed (it is an artifact, and matches the project's `*.zip`
ignore rule). Rebuild a package with `python scripts/package.py` and re-verify with
`python scripts/verify_archive.py`.

## Install

Dependencies are pinned. A virtual environment is required (PEP 668 blocks
system-wide installs on many platforms) and `.venv/` is gitignored.

```bash
cd AudioGuard
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
python -m pip install -r requirements.txt   # or requirements-core.txt for the NumPy baseline only
```

## Verification actually run here

Executed in this sandbox: Linux, Python 3.11.2, `numpy==1.26.4`,
`scikit-learn==1.5.2`, `torch==2.5.1` (CUDA **not** available; CPU only),
`gradio==5.49.1`.

| Check | Command | Result |
| --- | --- | --- |
| Test suite | `python -m pytest -q` | **PASS** — 23 passed |
| Research smoke (end-to-end) | `python scripts/research_smoke.py` | **PASS** — 3/3 stages |
| `verify_research.py` | (inside research smoke) | **PASS** — metrics, plots, thresholds, checkpoints, demo verified for `svm_mfcc`, `cnn_logmel`, `cnn_mfcc`, `shortcut` |
| Baseline smoke | `python scripts/smoke.py` | **PASS** — inspect, audit, train, evaluate, robustness, report all exit 0 |

Scope of that evidence, quoted from the run output:
`SYNTHETIC TONE SOFTWARE TEST ONLY; NOT ASVSPOOF`. Fixtures were 64 train / 16 dev /
16 eval synthetic tones. Sample counts, protocol hashes and
`subset_selection: "all protocol trials"` were logged, not assumed.

One honest note on process: the **first** research-smoke attempt failed at
`verify_research.py` with `ModuleNotFoundError: No module named 'gradio'`. That was an
incomplete environment, not a code defect; after installing the pinned `gradio==5.49.1`
the suite passed. That failure record remains in
`AudioGuard/docs/initial_smoke_failure.json` and the earlier archived attempts.

## Research rules in force (unchanged)

These are preserved exactly as implemented; this import did not relax them.

- **No invented metrics.** Every reported number must come from a rerunnable script
  that writes it to disk. There are no hand-entered benchmark values.
- **Official partitions only.** Use the official ASVspoof LA train/dev/eval splits.
  No random re-splitting; no PA protocols.
- **Leakage checks preserved.** `src/data/protocol.py` raises on speaker or utterance
  overlap between partitions *before* any model fitting.
- **Dev-only selection.** Operating thresholds and CNN checkpoints are chosen on
  clean train/dev only and frozen for every eval condition. Eval EER is a ranking
  diagnostic, never a new threshold.
- **Label convention.** Bonafide = 0, spoof = 1; higher scores mean spoof.
- **Explicit scope.** Exact sample counts and subset identity are logged; a
  sorted-prefix subset is debugging only.
- **Inspect real data first.** Do not assume paths or protocol formats — run
  `python scripts/audioguard.py inspect`.
- **Config discipline.** Hyperparameters live in YAML, seeds are fixed, dependencies
  pinned.
- **No artifacts in Git.** Datasets, feature caches, virtual environments and model
  weights are excluded.
- **Failures are kept**, and synthetic tests stay clearly separated from benchmark
  evidence.

## Known limitations

- The official dataset is absent; no real ASVspoof LA performance has been measured.
- Existing validation uses synthetic tone fixtures only, so it says nothing about
  real speech generalisation, unseen-attack robustness or speaker effects.
- Attack-ID membership can be checked against protocols, but protocol files do not
  describe synthesis algorithms — synthesis-reuse claims **cannot** be verified from
  trial protocols alone.
- CUDA execution, Windows and Colab workflows, and full-dataset resource usage remain
  **unverified** (only Linux/CPU was exercised here).
- The demo is not a forensic or production security tool. Do not load untrusted
  pickle/joblib checkpoints.

## Next step

Map the real official partitions and run the audit before any modelling:

```bash
cd AudioGuard
python scripts/audioguard.py inspect   # print actual files/protocol previews
python scripts/audioguard.py audit     # validates fields, rejects leakage
python scripts/experiment.py suite --config configs/research.yaml
python scripts/verify_research.py --config configs/research.yaml
```

## AI-assistance disclosure

This project was built with AI assistance and imported into this repository by an AI
coding agent. Follow your lab's disclosure requirements: do not falsify authorship,
provenance or evidence, and keep the synthetic-vs-benchmark distinction explicit in
any write-up.
