# Phase 2: Features and transparent baseline

## What was built
Validated mono audio reader, fixed log-spectral mean/std frontend, train-only standardizer, seeded logistic learner, CPU/CUDA auto selection, and persisted training history/model checksum.

## Why these decisions
Simple features and explicit gradients are defensible starting points. Whole-clip pooling avoids arbitrary cropping; fixed-size features keep the classifier inexpensive. Train-only normalization avoids leakage. Optional CUDA accelerates fitting without requiring GPU hardware.

## What to learn and explain
Derive binary cross-entropy plus L2 and the sigmoid gradient. Explain FFT resolution, log floors, constant feature handling, temporal pooling losses, imbalance, and why sigmoid scores are not calibrated probabilities.

## Smoke test (run before writing this note)
Source: `docs/validation.json`, copied from `results/smoke/status.json` by this script.
Scope: synthetic tone fixtures only, NOT ASVspoof speech or benchmark performance.
- `train`: PASS; see recorded stdout/stderr.
- `tests`: PASS; see recorded stdout/stderr.

## Honest limitations
Official LA data was unavailable. Real-file format, official speaker-disjointness, full-data training, unseen-attack effectiveness and real robustness remain unverified. CUDA, Windows and Colab execution remain untested in this sandbox.
The original missing-codec failure is in `docs/initial_smoke_failure.json`; do not delete failed attempts when conducting experiments.

## Viva questions
1. Why use a simple baseline before a deep network?
2. Which temporal cues does pooling discard?
3. Why is normalization fit only on train?
4. How does L2 affect the gradient?
5. Which operations actually run on CUDA?

## Research extension update
The research suite now extends this baseline; see `research_guide.md`, later phase notes and `research_validation.json`. Existing protocol/leakage checks and clean-dev threshold policy are retained. Synthetic validation does not establish official-data performance.
