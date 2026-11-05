# Phase 3: Dev selection and held-out evaluation

## What was built
Spoof-positive metrics, empirical EER selection on dev, frozen eval threshold, raw score CSVs with audio hashes, per-attack train-membership flags, and CSV-to-JSON metric verification.

## Why these decisions
Eval operating-point selection leaks evaluation labels into deployment decisions. A fixed clean-dev threshold shows threshold transfer honestly. Per-attack miss rates expose unseen-attack weaknesses without invalid one-class ROC calculations.

## What to learn and explain
Trace score >= threshold through every confusion-matrix cell. FAR here means bonafide flagged as spoof and FRR means spoof missed, not an unstated ASV acceptance convention. Understand the finite empirical EER approximation and why it differs from official interpolated tooling.

## Smoke test (run before writing this note)
Source: `docs/validation.json`, copied from `results/smoke/status.json` by this script.
Scope: synthetic tone fixtures only, NOT ASVspoof speech or benchmark performance.
- `evaluate`: PASS; see recorded stdout/stderr.
- `score_verification`: PASS; see recorded stdout/stderr.
- `tests`: PASS; see recorded stdout/stderr.

## Honest limitations
Official LA data was unavailable. Real-file format, official speaker-disjointness, full-data training, unseen-attack effectiveness and real robustness remain unverified. CUDA, Windows and Colab execution remain untested in this sandbox.
The original missing-codec failure is in `docs/initial_smoke_failure.json`; do not delete failed attempts when conducting experiments.

## Viva questions
1. Why can eval EER be reported but not deployed as the threshold?
2. What changes if spoof becomes label zero?
3. Why can one attack not have standalone ROC-AUC?
4. What makes an attack unseen in this code?
5. Why is accuracy insufficient under imbalance?

## Research extension update
The research suite now extends this baseline; see `research_guide.md`, later phase notes and `research_validation.json`. Existing protocol/leakage checks and clean-dev threshold policy are retained. Synthetic validation does not establish official-data performance.
