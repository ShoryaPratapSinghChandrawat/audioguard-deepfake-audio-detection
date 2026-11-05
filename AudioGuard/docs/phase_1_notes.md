# Phase 1: Workspace and official-data audit

## What was built
Isolated workspace, pinned dependencies, explicit YAML paths, real-file inspector, validated protocol parser, and full-protocol speaker/utterance-disjointness checks.

## Why these decisions
The existing repository is unrelated and must remain intact. Inspection prevents guessed paths and formats. Official partitions preserve the challenge generalization setting. Hashes and exact counts reveal data drift and subsets.

## What to learn and explain
Read the protocol parser and inspect your real protocol lines. Explain speaker leakage versus utterance leakage, metadata-only auditing versus model fitting, and why a sorted-prefix debugging subset is not representative.

## Smoke test (run before writing this note)
Source: `docs/validation.json`, copied from `results/smoke/status.json` by this script.
Scope: synthetic tone fixtures only, NOT ASVspoof speech or benchmark performance.
- `inspect`: PASS; see recorded stdout/stderr.
- `audit`: PASS; see recorded stdout/stderr.
- `tests`: PASS; see recorded stdout/stderr.

## Honest limitations
Official LA data was unavailable. Real-file format, official speaker-disjointness, full-data training, unseen-attack effectiveness and real robustness remain unverified. CUDA, Windows and Colab execution remain untested in this sandbox.
The original missing-codec failure is in `docs/initial_smoke_failure.json`; do not delete failed attempts when conducting experiments.

## Viva questions
1. Why not randomly split all utterances?
2. What does speaker-disjointness prevent?
3. Why inspect before parsing?
4. How can you prove which trials were used?
5. Why are protocol hashes not proof of official provenance?

## Research extension update
The research suite now extends this baseline; see `research_guide.md`, later phase notes and `research_validation.json`. Existing protocol/leakage checks and clean-dev threshold policy are retained. Synthetic validation does not establish official-data performance.
