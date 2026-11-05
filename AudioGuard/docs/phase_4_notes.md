# Phase 4: Robustness, honest reporting and handoff

## What was built
Deterministic waveform-SNR noise, real MP3 encode/decode, condition-level failures, JSON-derived reports, offline demo, Colab notebook and checksum-manifest source ZIP packaging.

## Why these decisions
Noise and compression stress distinct distribution shifts. A fixed clean-dev threshold avoids optimistic condition-specific tuning. The initial missing-FFmpeg failure was preserved; a pinned executable-wheel fallback corrected that environment problem. No bad result is silently replaced.

## What to learn and explain
Derive signal/noise power scaling in decibels. Explain the effects of codec filtering, nonlinearities and delay. Distinguish synthetic software validation from speech benchmark evidence. Discuss real-noise limitations and paired clean/degraded evaluation at identical trial scope.

## Smoke test (run before writing this note)
Source: `docs/validation.json`, copied from `results/smoke/status.json` by this script.
Scope: synthetic tone fixtures only, NOT ASVspoof speech or benchmark performance.
- `robustness`: PASS; see recorded stdout/stderr.
- `report`: PASS; see recorded stdout/stderr.
- `tests`: PASS; see recorded stdout/stderr.

## Honest limitations
Official LA data was unavailable. Real-file format, official speaker-disjointness, full-data training, unseen-attack effectiveness and real robustness remain unverified. CUDA, Windows and Colab execution remain untested in this sandbox.
The original missing-codec failure is in `docs/initial_smoke_failure.json`; do not delete failed attempts when conducting experiments.

## Viva questions
1. Why retain the clean-dev threshold under noise?
2. Why is quantization not an MP3 experiment?
3. How are noise draws reproducible independent of ordering?
4. Why avoid clipping additive-noise audio?
5. What remains unproven after synthetic smoke tests pass?

## Research extension update
The research suite now extends this baseline; see `research_guide.md`, later phase notes and `research_validation.json`. Existing protocol/leakage checks and clean-dev threshold policy are retained. Synthetic validation does not establish official-data performance.
