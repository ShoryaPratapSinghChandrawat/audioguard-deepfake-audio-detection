# Phase 5: Cached log-Mel and MFCC frontends

## What was built
Implemented configured HTK log-Mel, DCT MFCC, temporal regression deltas, pooling and atomic disk caches.

## Why
A shared frontend enables controlled input ablation; content-addressing prevents stale settings or changed audio from reusing old features.

## What to learn and explain
Explain STFT normalization, log-plus-epsilon, mel spacing, DCT and derivative boundaries. Contrast frame matrices with pooled vectors.

## Smoke test before this note
Scope: SYNTHETIC TONE SOFTWARE TEST ONLY; NOT ASVSPOOF

- `-m pytest -q`: PASS.

```text
.......................                                                  [100%]
23 passed in 3.02s
```

- `scripts/experiment.py suite --config configs/research_smoke.yaml`: PASS.
- `scripts/verify_research.py --config configs/research_smoke.yaml`: PASS.
Evidence source: `docs/research_validation.json`. No official-data metrics are available.

## Honest limitations
Only synthetic tone fixtures were executed here. Official-data effectiveness, taxonomy and full-data resource usage remain unverified. CUDA, Windows and Colab were not exercised.

## Viva questions
1. Why include waveform bytes in a cache key?
2. How is log-plus-epsilon different from log flooring?
3. Why retain delta and delta-delta?
4. What does pooling lose?
5. Why record mel-filter normalization?
