# Phase 8: Results-backed demo and source ZIP handoff

## What was built
Implemented a Gradio upload demo, JSON-backed evidence panel, limitations notice, glossary, research guide and integrity-checked complete source ZIP.

## Why
The demo must expose scope and unknown performance instead of hardcoded scores. The ZIP packages real files while excluding datasets, environments and weights.

## What to learn and explain
Distinguish score from calibrated confidence, inspect saved provenance and understand why a UI patch-export button cannot be changed by repository files.

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
1. Where do displayed metrics originate?
2. What if eval results are missing?
3. Why not load an unknown joblib artifact?
4. Which files belong outside the source ZIP?
5. What remains unknown after synthetic tests pass?
