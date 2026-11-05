# Phase 7: Ablation evaluation, shortcuts and error audits

## What was built
Implemented fixed-threshold ablation evaluation, diagnostic per-attack EER, per-speaker errors, ROC/confusion plots, ranked errors and silence/nuisance controls.

## Why
Eval thresholds cannot be retuned. Shared bonafide pools make attack-specific ROC meaningful. Nuisance baselines probe possible shortcuts without proving causation.

## What to learn and explain
Trace precision/recall orientation, explain null class-specific speaker rates and distinguish unseen attack IDs from unseen synthesis families. Never infer method reuse from IDs.

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
1. Which samples enter per-attack EER?
2. Why is a speaker's FAR sometimes null?
3. Can protocols prove synthesis reuse?
4. Why retain the clean-dev threshold after trimming?
5. How should high-confidence errors inform a held-out study?
