# Phase 6: CNN fitting and train-only sanity gate

## What was built
Implemented configurable convolution blocks, weighted BCE, AdamW, dev-loss early stopping, restored checkpoints and train/dev curves.

## Why
The train-only memorization gate finds plumbing failures before normal fitting; its weights are discarded. Train-only normalization and dev-only selection avoid leakage.

## What to learn and explain
Derive weighted BCE, distinguish regularization in sanity versus real fitting, and explain global pooling, BatchNorm buffers and center-crop limitations.

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
1. Why must the sanity subset exclude dev?
2. How is pos_weight calculated?
3. Why restart after memorization?
4. Which checkpoint is retained?
5. What confounds a CNN-versus-SVM comparison?
