# Defending the research extension

## Frontend conventions and cache

Read `configs/features.yaml` for the exact sample rate, FFT, hop, mel/cepstral
sizes and epsilon. The implementation uses a Hann window with no boundary extension;
short clips are zero-padded only to one window. SciPy's STFT spectrum normalization
is fixed; the mel bank uses HTK frequency spacing, unit-peak triangles and no Slaney
area normalization. The log is the natural logarithm of power plus epsilon, not a
clipped decibel transform. MFCC includes the zeroth cepstrum; DCT-II is orthonormal.
Deltas use centered linear-regression weights and replicated boundaries; applying
that operator again gives delta-delta. SVM pools all frames to mean/std vectors.

Audio must be mono at the configured rate: no implicit stereo reduction or resampling.
Cache keys include audio SHA-256, every frontend setting except its cache directory,
input kind and explicit frontend implementation version. Writes are atomic; corrupt
cached arrays fail validation. Bump FRONTEND_VERSION when changing conventions.
Neither normalization nor cache construction may fit using eval labels.

## CNN design

The exact channel widths, dropout, optimizer, crop length, seed and stopping settings
are YAML parameters. All convolution blocks preserve spatial dimensions before
pooling. Adaptive global averaging avoids a large dense head. MFCC input stacks
static/delta/delta-delta rows as one feature image, so feature-axis locality is an
assumption to critique, not an established equivalence to frequency locality.

Train-only per-feature frame statistics standardize both CNN inputs. All splits use
the same deterministic center crop to the configured frame length, padding shorter
clips with train-mean standardized values. This discards long-context information
and can expose length artifacts; the shortcut controls should be discussed alongside
any performance gain. The SVM uses whole-clip pooling, so its temporal budget differs;
this is a baseline comparison, not an architecture-only controlled experiment.

The balanced, unique train-only overfit subset is a debugging gate, not evidence of
generalization. Its membership, loss and accuracy history are saved. Dropout and
weight decay are disabled only for this diagnostic. Failed memorization blocks normal
training. Normal training starts from freshly seeded weights, uses dropout and
AdamW, and weights positive BCE by the actual negative/positive training-count ratio.
Dev loss selects the retained checkpoint; its state includes BatchNorm buffers.

## Thresholds and evidence

SVM/shortcut decision functions and CNN sigmoid scores all increase toward spoof.
Every model chooses its own operating threshold on clean dev after fitting. Eval
and every degraded condition use that model's unchanged threshold. Do not compare
absolute SVM and CNN scores as if they were calibrated probabilities.

EER uses the baseline finite empirical ROC convention, including its first-tie rule.
Reports distinguish diagnostic eval EER from deployed dev-threshold error rates.
Each per-attack ROC pairs the attack with the complete eval bonafide pool and logs
its class counts. Per-speaker FAR/FNR is null, not a fabricated number, when that
speaker lacks the relevant class. Top FP examples are sorted by descending spoof
score; top FN examples by ascending spoof score. These are descriptive error audits,
not a license to optimize using eval examples.

## Requested attack taxonomy: do not overclaim

The requested A07–A15 IDs are checked for eval presence and full-train absence before
being annotated as unseen IDs. The requested A16→A04 and A19→A06 mappings are
recorded as claims, not inferred facts. LA trial protocols contain no synthesis-method
column. Even if the parent and child IDs both appear where expected, that cannot
verify algorithm reuse. Without the official method metadata/reference, the output
states UNVERIFIED. No real protocol files were available in this workspace.

If authoritative mapping metadata is obtained, supply its extracted CSV via
`attack_reuse_evidence` with columns `attack_id,reuses_training_attack,source`.
A source reference is mandatory; the code cross-checks ID membership and preserves
that source in results. Status says supported by supplied metadata, not independently
authenticated. Never manufacture a CSV merely to satisfy an expected taxonomy.

## Shortcut and robustness controls

A separate train-only standardized logistic classifier receives duration, leading
edge-silence duration and waveform RMS only. Silence detection uses an absolute,
predeclared amplitude threshold. The same models are evaluated after trimming both
edges using the fixed threshold, without retraining or choosing a new operating
point. All-silent waveforms retain one sample to keep feature computation defined.
These controls may indicate nuisance exploitation but do not establish causality.

Noise power is scaled per waveform to the requested SNR. No clipping is applied,
because it would confound noise with nonlinear distortion. MP3 uses real FFmpeg
encode/decode; version and condition errors are logged. The current tests do not
represent real rooms, background speakers, codecs beyond those configured, or
adversarially optimized attacks.

## Reproducible execution and limits

Use a fresh output for every suite. Train all ablations before opening eval outputs.
Freeze configs, thresholds and checkpoints; retain failed attempts and raw-score CSVs.
`verify_research.py` recomputes metrics and diagnostic arrays from those CSVs,
checks dev-selected thresholds, loads saved models/scalers and builds the demo.
Curves and reports are generated from JSON, not numbers copied into prose.

The synthetic suite exercises all code paths but says nothing about authentic versus
synthetic human speech. Real corpus metrics, full-dataset resource usage and CUDA,
Windows/Colab execution remain unverified. Gradio shows saved result scope and errors;
its upload prediction is not a confidence-calibrated security verdict. Keep demo
instances private and do not load untrusted joblib checkpoints.
