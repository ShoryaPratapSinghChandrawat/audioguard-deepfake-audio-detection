# AudioGuard glossary

- **Bonafide:** genuine speech; the negative class in this workspace.
- **Spoof:** synthetic/manipulated speech; the positive class.
- **LA:** Logical Access; synthetic speech injected into an access channel, distinct
  from replay/Physical Access challenge data.
- **Protocol:** trial-level metadata associating speaker, utterance, attack ID and label.
- **Leakage:** unintended information sharing across training, selection and evaluation.
- **Dev partition:** the only partition used for operating thresholds, early stopping
  or future hyperparameter comparisons.
- **Eval partition:** held-out assessment after all decisions are frozen.
- **Log-Mel:** logarithm of spectral power projected through mel-spaced triangular filters.
- **MFCC:** cosine transform of log-Mel energies, retaining selected cepstral coefficients.
- **Delta / delta-delta:** local temporal regression derivative / repeated derivative.
- **Pooling:** summary over time; mean/std retain distributional but not sequence order cues.
- **SVM:** maximum-margin classifier; here a standardized nonlinear RBF decision function.
- **Logit:** unbounded signed CNN output before sigmoid; not calibrated probability.
- **pos_weight:** negative-to-positive train-count ratio in positive-class BCE terms.
- **AdamW:** adaptive gradient optimizer with decoupled weight decay.
- **Early stopping:** retain the checkpoint with best dev loss; never use eval for stopping.
- **Precision:** fraction of predicted spoof trials that are actually spoof.
- **Recall:** fraction of actual spoof trials detected as spoof.
- **F-score:** harmonic mean of precision and recall.
- **FAR / FPR here:** fraction of bonafide trials incorrectly flagged as spoof.
- **FRR / FNR here:** fraction of spoof trials incorrectly accepted as bonafide.
  These names must not be confused with an ASV system's acceptance orientation.
- **ROC-AUC:** threshold-independent ranking summary over both classes.
- **EER here:** empirical average of FPR/FNR at their nearest finite ROC crossing.
  It is not an exact interpolated official-toolkit EER; inspect the implementation.
- **Per-attack EER:** one spoof attack contrasted with the shared eval bonafide pool.
- **Unseen attack ID:** an eval protocol ID absent from train; not proof of a new method.
- **Synthesis reuse:** different IDs may use the same method; protocol IDs alone cannot prove it.
- **Shortcut:** nuisance cue correlated with labels, such as silence or recording duration.
- **dBFS:** amplitude relative to full scale; used here for fixed edge-silence detection.
- **SNR:** signal-to-noise power ratio in decibels; unrelated to classification accuracy.
- **MP3 round trip:** real lossy compression followed by decode, not bit-depth quantization.
- **Content-addressed cache:** disk artifact keyed by waveform bytes, frontend settings and version.
- **Synthetic fixture:** non-speech test input validating code, not a deepfake benchmark.
