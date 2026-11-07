# SYNTHETIC SMOKE-TEST REPORT — NOT ASVSPOOF BENCHMARK EVIDENCE

This file reports only a synthetic software smoke test. Its fixtures and every
metric below are synthetic; none of these values are ASVspoof results, real
speech results, or evidence of deepfake-detection accuracy. Real-data results
are reported separately in the Phase 2–5.1 reports linked from the READMEs.

Scope: results/smoke/run. Paths containing smoke refer to SYNTHETIC SOFTWARE TESTS, NOT ASVspoof.
Higher score = spoof; bonafide=0, spoof=1.
Operating threshold is selected on clean dev only. EER is empirical, not official toolkit EER.
No min-tDCF is claimed: ASV scores/costs and official implementation are not provided.

## Data scope
```json
{
  "full_counts": {
    "train": {
      "total": 8,
      "bonafide": 4,
      "spoof": 4,
      "speakers": 8,
      "attacks": {
        "-": 4,
        "A01": 4
      }
    },
    "dev": {
      "total": 8,
      "bonafide": 4,
      "spoof": 4,
      "speakers": 8,
      "attacks": {
        "-": 4,
        "A01": 4
      }
    },
    "eval": {
      "total": 8,
      "bonafide": 4,
      "spoof": 4,
      "speakers": 8,
      "attacks": {
        "-": 4,
        "A99": 4
      }
    }
  },
  "used_counts": {
    "train": {
      "total": 8,
      "bonafide": 4,
      "spoof": 4,
      "speakers": 8,
      "attacks": {
        "-": 4,
        "A01": 4
      }
    },
    "dev": {
      "total": 8,
      "bonafide": 4,
      "spoof": 4,
      "speakers": 8,
      "attacks": {
        "-": 4,
        "A01": 4
      }
    },
    "eval": {
      "total": 8,
      "bonafide": 4,
      "spoof": 4,
      "speakers": 8,
      "attacks": {
        "-": 4,
        "A99": 4
      }
    }
  },
  "subset_requested": false,
  "subset_selection": "all protocol trials",
  "checks": {
    "train_dev_speaker_overlap": [],
    "train_dev_utterance_overlap": [],
    "train_eval_speaker_overlap": [],
    "train_eval_utterance_overlap": [],
    "dev_eval_speaker_overlap": [],
    "dev_eval_utterance_overlap": []
  }
}
```
## Dev operating-point selection
```json
{
  "partition": "dev",
  "eer": 0.0,
  "threshold": 0.8875463613706606,
  "far": 0.0,
  "frr": 0.0,
  "method": "nearest finite empirical ROC; mean FAR/FRR"
}
```
## clean
```json
{
  "status": "completed",
  "metrics": {
    "n": 8,
    "bonafide": 4,
    "spoof": 4,
    "threshold": 0.8875463613706606,
    "tn": 4,
    "fp": 0,
    "fn": 0,
    "tp": 4,
    "far": 0.0,
    "frr": 0.0,
    "accuracy": 1.0,
    "roc_auc": 1.0,
    "diagnostic_eer": 0.0
  },
  "per_attack": {
    "A99": {
      "n": 4,
      "frr": 0.0,
      "seen_in_train": false
    }
  }
}
```
## mp3_32kbps
```json
{
  "status": "completed",
  "metrics": {
    "n": 8,
    "bonafide": 4,
    "spoof": 4,
    "threshold": 0.8875463613706606,
    "tn": 0,
    "fp": 4,
    "fn": 4,
    "tp": 0,
    "far": 1.0,
    "frr": 1.0,
    "accuracy": 0.0,
    "roc_auc": 0.0,
    "diagnostic_eer": 1.0
  },
  "per_attack": {
    "A99": {
      "n": 4,
      "frr": 1.0,
      "seen_in_train": false
    }
  }
}
```
## noise_10dB
```json
{
  "status": "completed",
  "metrics": {
    "n": 8,
    "bonafide": 4,
    "spoof": 4,
    "threshold": 0.8875463613706606,
    "tn": 0,
    "fp": 4,
    "fn": 4,
    "tp": 0,
    "far": 1.0,
    "frr": 1.0,
    "accuracy": 0.0,
    "roc_auc": 0.0,
    "diagnostic_eer": 1.0
  },
  "per_attack": {
    "A99": {
      "n": 4,
      "frr": 1.0,
      "seen_in_train": false
    }
  }
}
```
