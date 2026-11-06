import numpy as np

from scripts.extra_metrics import calculate_metrics


def test_always_spoof_classifier_has_half_balanced_accuracy_and_zero_bonafide_recall():
    labels = np.array([0, 0, 1, 1])
    scores = np.ones(4, dtype=float)

    result = calculate_metrics(labels, scores, threshold=0.5)

    assert result["balanced_accuracy"] == 0.5
    assert result["recall_by_class"]["bonafide"] == 0.0
    assert result["recall_by_class"]["spoof"] == 1.0
