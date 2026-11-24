import numpy as np
from fraudshield.modeling import build_models, evaluate_predictions


def test_build_models_returns_two_baselines():
    models = build_models()
    assert set(models) == {"logistic_regression", "random_forest"}


def test_evaluate_predictions_returns_expected_metrics():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    result = evaluate_predictions(y, probabilities)
    assert result["average_precision"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["recall"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
