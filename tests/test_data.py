import pandas as pd
import pytest

from fraudshield.data import load_transactions, validate_feature_frame


def test_load_transactions_valid_csv(tmp_path):
    path = tmp_path / "transactions.csv"
    pd.DataFrame({"amount": [1, 2, 3, 4], "Class": [0, 0, 1, 1]}).to_csv(path, index=False)
    X, y = load_transactions(path)
    assert list(X.columns) == ["amount"]
    assert y.tolist() == [0, 0, 1, 1]


def test_load_transactions_requires_target(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"amount": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Class"):
        load_transactions(path)


def test_validate_feature_frame_orders_columns():
    frame = pd.DataFrame({"b": [2], "a": [1]})
    aligned = validate_feature_frame(frame, ["a", "b"])
    assert list(aligned.columns) == ["a", "b"]
