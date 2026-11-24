"""Train, select, evaluate, and persist fraud-detection baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraudshield.data import load_transactions
from fraudshield.modeling import build_models, evaluate_predictions

RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts")


def train(data_path: str | Path, artifact_dir: str | Path = ARTIFACT_DIR) -> dict:
    X, y = load_transactions(data_path)
    if y.value_counts().min() < 4:
        raise ValueError("Need at least 4 examples in each class for train/validation/test splits.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.20,
        stratify=y_train_val, random_state=RANDOM_STATE
    )

    candidates = build_models(RANDOM_STATE)
    validation_results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_val)[:, 1]
        validation_results[name] = evaluate_predictions(y_val, probabilities)

    selected_name = max(
        validation_results,
        key=lambda name: validation_results[name]["average_precision"]
    )

    final_model = build_models(RANDOM_STATE)[selected_name]
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    final_model.fit(X_train_full, y_train_full)
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_probabilities)

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output / "fraudshield_model.joblib")

    metadata = {
        "model_name": selected_name,
        "feature_columns": list(X.columns),
        "target_column": "Class",
        "threshold": 0.5,
        "random_state": RANDOM_STATE,
        "data_path": str(data_path),
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "warning": "Educational prototype. Not for real financial decisions.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(validation_results).T.to_csv(output / "validation_metrics.csv")
    pd.DataFrame([test_metrics]).drop(columns=["confusion_matrix"]).to_csv(
        output / "test_metrics.csv", index=False
    )

    cm = test_metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)
    ax.set_title("Held-out test confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a CSV with binary 'Class' target.")
    parser.add_argument("--artifacts", default=str(ARTIFACT_DIR), help="Output directory.")
    args = parser.parse_args()
    try:
        result = train(args.data, args.artifacts)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Selected model: {result['model_name']}")
    print("Held-out test metrics:")
    for key, value in result["test_metrics"].items():
        if key != "confusion_matrix":
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
