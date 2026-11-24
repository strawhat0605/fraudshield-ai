"""Create a deterministic synthetic, imbalanced transaction dataset."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate(output: str = "data/processed/demo_transactions.csv", rows: int = 12000, seed: int = 42):
    if rows < 200:
        raise ValueError("Use at least 200 rows to create a useful demo dataset.")
    X, y = make_classification(
        n_samples=rows,
        n_features=30,
        n_informative=10,
        n_redundant=8,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.985, 0.015],
        class_sep=1.0,
        flip_y=0.01,
        random_state=seed,
    )
    columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    frame = pd.DataFrame(X, columns=columns)
    frame["Time"] = np.linspace(0, 172800, rows) % 172800
    frame["Amount"] = np.abs(frame["Amount"]) * 120 + 1
    frame["Class"] = y.astype(int)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {len(frame):,} synthetic rows to {path}")
    print(f"Fraud prevalence: {frame['Class'].mean():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/processed/demo_transactions.csv")
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.rows, args.seed)
