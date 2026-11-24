# Experiment log

Use this file to record real experiments. Do not add results until you have run the corresponding command.

## Baseline protocol

- Dataset: record the exact source, version, and checksum.
- Split: stratified train/validation/test; fixed seed 42.
- Selection: highest validation average precision.
- Final evaluation: held-out test set, evaluated once after model selection.
- Threshold: 0.50 for the initial baseline.
- Models: class-weighted Logistic Regression and Random Forest.
- Metrics: average precision, ROC-AUC, precision, recall, F1 and confusion matrix.

## Results

| Run | Dataset/version | Model | Validation AP | Test AP | Test ROC-AUC | Notes |
|---|---|---|---:|---:|---:|---|
| Not run | — | — | — | — | — | Populate after execution |

## Questions to investigate

1. How does class weighting affect precision and recall?
2. How sensitive are results to the decision threshold?
3. Does a time-based split produce different results from a random stratified split?
4. How stable are the metrics across multiple random seeds?
5. Which errors are most costly in a real fraud-screening workflow?

## Interpretation checklist

- Report class prevalence alongside all metrics.
- Do not use accuracy as the sole metric.
- Distinguish validation results from final test results.
- State that synthetic-data results are only a pipeline demonstration.
- Explain false positives and false negatives in the context of the application.
