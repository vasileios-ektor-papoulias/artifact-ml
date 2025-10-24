# Binary Classification Toolkit

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

The [`artifact-core`](../index.md) binary classification toolkit offers a comprehensive suite of artifacts for the evaluation of binary classification experiments.

## Usage Sketch

```python
from typing import Dict

import pandas as pd

from artifact_core.binary_classification import (
    BinaryClassificationEngine,
    BinaryClassificationPlotType,
    BinaryFeatureSpec
)

true: Dict[str, str] = df_classification_results["true"].to_dict()

predicted: Dict[str, str] = df_classification_results["predicted"].to_dict()

probs_pos: Dict[str, float] = df_classification_results["predicted_prob"].to_dict()

class_spec = BinaryFeatureSpec(
    ls_categories=["0", "1"],
    positive_category="1",
    feature_name="class"
)

engine = BinaryClassificationEngine(resource_spec=class_spec)

ground_truth_prob_pdf_plot = engine.produce_classification_plot(
    plot_type=BinaryClassificationPlotType.GROUND_TRUTH_PROB_PDF,
    true=true,
    predicted=predicted,
    probs_pos=probs_pos,
)

ground_truth_prob_pdf_plot
```

<p align="center">
  <img src="../../assets/ground_truth_distribution.png" width="600" alt="Ground Truth Distribution Artifact">
</p>

```python
roc_auc_plot = engine.produce_classification_plot(
    plot_type=BinaryClassificationPlotType.ROC_CURVE,
    true=true,
    predicted=predicted,
    probs_pos=probs_pos,
)

roc_auc_plot
```
<p align="center">
  <img src="../../assets/roc_plot.png" width="600" alt="ROC Artifact">
</p>

## Supported Artifacts

### Scores
- `ACCURACY` — Overall classification accuracy.
- `BALANCED_ACCURACY` — Mean of recall (TPR) and specificity (TNR).
- `PRECISION` — Positive predictive value (PPV).
- `NPV` — Negative predictive value.
- `RECALL` — True positive rate (TPR, sensitivity).
- `TNR` — True negative rate (specificity).
- `FPR` — False positive rate.
- `FNR` — False negative rate.
- `F1` — Harmonic mean of precision and recall.
- `MCC` — Matthews correlation coefficient.
- `ROC_AUC` — Area under the ROC curve.
- `PR_AUC` — Area under the Precision–Recall curve.
- `GROUND_TRUTH_PROB_MEAN` — Mean of ground-truth probabilities.
- `GROUND_TRUTH_PROB_STD` — Standard deviation of ground-truth probabilities.
- `GROUND_TRUTH_PROB_VARIANCE` — Variance of ground-truth probabilities.
- `GROUND_TRUTH_PROB_MEDIAN` — Median of ground-truth probabilities.
- `GROUND_TRUTH_PROB_FIRST_QUARTILE` — 25th percentile of ground-truth probabilities.
- `GROUND_TRUTH_PROB_THIRD_QUARTILE` — 75th percentile of ground-truth probabilities.
- `GROUND_TRUTH_PROB_MIN` — Minimum ground-truth probability.
- `GROUND_TRUTH_PROB_MAX` — Maximum ground-truth probability.

### Plots
- `CONFUSION_MATRIX_PLOT` — Visual confusion matrix.
- `ROC_CURVE` — Receiver Operating Characteristic curve.
- `PR_CURVE` — Precision–Recall curve.
- `DET_CURVE` — Detection Error Tradeoff curve.
- `RECALL_THRESHOLD_CURVE` — Recall as a function of decision threshold.
- `PRECISION_THRESHOLD_CURVE` — Precision as a function of decision threshold.
- `SCORE_PDF` — PDF of model scores.
- `GROUND_TRUTH_PROB_PDF` — PDF of ground-truth probabilities.

### Score Collections
- `NORMALIZED_CONFUSION_COUNTS` — TP/TN/FP/FN normalized counts across conditions.
- `BINARY_PREDICTION_SCORES` — Batched scalar metrics for predictions.
- `THRESHOLD_VARIATION_SCORES` — Metrics evaluated over threshold sweeps.
- `SCORE_STATS` — Summary stats for all scores.
- `POSITIVE_CLASS_SCORE_STATS` — Stats restricted to positive-class scores.
- `NEGATIVE_CLASS_SCORE_STATS` — Stats restricted to negative-class scores.
- `SCORE_MEANS` — Means of scores by grouping.
- `SCORE_STDS` — Standard deviations of scores by grouping.
- `SCORE_VARIANCES` — Variances of scores by grouping.
- `SCORE_MEDIANS` — Medians of scores by grouping.
- `SCORE_FIRST_QUARTILES` — 25th percentiles of scores by grouping.
- `SCORE_THIRD_QUARTILES` — 75th percentiles of scores by grouping.
- `SCORE_MINIMA` — Minimum scores by grouping.
- `SCORE_MAXIMA` — Maximum scores by grouping.
- `GROUND_TRUTH_PROB_STATS` — Summary stats for ground-truth probabilities.

### Arrays
- `CONFUSION_MATRIX` — Single confusion matrix array.

### Array Collections
- `CONFUSION_MATRICES` — Collection of confusion matrices (e.g., across thresholds or splits).

### Plot Collections
- `CONFUSION_MATRIX_PLOTS` — Set of confusion matrix visuals (e.g., per split).
- `THRESHOLD_VARIATION_CURVES` — Set of metric-vs-threshold curves.
- `SCORE_PDF_PLOTS` — Set of score PDF plots (e.g., per class or split).
