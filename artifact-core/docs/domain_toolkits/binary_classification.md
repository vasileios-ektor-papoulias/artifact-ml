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

score_pdf_plot = engine.produce_classification_plot(
    plot_type=BinaryClassificationPlotType.SCORE_PDF,
    true=true,
    predicted=predicted,
    probs_pos=probs_pos,
)

score_pdf_plot
```

<p align="center">
  <img src="../../assets/score_distribution.png" width="600" alt="Score Distribution Artifact">
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

> **Terminology Note** 
> 
> We refer to the model’s *predicted positive probabilities* — the continuous output values representing its confidence before thresholding — as *scores*.
>
> These are **not to be confused with metric scores**, which are scalar evaluation metrics such as accuracy, precision, recall, or ROC_AUC.
>
> Further, we use the term *predicted ground-truth probabilities* to refer to the probabilities the model assigns to the *true (ground-truth)* class for each sample.

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
- `GROUND_TRUTH_PROB_MEAN` — Mean of predicted ground-truth probabilities.  
- `GROUND_TRUTH_PROB_STD` — Standard deviation of predicted ground-truth probabilities.  
- `GROUND_TRUTH_PROB_VARIANCE` — Variance of predicted ground-truth probabilities.  
- `GROUND_TRUTH_PROB_MEDIAN` — Median of predicted ground-truth probabilities.  
- `GROUND_TRUTH_PROB_FIRST_QUARTILE` — 25th percentile of predicted ground-truth probabilities.  
- `GROUND_TRUTH_PROB_THIRD_QUARTILE` — 75th percentile of predicted ground-truth probabilities.  
- `GROUND_TRUTH_PROB_MIN` — Minimum predicted ground-truth probability.  
- `GROUND_TRUTH_PROB_MAX` — Maximum predicted ground-truth probability.

### Arrays

- `CONFUSION_MATRIX` — Single confusion matrix array of TP, TN, FP, and FN counts.

### Plots

- `CONFUSION_MATRIX_PLOT` — Visual confusion matrix.  
- `ROC_CURVE` — Receiver Operating Characteristic curve.  
- `PR_CURVE` — Precision–Recall curve.  
- `DET_CURVE` — Detection Error Tradeoff curve.  
- `RECALL_THRESHOLD_CURVE` — Recall as a function of decision threshold.  
- `PRECISION_THRESHOLD_CURVE` — Precision as a function of decision threshold.  
- `SCORE_PDF` — Probability density function (PDF) of scores (predicted positive probabilities).  
- `GROUND_TRUTH_PROB_PDF` — PDF of predicted ground-truth probabilities.

### Score Collections

- `NORMALIZED_CONFUSION_COUNTS` — normalized TP/TN/FP/FN counts.  
- `BINARY_PREDICTION_SCORES` — Batched scalar metrics (e.g., precision, recall).  
- `THRESHOLD_VARIATION_SCORES` — Metrics evaluated across decision threshold sweeps (e.g., PR AUC, ROC AUC etc.).  
- `SCORE_STATS` — Score (predicted positive probability) summary statistics.
- `POSITIVE_CLASS_SCORE_STATS` — Score (predicted positive probability) summary statistics restricted to the positive class.  
- `NEGATIVE_CLASS_SCORE_STATS` — Score (predicted positive probability) summary statistics restricted to the negative class.  
- `SCORE_MEANS` — Mean predicted score (positive probability) by split (all, positive, negative).  
- `SCORE_STDS` — Standard deviation of scores (predicted positive probabilities) by split (all, positive, negative).  
- `SCORE_VARIANCES` — Variance of scores (predicted positive probabilities) by split (all, positive, negative).  
- `SCORE_MEDIANS` — Median score (predicted positive probability) by split (all, positive, negative).  
- `SCORE_FIRST_QUARTILES` — 25th percentile of scores (predicted positive probabilities) by split (all, positive, negative).  
- `SCORE_THIRD_QUARTILES` — 75th percentile of scores (predicted positive probabilities) by split (all, positive, negative).  
- `SCORE_MINIMA` — Minimum score (predicted positive probability) by split (all, positive, negative).  
- `SCORE_MAXIMA` — Maximum score (predicted positive probability) by split (all, positive, negative).  
- `GROUND_TRUTH_PROB_STATS` — Summary statistics for predicted ground-truth probabilities.

### Array Collections

- `CONFUSION_MATRICES` — Collection of confusion matrices across normalization modes (`none`, `true`, `predicted`, `all`).

### Plot Collections

- `CONFUSION_MATRIX_PLOTS` — Set of confusion matrix visuals across normalization modes (`none`, `true`, `predicted`, `all`).  
- `THRESHOLD_VARIATION_CURVES` — Metric-vs-threshold curve set (e.g., ROC, PR, DET).  
- `SCORE_PDF_PLOTS` — Score (predicted positive probability) density plots across splits (all, positive, negative).
