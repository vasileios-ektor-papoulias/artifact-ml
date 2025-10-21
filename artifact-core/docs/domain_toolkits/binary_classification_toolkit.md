# Table Comparison Toolkit

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

`artifact-core` provides a concrete implementation for evaluation of binary classification results.

## Supported Artifacts

TBD

## Usage Example

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