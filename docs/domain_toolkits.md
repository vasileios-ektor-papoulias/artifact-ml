# Domain Toolkits

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

In line with our [design philosophy](design_philosophy.md), [Artifact-ML](https://github.com/vasileios-ektor-papoulias/artifact-ml) is organized in **domain toolkits**: self-contained validation ecosystems aligned with specific application domains (e.g., tabular data synthesis, binary classification).

Domains are specified by their required validation resources.

For instance, tabular synthesizers are evaluated by comparing the characteristics of the real and synthetic datasets---and therefore this tuple characterizes the tabular data synthesis domain.

## Supported Domains

Currently, we provide toolkits for the following domains:

- tabular data synthesis,
- binary classification.

## Package Organization

Each [package](packages.md) in the ecosystem is organized accordingly---bundling together toolkits for supported domains.

In particular, packages expose their user-facing components by domain as top-level modules in: `<package_name>/<domain_toolkit_name>`.

For instance, to compute validation artifacts for a binary classification experiment do:

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
...
```

## Relevant Pages

- [Domain Toolkits: artifact-core ](https://artifact-ml.readthedocs.io/en/latest/artifact-core/domain_toolkits/)
    - [Table Comparison Toolkit](https://artifact-ml.readthedocs.io/en/latest/artifact-core/domain_toolkits/table_comparison/)
    - [Binary Classification Toolkit](https://artifact-ml.readthedocs.io/en/latest/artifact-core/domain_toolkits/binary_classification/)