# Domain Toolkits

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

In line with our [design philosophy](design_philosophy.md), Artifact-ML is organized into **domain toolkits**, each offering validation workflows tailored to a specific machine learning task (e.g. binary classification).

Currently, we provide toolkits for the following domains:

- tabular data synthesis
- binary classification

Each [package](packages.md) in the ecosystem is organized accordingly---bundling together toolkits for supported domains.

For instance, to compute validation artifacts for a binary classification experiment, import the validation engine at:

```python
from artifact_core.binary_classification import BinaryClassificationEngine

...
```