# Domain Specific Toolkits

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>  

In line with our [design philosophy](design_philosophy.md), Artifact-ML is organized into **domain-specific toolkits**, each offering validation workflows tailored to a given machine learning task (e.g. binary classification).

Currently, the following domains are supported:

- tabular data synthesis,
- binary classification.

Each [package](pages/packages.md) in the ecosystem follows this structure, bundling together complete toolkits for all supported domains.

For instance, to compute validation artifacts for a binary classification experiments import the validation engine at:  
```python
from artifact_core.binary_classification import BinaryClassificationEngine

...