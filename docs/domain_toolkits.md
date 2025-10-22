# Domain Toolkits

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

In line with our [design philosophy](design_philosophy.md), Artifact-ML is organized in **domain toolkits**.

Each toolkit facilitates building experiment workflows tailored to a specific machine learning application domain (e.g. binary classification).

Broadly, domains are defined by the validation resources they require.

For instance, tabular synthesizers are evaluated by comparing the characteristics of the real and synthetic datasets.

Consequently, this tuple characterizes the tabular data synthesis domain.

## Supported Domains

Currently, we provide toolkits for the following domains:

- tabular data synthesis,
- binary classification.

## Package Organization

Each [package](packages.md) in the ecosystem is organized accordingly---bundling together toolkits for supported domains.

Accordingly, all packages expose their user-facing components by domain as top-level modules in: `<package_name>/<domain_toolkit_name>`.

For instance, to compute validation artifacts for a binary classification experiment, import the validation engine at:

```python
from artifact_core.binary_classification import BinaryClassificationEngine

...
```

## Relevant Pages

For documentation on the domain toolkits provided by each package within the Artifact-ML ecosystem, see:

- [Domain Toolkits: artifact-core ](https://artifact-ml.readthedocs.io/en/latest/artifact-core/domain_toolkits/)
    - [Table Comparison Toolkit](https://artifact-ml.readthedocs.io/en/latest/artifact-core/domain_toolkits/table_comparison/)
    - [Binary Classification Toolkit](https://artifact-ml.readthedocs.io/en/latest/artifact-core/domain_toolkits/binary_classification/)