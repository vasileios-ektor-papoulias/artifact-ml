# artifact-core

<p align="center">
  <img src="assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

The foundation of [**Artifact-ML**](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main).

This package stands alongside:

- [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment): The framework's experiment tracking extension.
- [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch): PyTorch integration for building reusable deep-learning workflows declaratively.

`artifact-core` provides a unified interface for the computation of diverse validation artifacts.

The goal is to enable reusable validation workflows by providing the tools to trigger diverse artifacts by name---with zero adapter code.

In line with our [design philosophy](https://artifact-ml.readthedocs.io/en/latest/value_philosophy/), achieving this establishes the foundation for [Artifact-ML’s](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) broader objective: eliminating imperative glue code in ML experiment workflows.

## Topics

- [Getting Started](getting_started.md) - quick installation instructions.
- [User Guide](user_guide.md) — general user instructions.
- [Domain Toolkits](domain_toolkits.md)
    - [Table Comparison Toolkit](table_comparison_toolkit.md) — guide to the tabular synthesis validation toolkit.
    - [Binary Classification Toolkit](binary_classification_toolkit.md) — guide to the binary classification validation toolkit.
- [Architecture](architecture.md) — high level framework architecture.  
- [Core Entities](core_entities.md) — framework core entity specification.
- [Development Guide](development_guide.md) — low-level development guidelines.
- [Contributing Artifacts](contributing_artifacts.md) — development guide for new validation artifacts.