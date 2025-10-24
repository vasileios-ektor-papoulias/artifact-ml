# artifact-core

<p align="center">
  <img src="assets/artifact_ml_logo.svg" width="300" alt="Artifact-ML Logo">
</p>

[`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core) constitutes the foundation of [**Artifact-ML**](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main).

It provides a unified interface for the declarative computation of diverse validation artifacts in ML experiments.

Its objective is to enable reusable validation workflows by providing the tools to trigger artifacts by name---with zero adapter code.

In line with our [design philosophy](https://artifact-ml.readthedocs.io/en/latest/design_philosophy/), achieving this sets the stage for [Artifact-ML’s](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) broader objective: the elimination of imperative glue code in ML experiments at large.

[`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core) stands alongside:

- [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment): experiment orchestration extension for building reusable validation workflows with integrated tracking.
- [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch): PyTorch integration for building reusable deep-learning workflows declaratively.

## Topics

- [Getting Started](getting_started.md) - quick installation instructions.
- [User Guide](user_guide.md) — general user instructions.
- [Domain Toolkits](domain_toolkits.md)
    - [Table Comparison Toolkit](domain_toolkits/table_comparison.md) — guide to the tabular synthesis validation toolkit.
    - [Binary Classification Toolkit](domain_toolkits/binary_classification.md) — guide to the binary classification validation toolkit.
- [Architecture](architecture.md) — high level framework architecture.  
- [Core Entities](core_entities.md) — framework core entity specification.
- [Development Guide](development_guide.md) — low-level development guidelines.
- [Contributing Artifacts](contributing_artifacts.md) — development guide for new validation artifacts.