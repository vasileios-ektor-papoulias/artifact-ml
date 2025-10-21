# artifact-core

<p align="center">
  <img src="assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

The foundation of [**Artifact-ML**](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main).

This package stands alongside:

- [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment): The framework's experiment tracking extension.
- [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch): PyTorch integration for building reusable deep-learning workflows.

`artifact-core` provides a unified minimal interface for the computation of heterogeneous validation artifacts in machine learning experiments.

Here, we use the word *minimal* to refer to an interface that is as thin as possible given its purpose.

The goal is to enable declarative experiment orchestration through simple enum-based configuration.

By abstracting away unique parameter requirements (static data specifications, hyperparameters) into framework-managed components, `artifact-core` enables downstream client code (e.g. experiment scripts) to invoke a wide array of validation artifacts using only type enumerations---as opposed to artifact-specific argument profiles.

This design eliminates the need for custom integration code per artifact, enabling generic experiment scripts that scale seamlessly across diverse validation requirements with zero modification/ friction.

## Topics

- [User Guide](user_guide.md) — general user instructions.
- [Domain Toolkits](domain_toolkits.md) — project organization by application domain.
- [Table Comparison Toolkit](table_comparison_toolkit.md) — table comparison toolkit user guide.
- [Architecture](architecture.md) — high level framework architecture.  
- [Core Entities](core_entities.md) — framework core entity specification.
- [Implementation Guide](implementation_guide.md) — implementation deep-dive.
- [Contributing Artifacts](contributing_artifacts.md) — development guide for new validation artifacts.