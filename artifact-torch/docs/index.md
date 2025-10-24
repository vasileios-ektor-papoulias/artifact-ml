# artifact-torch

<p align="center">
  <img src="assets/artifact_ml_logo.svg" width="300" alt="Artifact-ML Logo">
</p>

[`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/artifact-torch) provides Pytorch integration for [Artifact-ML](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/).

It offers the tools to build reusable, end-to-end deep learning workflows declaratiely, abstracting away engineering complexity to let researchers focus on architectural innovation. 

It handles **all training loop concerns** aside from model architecture and data pipelines, enabling seamless, declarative customization.  

Validation workflows are relegated to [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core), while deep learning–specific workflows are structured by organizing competing models into a type hierarchy and implementing a dual callback system.

It stands alongside:

- [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core): a unified interface for the declarative computation of diverse validation artifacts in ML experiments.
- [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment): experiment orchestration extension for building reusable validation workflows with integrated tracking.

## Topics

- [Getting Started](getting_started.md) - quick installation instructions.
- [User Guide](user_guide.md) — general user instructions.
- [Architecture](architecture.md) — high level framework architecture.  
- [Core Entities](core_entities.md) — framework core entity specification.
- [Development Guide](development_guide.md) — low-level development guidelines.