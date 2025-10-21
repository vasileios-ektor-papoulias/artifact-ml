# artifact-experiment

<p align="center">
  <img src="assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

`artifact-experiment` constitutes the experiment orchestration and tracking extension to Artifact-ML.

It bridges the gap between validation computation and experiment tracking through a core **validation plan** abstraction responsible for the execution and tracking of artifact collections.

It stands alongside:
- [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core): Framework foundation providing a flexible uniform interface for the computation of validation artifacts.
- [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch): PyTorch integration for building reusable deep-learning workflows.

## Topics

- [User Guide](user_guide.md) — general user instructions.
- [Architecture](architecture.md) — high level framework architecture.  
- [Core Entities](core_entities.md) — framework core entity specification.
- [Development Guide](development_guide.md) — low-level development guidelines.