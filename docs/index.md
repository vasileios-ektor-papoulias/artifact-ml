# Artifact-ML

<p align="center">
  <img src="assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

## Overview

[Artifact-ML](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) eliminates imperative glue code in machine learning experiments by providing the tools to build **shareable** workflows **declaratively**.

By *shareable*, we refer to workflows that are **defined once** and **reused across multiple models within the same task category**.

By *declarative*, we refer to building through expressing high-level intent---rather than catering to implementation details.

The project comprises three packages:

- [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core): foundational interfaces and abstractions for building validation workflows declaratively.
- [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment): experiment tracking toolkit supporting popular tracking backends (e.g. [Mlflow](https://mlflow.org/)).
- [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch): interfaces and abstractions for building shareable deep learning experiments declaratively.

## Contents

- [Packages](pages/packages.md) — overview of the core packages comprising Artifact-ML.  
- [Getting Started](pages/getting_started.md) — how to install and begin using Artifact-ML.  
- [Value Proposition](pages/value_proposition.md) — a concrete demonstration of the problem (and solution) addressed by Artifact-ML.  
- [Motivating Example](pages/motivating_example.md) — a concrete demonstration of the problem (and solution) addressed by Artifact-ML.  
- [Design Philosophy](pages/design_philosophy.md) — a deep dive into the core principles underlying the project.  
- [Domain Specific Toolkits](pages/domain_specific_toolkits.md) — a deep dive into the core principles underlying the project.  