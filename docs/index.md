# Artifact-ML

<p align="center">
  <img src="assets/artifact_ml_logo.svg" width="300" alt="Artifact-ML Logo">
</p>

## Overview

[Artifact-ML](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) eliminates imperative glue code in machine learning experiments by providing the tools to build **shareable** workflows **declaratively**.

By *shareable*, we refer to workflows that are **defined once** and **reused across multiple models within the same task category**.

By *declarative*, we refer to building through expressing high-level intent---rather than catering to implementation details.

## Packages

The project comprises three packages:

- [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core) [docs](../../artifact-core/docs/pages/home.md): foundational interfaces and abstractions for building validation workflows.
- [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment) [docs](../../artifact-experiment/docs/pages/home.md): experiment tracking toolkit supporting popular tracking backends (e.g. [Mlflow](https://mlflow.org/)).
- [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch) [docs](../../artifact-torch/docs/pages/home.md): interfaces and abstractions for building deep learning experiments.

## Contents

- [Packages](pages/packages.md) — overview of the packages comprising the framework.  
- [Getting Started](pages/getting_started.md) — quick installation instructions.  
- [Value Proposition](pages/value_proposition.md) — high-level description of the problem addressed by Artifact-ML.  
- [Motivating Example](pages/motivating_example.md) — a concrete example illustrating the problem (and solution) in action.  
- [Design Philosophy](pages/design_philosophy.md) — the core ideas and principles that shape the framework.  
- [Domain Toolkits](pages/domain_toolkits.md) — how Artifact-ML structures reusable workflows across application domains.