# Packages

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>  
  
Artifact-ML comprises **three** packages:  

## [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core)

The framework foundation, defining the base abstractions and interfaces for the design and execution of validation artifacts.  

It offers pre-built out-of-the-box artifact implementations with seamless support for custom extensions.

For more details consult the [package's docs](artifact-core/docs/index.md).

## [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment)

The experiment orchestration and tracking extension to Artifact-ML.  

It facilitates the design of purely declarative validation workflows (validation plans) leveraging `artifact-core`.  

It provides fully automated tracking capabilities with popular backends (e.g. Mlflow).

For more details consult the [package's docs](artifact-experiment/docs/index.md).

## [`artifact-torch`]((https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment))

A deep learning framework built on top of `artifact-core` and `artifact-experiment`, abstracting away engineering complexity to let researchers focus on architectural innovation.  

It handles **all training loop concerns** aside from model architecture and data pipelines, enabling seamless, declarative customization.  

Validation workflows are relegated to `artifact-core`, while deep learning–specific workflows are structured by organizing competing models into a type hierarchy and implementing a dual, strongly typed callback system.

For more details consult the [package's docs](artifact-torch/docs/index.md).

## Getting Started

For installation instructions refer to the following [page](getting_started.md).