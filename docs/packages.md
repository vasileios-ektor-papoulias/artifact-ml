# Packages

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>  
  
[Artifact-ML](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/) comprises **three** packages:  

## [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core)

The framework foundation, providing a unified interface for the declarative computation of diverse validation artifacts in ML experiments.  

Its objective is to enable reusable validation workflows by providing the tools to trigger artifacts by name---with zero adapter code.

For more details consult the [package's docs](https://artifact-ml.readthedocs.io/en/latest/artifact-core).

## [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment)

The experiment orchestration extension to [Artifact-ML](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/).  

It provides the tools to build reusable validation workflows with integrated tracking using popular backend services e.g. [Mlflow](https://mlflow.org/).

For more details consult the [package's docs](https://artifact-ml.readthedocs.io/en/latest/artifact-experiment).

## [`artifact-torch`]((https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment))

Pytorch integration for [Artifact-ML](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/).

It offers the tools to build reusable, end-to-end deep learning workflows declaratiely, abstracting away engineering complexity to let researchers focus on architectural innovation. 

It handles **all training loop concerns** aside from model architecture and data pipelines, enabling seamless, declarative customization.  

Validation workflows are relegated to [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core), while deep learning–specific workflows are structured by organizing competing models into a type hierarchy and implementing a dual callback system.

For more details consult the [package's docs](https://artifact-ml.readthedocs.io/en/latest/artifact-torch).

## Getting Started

For installation instructions refer to the following [page](getting_started.md).