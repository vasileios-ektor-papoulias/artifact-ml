# Packages

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>  
  
[Artifact-ML](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/) comprises **three** packages:  

## [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core)

The framework foundation.

It provides a unified interface for the declarative computation of diverse validation artifacts in ML experiments.

Its objective is to enable reusable validation workflows by providing the tools to trigger artifacts by name---with zero adapter code.

In line with our [design philosophy](design_philosophy.md), achieving this sets the stage for [Artifact-ML’s](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) broader objective: the elimination of imperative glue code in ML experiments at large.

For more details consult the [package's docs](https://artifact-ml.readthedocs.io/en/latest/artifact-core).

## [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment)

Experiment orchestration extension.

It provides the tools to build reusable validation workflows with integrated tracking using popular backend services (e.g. [Mlflow](https://mlflow.org/)).

For more details consult the [package's docs](https://artifact-ml.readthedocs.io/en/latest/artifact-experiment).

## [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch)

Pytorch integration.

It offers the tools to build reusable, end-to-end deep learning workflows declaratively.

It handles **all aspects of the training loop** aside from model architecture and data pipelines, abstracting away engineering complexity to let researchers focus on architectural innovation.

For more details consult the [package's docs](https://artifact-ml.readthedocs.io/en/latest/artifact-torch).

## Getting Started

For installation instructions refer to the following [page](getting_started.md).