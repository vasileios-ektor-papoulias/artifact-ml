# Packages

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

Artifact-ML consists of three packages:

## [`artifact-core`](../../../artifact-core/docs/pages/home.md)

The framework foundation, defining the base abstractions and interfaces for the design and execution of validation artifacts.

It offers pre-built out-of-the-box artifact implementations with seamless support for custom extensions.

## [`artifact-experiment`](../../../artifact-experiment/docs/pages/home.md)

The experiment orchestration and tracking extension to Artifact-ML.

It facilitates the design of purely declarative validation workflows leveraging `artifact-core`.

It provides fully automated tracking capabilities with popular backends (e.g. Mlflow).

## [`artifact-torch`](../../../artifact-torch/docs/pages/home.md)

A deep learning framework built on top of `artifact-core` and `artifact-experiment`, abstracting away engineering complexity to let researchers focus on architectural innovation.

It handles all training loop concerns aside from model architecture and data pipelines, enabling seamless, declarative customization via a system of typed callbacks.

Models, trainers, and workflows are all strongly typed, and the system leverages type variance and inference to ensure that the right callbacks fit the right trainers and workflows.