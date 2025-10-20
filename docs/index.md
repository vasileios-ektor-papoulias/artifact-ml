# Artifact-ML

> Artifact-ML eliminates imperative glue code in machine learning experiments by providing the tools to build **shareable** workflows **declaratively**.


<p align="center">
  <img src="assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

## Overview

Machine learning experiment code is often cluttered with imperative logic and repeated boilerplate, making it difficult to maintain, scale, or reuse across projects. Artifact-ML addresses this by providing reusable experiment infrastructure with a primary focus on standardized validation.

It enables the design of shareable validation logic that is reusable by any experiment within a given task category.

This is achieved through carefully designed type hierarchies and clean interface contracts serving to decouple high-level experiment orchestration from low-level model implementation.

The project is organized into domain-specific toolkits, each offering validation workflows tailored to common machine learning tasks (e.g., tabular data synthesis, binary classification).

The upshot is:

- **Reduced friction in the research process** — researchers can focus on iterating and exploring new ideas, with immediate, effortless feedback enabled by the seamless presentation of declaratively defined validation artifacts.

- **Eliminated duplication of code** — no need for model-specific validation logic or imperative glue code; validation workflows are defined once and reused across experiments.

- **Consistent and trustworthy evaluation** — validation is standardized across experiments, eliminating variance caused by subtle discrepancies in custom logic.

## Contents

- [Motivating Example](pages/motivating_example.md) — a concrete demonstration of the problem (and solution) addressed by Artifact.  
- [Design Philosophy](pages/design_philosophy.md) — a deep dive into the core principles underlying the project.  
- [Packages](pages/packages.md) — overview of the core packages comprising Artifact-ML.  
- [Getting Started](pages/getting_started.md) — how to install and begin using Artifact-ML.