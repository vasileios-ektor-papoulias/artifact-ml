# ⚙️ Artifact-ML

> Artifact-ML provides shareable machine learning experiment infrastructure with a primary focus on declarative validation. It minimizes imperative code to eliminate duplication and promote concise, reusable experiment workflows.

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="600" alt="Artifact-ML logo">
</p>

---

## Welcome to the Artifact-ML Documentation

This is the main entry point for the docs. Use the **sidebar** to browse component docs, guides, and references.  
If you’re new, start with **Getting Started** for setup and a quick tour.

- **Overview & Philosophy** — the reasoning and design behind Artifact-ML.
- **Core Components** — package docs for `artifact-core`, `artifact-experiment`, and `artifact-torch`.
- **Guides & Examples** — practical workflows and end-to-end demos.
- **Reference** — API and type specifications.

---

## Overview & Purpose

Machine learning experiment code is often cluttered with imperative logic and repeated boilerplate, making it difficult to maintain, scale, or reuse across projects. Artifact-ML addresses this by providing reusable experiment infrastructure with a primary focus on standardized validation.

It enables the design of shareable validation logic that is reusable by any experiment within a given task category.

This is achieved through carefully designed type hierarchies and clean interface contracts serving to decouple high-level experiment orchestration from low-level model implementation.

The project is organized into domain-specific toolkits, each offering validation workflows tailored to common machine learning tasks (e.g., tabular data synthesis, binary classification).

The upshot is:

- **Reduced friction in the research process** — researchers can focus on iterating and exploring new ideas, with immediate, effortless feedback enabled by the seamless presentation of declaratively defined validation artifacts.
- **Eliminated duplication of code** — no need for model-specific validation logic or imperative glue code; validation workflows are defined once and reused across experiments.
- **Consistent and trustworthy evaluation** — validation is standardized across experiments, eliminating variance caused by subtle discrepancies in custom logic.

- See the **motivating example**: [motivating_example.md](motivating_example.md)  
- Deep dive into **design philosophy**: [design_philosophy.md](design_philosophy.md)

<p align="center">
  <img src="../../assets/pdf_comparison.png" width="600" alt="PDF Comparison">
</p>

---

## Packages

<div class="grid cards" markdown>

-   :material-cube-outline: **artifact-core**  
    The framework foundation, defining the base abstractions and interfaces for the design and execution of validation artifacts.  
    It offers pre-built out-of-the-box artifact implementations with seamless support for custom extensions.

-   :material-flask-outline: **artifact-experiment**  
    The experiment orchestration and tracking extension to Artifact-ML.  
    It facilitates the design of purely declarative validation workflows leveraging `artifact-core`.  
    It provides fully automated tracking capabilities with popular backends (e.g. Mlflow).

-   :material-lightning-bolt-outline: **artifact-torch**  
    A deep learning framework built on top of `artifact-core` and `artifact-experiment`, abstracting away engineering complexity to let researchers focus on architectural innovation.  
    It handles all training loop concerns aside from model architecture and data pipelines, enabling seamless, declarative customization via a system of typed callbacks.

</div>

---

## Getting Started (at a glance)

For full details, open **Getting Started** from the sidebar.

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-ml.git
cd artifact-ml
# Example: install artifact-core
cd artifact-core && poetry install