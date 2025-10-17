# Artifact-ML Design Philosophy

This document outlines the core design principles that guide the development of Artifact-ML. These principles address the fundamental challenges in machine learning experiment infrastructure and explain how Artifact-ML's architecture enables reusable, maintainable validation workflows.

Understanding these principles is essential for appreciating how Artifact-ML eliminates code duplication and enables researchers to focus on innovation rather than infrastructure maintenance.

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

## 🎯 Design Philosophy

Artifact-ML addresses the issues exhibited in the [motivating example](motivating_example.md) through four core design principles:

### Code Should Express Intent, Nothing More

Previously, code could not be shared because it attempted to express conceptually general requirements but became contaminated with implementation specifics. For example, "compute PCA projection" (achievable by any tabular synthesizer) would become "compute PCA projection after extracting numeric columns, handling this model's NaN patterns, and converting from this model's specific output format." Similarly, validation workflows would reference "neural network models" when they should simply express "models that generate tabular data"—the internal architecture being irrelevant to validation logic. This contamination forced researchers to rewrite validation logic for each model, even when expressing identical conceptual requirements. By ensuring code expresses only true intent—"compute PCA projection," "generate marginal plots," "compare model-generated data to real data"—without embedding extraneous information-validation workflows remain truly general and reusable.

But how do we achieve this pure expression of intent? The answer lies in carefully designed interface contracts that abstract away implementation specifics while preserving essential functionality.

### Interface Contracts Enable Reusable Validation Infrastructure

Client code duplication and reusability challenges can be fundamentally resolved through adherence to well-designed interface contracts that are sufficiently flexible to accommodate the full scope of intended functionality. When interface contracts properly abstract implementation details while preserving expressive power, they eliminate adapter code requirements and enable truly reusable validation workflows.

But this raises a critical design challenge: how do we determine what structure interface contracts should preserve? What is the minimal shared foundation that enables reusability without over-abstracting away essential functionality?

### Domain-Specific Interface Specification Through Resource Grouping

Reusable validation logic can only exist among models within the same application domain—a tabular synthesizer cannot share validation logic with an image classifier, as they require fundamentally different validation approaches. This constraint shapes how we design interface contracts: rather than pursuing universal contracts across all ML domains, we must specify contracts that enable reusability within domain boundaries.

We address this by identifying model families through concrete specifications of the ***minimal shared structure*** unifying them. Rather than grouping models by architectural similarities, we carefully design flexible hierarchical types for the resources required to execute validation logic—thereby implicitly grouping models by their ability to produce the same validation resources.

This approach enables the creation of **domain-specific toolkits**—complete validation ecosystems tailored to specific ML application domains (e.g., tabular data synthesis, image generation). Each toolkit leverages shared framework infrastructure while providing specialized artifacts, training infrastructure, and validation plans optimized for its domain's unique validation requirements.

The result is a hierarchical design: universal abstractions at the core, domain-specific configurations in the middle layer, and model-specific implementations at the edges.

### Framework-Agnostic Design Through High-Level Abstractions

We work with abstract generic classes to avoid coupling to concrete types that would tie us to specific frameworks or model implementations.

Our goal is to provide unified reusable experiment/ validation code for thematically related ML models. If our attempt is to be successful, these plans must remain framework-independent and model-agnostic. The infrastructure we produce ought to work equally well with graphical models as it does with neural networks and everything in between.

If we achieve this design goal, it should be straightforward to provide integration with any ML framework of interest—as we've demonstrated with PyTorch through `artifact-torch`. The same abstract foundations can support TensorFlow, JAX, scikit-learn, or any other framework through lightweight adapter layers. 