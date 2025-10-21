# Value Proposition

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## Problem: Imperative Experiment Logic

Machine learning experiment code is often cluttered with imperative logic making it difficult to maintain, scale, or reuse across projects.

This is typically the consequence of experiment workflows being unnecessarily coupled to model implementations.

To illustrate: a simple conceptual requirement such as:

*"compute the JS divergence between the synthetic and real datasets"*,

---an operation that should be compatible with any tabular synthesizer---often turns into something like:

*"compute the JS divergence after binning numeric columns and adapting to the output profile of the model at hand".*

In the same spirit, workflows are often unnecessarily tied to specific model architectures. For example, when working with neural networks, validation code may be written against deep learning model types, making it unusable if one later wants to switch to, say, a graphical model.

This contamination of intent with implementation details forces researchers to re-implement validation logic across models, even when conceptual requirements remain identical.

## Solution: Reusable Workflows Defined Declaratively

Artifact-ML addresses the above by providing the tools to build reusable ML experiment workflows declaratively.

By *reusable*, we refer to workflows that are defined once with the potential to be reused by any model within the same task category.

By *declarative*, we refer to building through expressing high-level intent---rather than catering to implementation details.

Experiment workflows become first-class entities in their own right, enabling researchers to pair compatible models and workflows directly, rather than writing custom scripts.

## Upshot

- **Elimination of Imperative Glue Code** — validation workflows are built declaratively, without any model-specific wiring.

- **Elimination of Code Duplication** — a single workflow can be reused with any compatible model, eliminating the need to reimplement the same logic twice.

- **Simpler Workflow Maintenance** — because workflows exist as reusable, independent entities, they are easier to manage, scale and maintain.

- **Reduced Friction in the Research Process** — with custom experiment workflows no longer required, researchers can focus on architectural innovation over repetitive infrastructure setup.
 
- **Consistent and Trustworthy Validation** — validation is standardized across experiments, eliminating unwanted variance caused by subtle discrepancies in custom logic.