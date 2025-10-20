# Value Proposition

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

Machine learning experiment code is often cluttered with imperative logic making it difficult to maintain, scale, or reuse across projects.

his typically stems from unnecessary coupling between experiment workflows and model implementation details.

To illustrate: a simple conceptual requirement such as *compute PCA projection*—something any tabular synthesizer should be able to do—often turns into *compute PCA projection after extracting numeric columns, handling this model’s NaN patterns, and converting from this model’s specific output format.* Similarly, workflows that should operate on *“models that generate tabular data”* end up tied to *neural network models* embedding architectural assumptions where none are needed.

This contamination of intent with implementation details forces researchers to re-implement validation logic across models, even when conceptual requirements remain identical.  

Artifact-ML addresses this by providing the tools to build reusable ML experiment workflows declaratively.

By *reusable*, we refer to workflows that are defined once with the potential to be reused by any model within the same task category.

By *declarative*, we refer to building through expressing high-level intent---rather than catering to implementation details.

The upshot is:

- **Elimination of Imperative Glue Code** — validation workflows are defined declaratively, removing the need for model-specific wiring or boilerplate code.

- **Elimination of Experiment Code Duplication** — a single declarative workflow can be reused across multiple models and experiments, avoiding repeated implementation of the same logic.

- **Reduced Friction in the Research Process** — models integrate directly with existing reusable experiment workflows, minimizing---or even eliminating---the need to develop custom ones. This allows researchers to focus on architectural innovation over repetitive infrastructure setup.
 
- **Consistent and Trustworthy Validation** — validation is standardized across experiments, eliminating unwanted variance caused by subtle discrepancies in custom logic.
