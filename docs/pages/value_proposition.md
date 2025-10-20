# Value Proposition

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

Machine learning experiment code is often cluttered with imperative logic and repeated boilerplate, making it difficult to maintain, scale, or reuse across projects.

This typically traces back to unnecessary couplings between experiment workflows and model implementation details.

To illustrate: a simple conceptual requirement such as *compute PCA projection*—something any tabular synthesizer should be able to do—often turns into *compute PCA projection after extracting numeric columns, handling this model’s NaN patterns, and converting from this model’s specific output format.* Similarly, workflows that should operate on *“models that generate tabular data”* end up tied to *neural network models* embedding architectural assumptions where none are needed.

This contamination of intent forces researchers to re-implement validation logic across models, even when the conceptual requirements remain identical.  

Artifact-ML addresses this by providing shareable ML experiment workflows defined declaratively.

By *shareable*, we refer to workflows that are **defined once** and **reused across multiple models within the same task category**.

By *declarative*, we refer to building through expressing high-level intent---rather than catering to implementation details.

The upshot is:

- **Reduced friction in the research process** — researchers can focus on iterating and exploring new ideas, with immediate, effortless feedback enabled by the seamless presentation of declaratively defined validation artifacts.

- **Eliminated duplication of code** — no need for model-specific validation logic or imperative glue code; validation workflows are defined once and reused across experiments.

- **Consistent and trustworthy evaluation** — validation is standardized across experiments, eliminating variance caused by subtle discrepancies in custom logic.
