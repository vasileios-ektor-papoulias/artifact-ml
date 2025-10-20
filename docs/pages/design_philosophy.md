# Design Philosophy

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## Organization in Validation-Oriented Toolkits

Our view is that the **natural common ground between experiments lies in validation, not training**.

When deciding which logic should be shareable between models, we start by asking a simple question: *What does the model do?* Any model that can answer this question in the same way—e.g., any model that performs the same task—should be able to share the same experiment workflow.

Training may differ across architectures, but **validation defines the essence of the task itself**.

To formalize this, we identify **model families through validation resource grouping**: we determine the ***minimal resources*** required to execute validation logic and group models by their capacity to emit it. 

This principle gives rise to [**domain-specific toolkits**](domain_specific_toolkits.md): self-contained validation ecosystems aligned with specific application domains (e.g., tabular data synthesis, binary classification).

Each toolkit integrates shared framework infrastructure with domain-specific evaluation primitives, facilitating the development of experiment workflows reusable by all models that *do the same thing*.

## Type-Driven Extensions Within Domain Toolkits

The key idea is to treat **auxiliary model structure as orthogonal to validation requirements**.  

Within a given validation domain, models are organized into a **type hierarchy** that captures this additional structure.

Workflows are then defined in a **dual hierarchy**: allowing them to automatically determine which models they are compatible with.  

This approach makes it possible to precisely match models to workflows, enabling structured reuse rather than ad-hoc coupling.

Once established, it allows us to **abstract away entire experiment workflows**, not just validation components.  

In the case of `artifact-torch`, the shared structure among models is characterized by the **I/O profile of their forward pass**.

We build a **strongly typed callback system** dual to the resulting hierarchy.

Ultimately, this allows us to develop workflows usable by any network within the same domain area adhering to at least the required forward pass IO profile.
