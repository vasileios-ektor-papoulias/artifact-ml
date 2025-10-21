# Domain Toolkits

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

`artifact-core` is organized by ML application domain (e.g. tabular data synthesis, binary classification).

Broadly, domains are defined by the validation resources they require.

For instance, in the context of tabular data synthesis, models are evaluated by comparing the real and synthetic datasets---and this tuple constitutes the domain's validation resources.

For each supported domain  the relevant toolkit offers implementations for all core abstractions.

This approach results in focused validation ecosystems benefitting from common infrastructure.

Each domain toolkit implements its own:

**Validation Resources**: resource specification protocol and associated artifact resources.

**Artifact Registries**: specialized registries managing the collection and retrieval of domain-specific artifacts.

**Artifact Engine**: specialized artifact engines providing a unified interface for the computation of domain-specific artifacts.

## Supported Toolkits

- [Table Comparison Toolkit](domain_toolkits/table_comparison_toolkit.md) — toolkit supporting tabular data synthesis workflows.
- [Domain Toolkits](domain_toolkits/binary_classification_toolkit.md) — toolkit supporting binary classification workflows.