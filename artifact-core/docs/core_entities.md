# Core Entities

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

`artifact-core` operates by coordinating the interaction of specialized entities across its [architectural](architecture.md) layers:

## **User Interaction Layer**
The primary interface layer that users interact with to orchestrate comprehensive validation workflows:

- **ArtifactEngine**: Simple yet flexible entry point providing a unified interface for executing individual validation artifacts. It manages registry access, handles artifact instantiation with appropriate configurations, and provides straightforward methods for computing validation results. Crucially method signatures are minimal, relegating secondary arguments like static data spec information and hyperparameters to be handled by the framework.
```python
engine = TableComparisonEngine(resource_spec=spec)
pca_plot = engine.produce_dataset_comparison_plot(
    plot_type=TableComparisonPlotType.PCA_PROJECTION_PLOT,
    dataset_real=df_real,
    dataset_synthetic=df_synthetic,
)
```

**Architecture Role**: ArtifactEngine serves as the main entry point that abstracts artifact lookup, instantiation, and execution complexity while providing a clean interface for individual artifact computation - users specify what artifact they want and the data to compute it on, and get back the validation result.

## **Framework Infrastructure Layer**
Internal framework components that provide the computational foundation and artifact management system:

- **Artifact**: Abstract computation units that define the `compute()` method contract. Artifacts are heterogeneous (multi-modal) and categorized by return type:
  - **Scores**: Single numerical metrics
  - **Arrays**: Numpy arrays containing computed data  
  - **Plots**: Matplotlib figures for visualization
  - **Collections**: Groups of related artifacts (e.g., multiple scores or plots)

- **ArtifactRegistry**: Management system that organizes artifacts by type and coordinates registration, retrieval, and instantiation. Artifacts in the same registry share resources, return types, and resource specification types.

- **ArtifactType**: Enumeration system that provides unique identifiers for different artifact implementations, enabling dynamic lookup and instantiation within registries.

- **ArtifactResources**: Generic data objects that artifacts operate on, providing the input datasets or resources required for validation computation. The design of resource types is central to the framework's extensibility—by defining domain-specific resource contracts, we naturally group thematically related models that share validation requirements, enabling them to leverage common validation logic regardless of their internal architectural differences.

- **ArtifactResourceSpec**: Protocol definitions that capture the structural properties and schema characteristics of validation resources (e.g., feature types and data schemas for tabular data).

- **ArtifactHyperparams**: Configuration objects that enable customizable artifact behavior through domain-specific parameter settings.

**Entity Coordination**: ArtifactEngine coordinates with ArtifactRegistry for artifact lookup and instantiation using ArtifactResourceSpec for type validation, while Artifact implementations use ArtifactResources for computation and ArtifactHyperparams for configuration. The interplay between ArtifactResources and ArtifactResourceSpec ensures type safety while enabling the framework's core capability of grouping diverse models by their validation resource compatibility.

## **External Dependencies**
Configuration and data inputs that drive artifact computation and enable framework customization:

- **Configuration Files**: JSON-based parameter definitions that control artifact behavior, enable project-specific customization, and support configuration inheritance through the `.artifact` directory system.

- **Resource Data**: Input datasets and validation resources that provide the raw data for artifact computation, typically domain-specific data formats (e.g., pandas DataFrames for tabular data).

**Integration Flow**: Configuration Files define artifact parameters loaded by ArtifactHyperparams, while Resource Data flows through ArtifactResources to enable domain-specific validation computations.