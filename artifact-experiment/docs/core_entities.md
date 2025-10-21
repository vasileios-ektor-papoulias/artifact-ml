# Core Entities

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>


The framework operates by coordinating the interaction of specialized entities across its four [architectural](architecture.md) layers:

## **User Specification Layer**
Users define validation requirements through simple subclass hooks, eliminating complex implementation details:

- **ValidationPlan**: Provides declarative validation specification through subclass hooks, transforming user requirements into executable workflows with experiment tracking capabilities.

```python
class MyValidationPlan(TableComparisonPlan):
    @staticmethod
    def _get_score_types() -> List[TableComparisonScoreType]:
        return [TableComparisonScoreType.MEAN_JS_DISTANCE]
    
    @staticmethod 
    def _get_plot_types() -> List[TableComparisonPlotType]:
        return [TableComparisonPlotType.PDF]
```

**Architecture Role**: ValidationPlan orchestrates the entire validation workflow by using ArtifactFactories to create computation callbacks and CallbackHandlers to execute them, transforming specifications into executable validation workflows.

**Result Management**: ValidationPlan caches all computed artifacts in RAM for immediate access and inspection, while simultaneously leveraging experiment tracking exports for persistent storage and collaboration.

## **Execution Orchestration Layer**
The orchestration layer transforms user specifications into executable workflows through coordinated entity interaction:

- **ArtifactFactories**: Create callbacks that integrate with artifact-core's computation engine, bridging validation specification with actual computation
- **Callbacks**: Execute individual validation computations and report results to tracking clients for export
- **CallbackHandlers**: Orchestrate callback execution across artifact types, managing validation workflow execution and coordinating with tracking clients

## **Backend Integration Layer**
The integration layer provides backend-agnostic experiment tracking through specialized components:

- **TrackingClients**: Coordinate experiment export by orchestrating loggers and adapters for unified backend interaction
- **ArtifactLoggers**: Handle artifact-specific export logic, converting computed results into backend-compatible formats
- **RunAdapters**: Normalize backend-specific run objects, providing consistent interfaces across different experiment tracking platforms

```python
# Unified interface across backends
mlflow_client = MlflowTrackingClient.build(experiment_id="my_experiment")
clearml_client = ClearMLTrackingClient.build(experiment_id="my_project") 
neptune_client = NeptuneTrackingClient.build(experiment_id="my_project")
filesystem_client = FilesystemTrackingClient.build(experiment_id="my_experiment")
```

**Entity Coordination**: TrackingClients coordinate experiment export by using ArtifactLoggers for artifact-specific export logic and RunAdapters for backend normalization. RunAdapters interface directly with experiment backends while ArtifactLoggers depend on adapters for actual export execution.

## **External Dependencies**
External systems and frameworks that the validation plan ecosystem depends on for computation and persistence:

- **artifact-core Computation Engine**: Individual validation computation units derive from `artifact-core`. `artifact-experiment` delegates validation logic to `artifact-core` **Artifact** instances. These are wrapped in callbacks and executed through handlers to build comprehensive validation workflows.

- **Experiment Tracking Backends**: External platforms that provide persistent storage and collaboration capabilities for experiment results. Supported backends include MLflow, ClearML, Neptune, and local filesystem, all accessed through the unified RunAdapter interface.

**Integration Flow**: ArtifactFactories wrap artifact-core artifacts in callbacks for validation computation, while RunAdapters interface with experiment tracking backends for result persistence, creating a seamless bridge between validation computation and experiment tracking ecosystems.

## **Seamless Integration Flow**
The complete flow demonstrates how entities collaborate to achieve the framework's goals:

1. **ValidationPlan** defines artifacts through subclass hooks
2. **ArtifactFactories** create callbacks integrating with artifact-core computation
3. **CallbackHandlers** orchestrate callback execution workflows
4. **Callbacks** perform computations and report to tracking clients
5. **TrackingClients** coordinate export using loggers and adapters  
6. **RunAdapters** normalize backend interfaces for seamless integration
7. **ArtifactLoggers** handle artifact-specific export to experiment backends

This coordinated interaction transforms artifact-core's raw validation capabilities into reusable, executable validation plans with automatic experiment tracking across multiple backends.