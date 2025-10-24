# Core Entities

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>


[`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment) operates by coordinating the interaction of specialized entities across its four [architectural](architecture.md) layers:

## User Specification Layer

- **ValidationPlan**: Provides declarative validation specification through subclass hooks.

```python
class MyValidationPlan(TableComparisonPlan):
    @staticmethod
    def _get_score_types() -> List[TableComparisonScoreType]:
        return [TableComparisonScoreType.MEAN_JS_DISTANCE]
    
    @staticmethod 
    def _get_plot_types() -> List[TableComparisonPlotType]:
        return [TableComparisonPlotType.PDF]
```

## Execution Orchestration Layer

- **ArtifactFactories**: Create callbacks that integrate with `artifact-core`'s computation engine.
- **Callbacks**: Execute individual validation computations and report results to tracking clients for export.
- **CallbackHandlers**: Orchestrate callback execution.

## Backend Integration Layer

- **TrackingClients**: Coordinate experiment export by orchestrating loggers and run adapters for unified backend interaction.
- **ArtifactLoggers**: Handle export logic, converting computed results into backend-compatible formats.
- **RunAdapters**: Normalize backend-specific run objects, providing consistent interfaces across different experiment tracking platforms.

```python
# Unified interface across backends
mlflow_client = MlflowTrackingClient.build(experiment_id="my_experiment")
clearml_client = ClearMLTrackingClient.build(experiment_id="my_project") 
neptune_client = NeptuneTrackingClient.build(experiment_id="my_project")
filesystem_client = FilesystemTrackingClient.build(experiment_id="my_experiment")
```

## External Dependencies

- **`artifact-core`**: Individual validation computation units derive from `artifact-core`. These are wrapped in callbacks and executed through handlers to build comprehensive validation workflows.

- **Experiment Tracking Backends**: External platforms that provide persistent storage and collaboration capabilities for experiment results.

Supported backends include:

  - [MLflow](https://mlflow.org/),
  - [ClearML](https://clear.ml/),
  - [Neptune](https://neptune.ai/),
  - local filesystem,
  - in-memory caching,

all accessed through the unified RunAdapter interface.

## Integration Flow
The complete flow demonstrates how entities collaborate to achieve the framework's goals:

1. **ValidationPlan** specifies artifacts of interest through subclass hooks.
2. **CallbackFactories** create callbacks wrapping `artifact-core` computation.
3. **CallbackHandlers** orchestrate callback execution workflows.
4. **Callbacks** perform computations and report to tracking clients,
5. **TrackingClients** coordinate export using loggers and run adapters,
6. **RunAdapters** normalize tracking backend service interfaces,
7. **ArtifactLoggers** handle export to tracking backend services.