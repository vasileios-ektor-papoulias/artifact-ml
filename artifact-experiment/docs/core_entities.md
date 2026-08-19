# Core Entities

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## Entities by Layer

[`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment) operates by coordinating the interaction of specialized entities across its four [architectural](architecture.md) layers:

### User Specification Layer

- **ArtifactPlan**: Provides declarative validation specification through subclass hooks. Users subclass a domain plan such as `TableComparisonPlan` or `BinaryClassificationPlan` and instantiate it with `create(resource_spec=..., tracking_client=...)`.

```python
class MyArtifactPlan(TableComparisonPlan):
    @staticmethod
    def _get_score_types() -> List[TableComparisonScoreType]:
        return [TableComparisonScoreType.MEAN_JS_DISTANCE]

    @staticmethod
    def _get_array_types() -> List[TableComparisonArrayType]:
        return []

    @staticmethod
    def _get_plot_types() -> List[TableComparisonPlotType]:
        return [TableComparisonPlotType.PDF]

    @staticmethod
    def _get_score_collection_types() -> List[TableComparisonScoreCollectionType]:
        return []

    @staticmethod
    def _get_array_collection_types() -> List[TableComparisonArrayCollectionType]:
        return []

    @staticmethod
    def _get_plot_collection_types() -> List[TableComparisonPlotCollectionType]:
        return []
```

### Execution Orchestration Layer

- **ArtifactCallbackFactory**: Domain factories (e.g. `TableComparisonCallbackFactory`) create callbacks that integrate with `artifact-core`'s computation engine.
- **Callbacks**: Execute individual validation computations and write results to the tracking queue through queue writers.
- **CallbackHandlers**: Orchestrate callback execution.

### Backend Integration Layer

- **TrackingClients**: The unified user-facing API for experiment tracking. Each client owns a `TrackingQueue` and a `BackendLoggingWorker` that consumes it asynchronously.
- **BackendLoggingWorker & ArtifactLoggers**: The worker holds one logger per artifact kind (scores, arrays, plots, their collections, and files); loggers handle export logic, converting computed results into backend-compatible formats.
- **RunAdapters**: Normalize backend-specific run objects, providing consistent interfaces across different experiment tracking platforms.

```python
# Unified interface across backends
mlflow_client = MlflowTrackingClient.build(experiment_id="my_experiment")
clearml_client = ClearMLTrackingClient.build(experiment_id="my_project")
neptune_client = NeptuneTrackingClient.build(experiment_id="my_project")
filesystem_client = FilesystemTrackingClient.build(experiment_id="my_experiment")
in_memory_client = InMemoryTrackingClient.build(experiment_id="my_experiment")
```

### External Dependencies

- **`artifact-core`**: Individual validation computation units derive from `artifact-core`. These are wrapped in callbacks and executed through handlers to build comprehensive validation workflows.

- **Experiment Tracking Backends**: External platforms that provide persistent storage and collaboration capabilities for experiment results.

Supported backends include:

  - [MLflow](https://mlflow.org/),
  - [ClearML](https://clear.ml/),
  - [Neptune](https://neptune.ai/),
  - local filesystem,
  - in-memory caching,

all accessed through the unified RunAdapter interface.

## Entity Integration
The complete flow demonstrates how entities collaborate to achieve the framework's goals:

1. **ArtifactPlan** subclasses specify artifacts of interest through subclass hooks.
2. **ArtifactCallbackFactory** creates callbacks wrapping `artifact-core` computation.
3. **CallbackHandlers** orchestrate callback execution workflows.
4. **Callbacks** perform computations and write results to the `TrackingQueue` through queue writers,
5. **TrackingClients** own the queue and the `BackendLoggingWorker` that consumes it,
6. **ArtifactLoggers** (held by the worker) handle export to tracking backend services,
7. **RunAdapters** normalize tracking backend service interfaces, performing the final native calls.
