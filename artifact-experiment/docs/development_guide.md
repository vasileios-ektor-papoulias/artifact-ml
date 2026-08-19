# Development Guide

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>


## Creating Artifact Plans for New Domains

Each artifact engine in `artifact-core` should have a corresponding plan in `artifact-experiment`. Plans derive from the generic `ArtifactPlan` base class, exposed via `artifact_experiment.spi.plans`; the shipped domain plans (`TableComparisonPlan` in `artifact_experiment.table_comparison` and `BinaryClassificationPlan` in `artifact_experiment.binary_classification`) are the reference implementations to study.

When contributing new artifact types to `artifact-core`, extend `artifact-experiment` with the corresponding plan:

```python
from typing import Type

from artifact_experiment.spi.plans import ArtifactPlan

class NewDomainPlan(ArtifactPlan[...]):
    @staticmethod
    def _get_callback_factory() -> Type[NewDomainCallbackFactory]:
        return NewDomainCallbackFactory

    # Provide a domain-specific execution entrypoint delegating to
    # execute_artifacts (cf. TableComparisonPlan.execute_table_comparison).

    # Implement the export hooks:
    # _get_export_callback(context) and _get_export_resources(resources).
```

The six artifact type getters (`_get_score_types`, `_get_array_types`, `_get_plot_types`, `_get_score_collection_types`, `_get_array_collection_types`, `_get_plot_collection_types`) remain abstract on the domain plan: they are the hooks through which end users declare the artifacts they want. The callback factory referenced above is an `ArtifactCallbackFactory` subclass (cf. `TableComparisonCallbackFactory`) that builds the callbacks wrapping `artifact-core` computation.

Users instantiate the finished plan with `NewDomainPlan.create(resource_spec=..., tracking_client=...)`; the `build(context=...)` classmethod it delegates to is framework-internal.

## Adding New Tracking Backends

To support a new experiment tracking backend, implement four pieces, all extending the interfaces exposed by `artifact_experiment.tracking.spi`:

1. **RunAdapter**: Normalize the backend's native run object
2. **BackendLoggers**: Handle backend-specific artifact export
3. **BackendLoggingWorker**: Wire one logger per artifact kind
4. **TrackingClient**: Expose the unified tracking interface

The MLflow backend (`artifact_experiment/_impl/backends/mlflow/`) is the reference implementation for this recipe; the sketches below follow its structure.

```python
import os
from typing import Optional

from artifact_core.typing import Score

from artifact_experiment.tracking.spi import (
    ArtifactLogger,
    BackendLogger,
    BackendLoggingWorker,
    RunAdapter,
    TrackingClient,
    TrackingQueue,
)

# 1. Create RunAdapter: wraps the backend's native run object
class MyBackendRunAdapter(RunAdapter[MyNativeRun]):
    @property
    def experiment_id(self) -> str: ...

    @property
    def run_id(self) -> str: ...

    @property
    def is_active(self) -> bool: ...

    def stop(self):
        # Terminate the native run
        ...

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> MyNativeRun:
        # Create or attach the native run; create the experiment
        # here if it doesn't exist (cf. MlflowRunAdapter._create_experiment)
        ...

# 2. Implement BackendLoggers: one per artifact kind
class MyBackendScoreLogger(BackendLogger[MyBackendRunAdapter, Score]):
    def _append(self, item_path: str, item: Score):
        # Backend-specific export through self._run
        ...

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("scores", item_name)

    def _get_root_dir(self) -> str:
        return "artifacts"

# 3. Create BackendLoggingWorker: exposes the logger getters
class MyBackendLoggingWorker(BackendLoggingWorker[MyBackendRunAdapter]):
    @staticmethod
    def _get_score_logger(
        run: MyBackendRunAdapter,
    ) -> ArtifactLogger[MyBackendRunAdapter, Score]:
        return MyBackendScoreLogger(run=run)

    # Implement the remaining getters: _get_array_logger, _get_plot_logger,
    # _get_score_collection_logger, _get_array_collection_logger,
    # _get_plot_collection_logger, _get_file_logger...

# 4. Create TrackingClient: the unified user-facing entrypoint
class MyBackendTrackingClient(TrackingClient[MyBackendRunAdapter]):
    @classmethod
    def build(
        cls, experiment_id: str, run_id: Optional[str] = None
    ) -> "MyBackendTrackingClient":
        run = MyBackendRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        return cls._build(run=run)

    @classmethod
    def from_native_run(cls, native_run: MyNativeRun) -> "MyBackendTrackingClient":
        run = MyBackendRunAdapter.from_native_run(native_run=native_run)
        return cls._build(run=run)

    @staticmethod
    def _get_worker(
        run: MyBackendRunAdapter, tracking_queue: TrackingQueue
    ) -> BackendLoggingWorker[MyBackendRunAdapter]:
        return MyBackendLoggingWorker.build(
            run=run, queue=tracking_queue.queue, temp_dir=tracking_queue.temp_dir
        )
```

Key points to keep in mind:

- Loggers are generic over `[RunAdapterT, ArtifactResultT]` (adapter first) and their public method is `log(item_name: str, item: ...)`; `ArtifactLogger` is `BackendLogger` constrained to artifact results, while the file logger returned by `_get_file_logger` is a `BackendLogger[..., File]`.
- Logger getters live on the `BackendLoggingWorker` subclass, not on the client; the client's only abstract hook is `_get_worker(run, tracking_queue)`.
- `RunAdapter` performs export through backend-specific methods invoked by the loggers (there is no generic `upload` hook on the base class); run and experiment creation belong in `_build_native_run`.
- The base `TrackingClient` already provides the queue writers, the `log_*` convenience methods, and `stop()` (which stops the worker and the run) — backend subclasses don't reimplement these.
