# Core Entities

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## Entities by Layer

[`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch) delivers on its objective through the coordinated interaction of specialized abstractions across its [four architectural layers](architecture.md):

### User Implementation Layer

- **Model I/O Types**: Type-safe contracts using `ModelInput` and `ModelOutput` TypedDict classes (exposed via `artifact_torch.nn`) that specify exactly what flows through models during training, enabling static type checking and callback compatibility verification. `ModelOutput` carries an optional `t_loss` entry: the training loop reads `model_output["t_loss"]` to drive backpropagation.

- **Model Interfaces**: The generic `Model[ModelInputT, ModelOutputT]` base (exposed via `artifact_torch.nn`) and domain-specific interfaces (e.g. `TableSynthesizer` from `artifact_torch.table_comparison`, `BinaryClassifier` from `artifact_torch.binary_classification`) that define contracts for model integration with the training framework. Researchers extend these interfaces and implement required methods for training and validation.

- **Data Abstractions**: Type-safe wrappers around PyTorch's data primitives with enhanced functionality, including the generic `Dataset[ModelInputT]` wrapper and an enhanced `DataLoader` with automatic device management (both exposed via `artifact_torch.nn`).

```python
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from artifact_torch.nn import ModelInput, ModelOutput
from artifact_torch.table_comparison import GenerationParams, TableSynthesizer


class TabularVAEInput(ModelInput):
    t_features: torch.Tensor


class TabularVAEOutput(ModelOutput):
    t_loss: Optional[torch.Tensor]


@dataclass
class TabularVAEGenerationParams(GenerationParams):
    n_records: int
    temperature: float


class TabularVAESynthesizer(
    TableSynthesizer[TabularVAEInput, TabularVAEOutput, TabularVAEGenerationParams]
):
    def forward(self, model_input: TabularVAEInput) -> TabularVAEOutput:
        ...

    def generate(self, params: TabularVAEGenerationParams) -> pd.DataFrame:
        ...
```

### User Configuration Layer

- **Experiment**: The top-level orchestrator (base class in `artifact_torch.spi`, domain specializations `TabularSynthesisExperiment` and `BinaryClassificationExperiment` in the respective domain packages). Users subclass a domain experiment and declare—via classmethod hooks—which trainer and routine *classes* make up the workflow: `_get_trainer`, `_get_train_diagnostics_routine`, `_get_loader_routine`, `_get_artifact_routine`. The framework builds and wires all components in `Experiment.build(...)`; training runs with `experiment.run()`.

```python
from typing import Any, Optional, Type

from artifact_torch.table_comparison import (
    TableComparisonRoutine,
    TableSynthesizer,
    TabularSynthesisExperiment,
)


class TabularVAEExperiment(
    TabularSynthesisExperiment[
        TableSynthesizer[Any, TabularVAEOutput, TabularVAEGenerationParams],
        TabularVAEInput,
        TabularVAEOutput,
        TabularVAEGenerationParams,
    ]
):
    # Return annotations elided for brevity; see the user guide for full signatures.
    @classmethod
    def _get_trainer(cls):
        return TabularVAETrainer

    @classmethod
    def _get_train_diagnostics_routine(cls):
        return TabularVAETrainDiagnosticsRoutine

    @classmethod
    def _get_loader_routine(cls):
        return TabularVAELoaderRoutine

    @classmethod
    def _get_artifact_routine(
        cls,
    ) -> Optional[Type[TableComparisonRoutine[TabularVAEGenerationParams]]]:
        return TabularVAEComparisonRoutine
```

- **Trainer**: Orchestrates the complete training process while providing configuration hooks for declarative customization (exposed via `artifact_torch.nn`). It is generic in `[ModelT, ModelInputT, ModelOutputT, StopperUpdateDataT, ModelTrackingCriterionT]`. Users implement hook methods for optimizer selection, scheduling, device placement, checkpointing period, early stopping and model tracking, while the framework handles training loop execution, device management, and gradient computation. Routines are *not* configured on the trainer: they are injected via `Trainer.build(..., train_diagnostics_routine=..., loader_routine=..., artifact_routine=..., file_writer=...)`—typically by the `Experiment`—and grouped internally in a `RoutineSuite`.

```python
from typing import Any, Optional

import torch
from artifact_torch.nn import Trainer
from artifact_torch.nn.early_stopping import EarlyStopper, EpochBoundStopper, StopperUpdateData
from artifact_torch.nn.model_tracking import ModelTracker, ModelTrackingCriterion
from artifact_torch.table_comparison import TableSynthesizer
from torch import optim


class TabularVAETrainer(
    Trainer[
        TableSynthesizer[Any, Any, Any],  # Expected model type.
        TabularVAEInput,  # Expected forward pass input.
        TabularVAEOutput,  # Expected forward pass output.
        StopperUpdateData,  # Data consumed by the early stopper.
        ModelTrackingCriterion,  # Criterion consumed by the model tracker.
    ]
):
    @staticmethod
    def _get_optimizer(model: TableSynthesizer[Any, Any, Any]) -> optim.Optimizer:
        return optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    @staticmethod
    def _get_scheduler(optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        _ = optimizer

    @staticmethod
    def _get_device() -> torch.device:
        return DEVICE

    @staticmethod
    def _get_checkpoint_period() -> Optional[int]:
        return CHECKPOINT_PERIOD

    @staticmethod
    def _get_early_stopper() -> EarlyStopper[StopperUpdateData]:
        return EpochBoundStopper(max_n_epochs=MAX_N_EPOCHS)

    def _get_stopper_update_data(self) -> StopperUpdateData:
        return StopperUpdateData(n_epochs_elapsed=self.n_epochs_elapsed)

    @staticmethod
    def _get_model_tracker() -> Optional[ModelTracker[ModelTrackingCriterion]]:
        pass

    def _get_model_tracking_criterion(self) -> Optional[ModelTrackingCriterion]:
        pass
```

- **Validation Routines**: Validation workflow executors that integrate into the training pipeline (exposed via `artifact_torch.nn.routines` and `artifact_torch.spi`):
  - **TrainDiagnosticsRoutine**: monitors the training loop itself—its plans attach to the model during training batches (model I/O, forward hooks, backward hooks) and execute at epoch end. Users implement `_get_model_io_plan()`, `_get_forward_hook_plan()`, `_get_backward_hook_plan()`, each returning a plan class (or `None`).
  - **DataLoaderRoutine**: post-epoch callback execution on prescribed data loaders (e.g. train/validation splits). Built with `DataSplit`-keyed data loaders; users implement `_get_model_io_plan(data_split)` and `_get_forward_hook_plan(data_split)`, returning plan classes per split.
  - **ArtifactRoutine**: periodic execution of domain-specific validation workflows provided by `artifact-core` (e.g. `TableComparisonRoutine`, `BinaryClassificationRoutine`). Users implement `_get_period(data_split)` and `_get_artifact_plan(data_split)` (returning an artifact plan *class*, e.g. a `TableComparisonPlan` subclass), plus domain hooks such as `_get_generation_params()`. Routines are built via `.build(data=..., data_spec=..., tracking_queue=...)` with `DataSplit`-keyed data.

```python
from typing import Any, Optional, Type

from artifact_experiment.tracking import DataSplit
from artifact_torch.nn import Model
from artifact_torch.nn.plans import ForwardHookPlan, ModelIOPlan
from artifact_torch.nn.routines import DataLoaderRoutine


class TabularVAELoaderRoutine(
    DataLoaderRoutine[Model[Any, TabularVAEOutput], TabularVAEInput, TabularVAEOutput]
):
    @classmethod
    def _get_model_io_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[ModelIOPlan[TabularVAEInput, TabularVAEOutput]]]:
        if data_split is DataSplit.TRAIN:
            return TabularVAEModelIOPlan

    @classmethod
    def _get_forward_hook_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[ForwardHookPlan[Model[Any, Any]]]]:
        if data_split is DataSplit.TRAIN:
            return TabularVAEForwardHookPlan
```

- **Plans**: Declarative groupings of callbacks (exposed via `artifact_torch.nn.plans`): `ModelIOPlan[ModelInputT, ModelOutputT]`, `ForwardHookPlan[ModelT]` and `BackwardHookPlan[ModelT]`. Users implement classmethod hooks (`_get_score_callbacks`, `_get_array_callbacks`, `_get_plot_callbacks` and their collection counterparts) that receive a build context exposing tracking writers (`context.score_writer`, `context.plot_writer`, ...). Plans are returned as *classes* from routine hooks; the framework instantiates them with the appropriate context.

```python
from typing import List

from artifact_torch.nn.callbacks.model_io import LossCallback, ModelIOScoreCallback
from artifact_torch.nn.plans import ModelIOPlan, ModelIOPlanBuildContext


class TabularVAEModelIOPlan(ModelIOPlan[TabularVAEInput, TabularVAEOutput]):
    @classmethod
    def _get_score_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCallback[TabularVAEInput, TabularVAEOutput]]:
        return [LossCallback(period=TRAIN_LOADER_ROUTINE_PERIOD, writer=context.score_writer)]

    # ... analogous hooks for arrays, plots and their collections ...
```

### Framework Infrastructure Layer

- **Callback System**: Type-aware execution hooks that inject custom behavior at specific training points (exposed via `artifact_torch.nn.callbacks`, with `model_io`, `forward_hook` and `backward_hook` submodules). Callbacks use variance-based type parameters to ensure compatibility with model I/O types through static analysis. Ready-made implementations include `LossCallback` (averages `t_loss` over batches) and `AllActivationsPDF` (plots activation distributions via forward hooks).

- **RoutineSuite**: Internal grouping of the injected routines. The trainer executes the suite after each epoch and merges the resulting scores, arrays, plots and collections.

- **Training Infrastructure Components**: Automatic systems that operate behind the scenes:
  - **Device Management**: Automatic tensor placement and device coordination.
  - **In-Memory Caching**: The `ScoreCache` (an `AlignedCache` specialization) accumulates per-epoch validation scores, exposed as a dataframe via `trainer.epoch_scores`/`experiment.epoch_scores`.
  - **Early Stopping**: Configurable training termination (`artifact_torch.nn.early_stopping`): the `EarlyStopper[StopperUpdateDataT]` base plus implementations such as `EpochBoundStopper(max_n_epochs=...)`, `ScoreMinimizationStopper` and `ScoreMaximizationStopper`.
  - **Model Tracking**: Best-model state management (`artifact_torch.nn.model_tracking`): the `ModelTracker[ModelTrackingCriterionT]` base with implementations such as `SingleScoreTracker`; best state is exposed via `trainer.best_model_state`.
  - **Checkpointing**: Users only implement `_get_checkpoint_period() -> Optional[int]` on the trainer; the framework builds the `CheckpointCallback` itself when a file writer is available (i.e. when a tracking client is passed to the experiment).

### External Integration Layer

- **Integration with `artifact-core`**: Automatic validation artifact computation during training through artifact routines that coordinate with the appropriate Artifact-ML [domain toolkit](https://artifact-ml.readthedocs.io/en/latest/artifact-core/domain_toolkits/). Artifact plans (e.g. `TableComparisonPlan` subclasses) are declared as classes and built by the framework against the experiment's resource spec.

- **Integration with `artifact-experiment`**: Experiment tracking using popular backend services (e.g. [MLflow](https://mlflow.org/), [ClearML](https://clear.ml/), [Neptune](https://neptune.ai/)) or simple filesystem/ in-memory caching. Results flow from callbacks and artifact plans through the tracking queue to the configured tracking client.
