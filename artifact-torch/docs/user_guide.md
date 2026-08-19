# User Guide

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## End to End Demo Projects

For comprehensive usage examples and detailed implementation patterns, refer to our end-to-end demo projects:

- [synthetic tabular data demo project](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch/demos/table_comparison),
- [binary classification demo project](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch/demos/binary_classification).


## Building a Project with Artifact-Torch

This section provides a step-by-step guide for building your own deep learning project using [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch).

We use **tabular data synthesis** as the running example, mirroring the [table comparison demo](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch/demos/table_comparison).

### Suggested Project Organization

```
project_root/
├── contracts/
│   └── model.py                # ModelInput/ModelOutput/GenerationParams contracts
├── model/
│   ├── synthesizer.py          # Framework interface implementation
│   └── architectures/          # Neural network implementations
├── data/
│   └── dataset.py              # Type-safe dataset implementation
├── components/
│   ├── plans/
│   │   ├── artifact.py         # Artifact plan (validation artifact selection)
│   │   ├── model_io.py         # Model I/O callback plan
│   │   └── forward_hook.py     # Forward hook callback plan
│   └── routines/
│       ├── artifact.py         # Artifact routine configuration
│       ├── loader.py           # DataLoader routine configuration
│       └── train_diagnostics.py # Train diagnostics routine configuration
├── trainer/
│   └── trainer.py              # Trainer extension
├── experiment/
│   └── experiment.py           # Experiment orchestration
└── config/
    └── configuration files
```

### Step 1: Application Domain (Domain Toolkit) Selection

The **first step** is identifying your ML task and checking if [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch) provides a domain toolkit to support it.

Currently, supported domains include:

- **tabular data synthesis** (used as an example in this guide): `artifact_torch.table_comparison`,
- **binary classification**: `artifact_torch.binary_classification`.

For each supported domain, [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch) provides three **core interfaces** for you to implement:

- model: your model architecture,
- artifact routine: domain-specific validation workflows periodically injected into the training loop,
- experiment: the top-level orchestrator wiring your trainer and routines together.

For **tabular data synthesis**, these interfaces are:

- `TableSynthesizer` for models,
- `TableComparisonRoutine` for validation,
- `TabularSynthesisExperiment` for orchestration.

### Step 2: Model Input/Output Type Specification

**What you need to do**: Define strict type contracts for your model's forward pass signature.

Suggested directory: `contracts/model.py`

```python
from dataclasses import dataclass
from typing import List, Optional

import torch
from artifact_torch.nn import ModelInput, ModelOutput
from artifact_torch.table_comparison import GenerationParams


class TabularVAEInput(ModelInput):
    t_features: torch.Tensor


class TabularVAEOutput(ModelOutput):
    ls_t_logits: List[torch.Tensor]
    t_latent_mean: torch.Tensor
    t_latent_log_var: torch.Tensor
    t_loss: Optional[torch.Tensor]


@dataclass
class TabularVAEGenerationParams(GenerationParams):
    n_records: int
    temperature: float
```

`ModelInput` and `ModelOutput` are TypedDict contracts. `ModelOutput` requires an optional `t_loss` entry: the training loop reads `model_output["t_loss"]` to drive backpropagation, so your model must populate it during training.

Keep in mind that all experiment workflows you'll build later on will be type-aware (with variance) and able to detect compatibility with your model given the above type contract specifications.

For this reason, it's beneficial to keep type requirements as lenient as possible (e.g. few standard inputs, many outputs).

Doing so expands the space of compatible workflows.

`GenerationParams` is the (empty) base contract for generation hyperparameters: extend it with the minimal set of standard parameters your synthesizer needs (here `n_records` and `temperature`).

### Step 3: Model Implementation

**What you need to do**: Implement the model interface for your specific architecture while respecting the IO contracts.

The `TableSynthesizer` interface is generic in `[ModelInputT, ModelOutputT, GenerationParamsT]` and requires two methods:

- `forward(self, model_input: ...) -> ...`: the training forward pass, consuming your `ModelInput` and producing your `ModelOutput` (with `t_loss` populated),
- `generate(self, params: ...) -> pd.DataFrame`: synthetic data generation, consumed by the artifact routine during validation.

Suggested directory: `model/synthesizer.py`

```python
import pandas as pd
from artifact_torch.table_comparison import TableSynthesizer

from project.contracts.model import (
    TabularVAEGenerationParams,
    TabularVAEInput,
    TabularVAEOutput,
)


class TabularVAESynthesizer(
    TableSynthesizer[TabularVAEInput, TabularVAEOutput, TabularVAEGenerationParams]
):
    def __init__(self, vae: VariationalAutoencoder, discretizer: Discretizer, encoder: Encoder):
        super().__init__()
        self._vae = vae                  # Your actual neural network
        self._discretizer = discretizer  # Your preprocessing
        self._encoder = encoder          # Your encoding logic

    def forward(self, model_input: TabularVAEInput) -> TabularVAEOutput:
        t_features = model_input.get("t_features")
        ls_t_logits, t_latent_mean, t_latent_log_var, t_loss = self._vae(t_features=t_features)
        return TabularVAEOutput(
            ls_t_logits=ls_t_logits,
            t_latent_mean=t_latent_mean,
            t_latent_log_var=t_latent_log_var,
            t_loss=t_loss,
        )

    def generate(self, params: TabularVAEGenerationParams) -> pd.DataFrame:
        self.eval()
        t_preds = self._vae.generate(
            n_records=params["n_records"],
            temperature=params["temperature"],
            device=self.device,
        )
        return self._postprocess_generated_data(t_preds)
```

### Step 4: Dataset Implementation

**What you need to do**: Implement the pipeline responsible for preparing individual training samples.

This is achieved by extending the Artifact-ML `Dataset` interface (type-aware torch-native Dataset wrapper) while respecting the expected type contracts.

Suggested directory: `data/dataset.py`

```python
import pandas as pd
import torch
from artifact_torch.nn import Dataset

from project.contracts.model import TabularVAEInput


class TabularVAEDataset(Dataset[TabularVAEInput]):
    def __init__(self, df_raw: pd.DataFrame, discretizer: Discretizer, encoder: Encoder):
        df_discretized = discretizer.transform(df=df_raw)
        df_encoded = encoder.transform(df=df_discretized)
        self._t_data = torch.tensor(df_encoded.values, dtype=torch.float32)

    def __len__(self) -> int:
        return self._t_data.size(0)

    def __getitem__(self, idx: int) -> TabularVAEInput:
        row = self._t_data[idx]
        return TabularVAEInput(t_features=row)
```

**Using Dataset with DataLoader**: Once you've implemented your dataset, you can use it with an Artifact-ML `DataLoader` (type-aware torch-native DataLoader wrapper) for batch preparation:

```python
from artifact_torch.nn import DataLoader

loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    drop_last=DROP_LAST,
    shuffle=SHUFFLE,
)
```

### Step 5: Artifact Plan Specification

**What you need to do**: Select the artifact collection you'd like to track as training progresses. Your customized plan will configure the artifact routine injected into the training loop---see step 6.

Suggested directory: `components/plans/artifact.py`

```python
from typing import List

from artifact_core.table_comparison import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)
from artifact_experiment.table_comparison import TableComparisonPlan


class TabularVAEComparisonPlan(TableComparisonPlan):
    @staticmethod
    def _get_score_types() -> List[TableComparisonScoreType]:
        return [
            TableComparisonScoreType.MEAN_JS_DISTANCE,
            TableComparisonScoreType.CORRELATION_DISTANCE,
        ]

    @staticmethod
    def _get_array_types() -> List[TableComparisonArrayType]:
        return []

    @staticmethod
    def _get_plot_types() -> List[TableComparisonPlotType]:
        return [
            TableComparisonPlotType.PDF,
            TableComparisonPlotType.CDF,
            TableComparisonPlotType.DESCRIPTIVE_STATS_ALIGNMENT,
            TableComparisonPlotType.PCA_JUXTAPOSITION,
            TableComparisonPlotType.CORRELATION_HEATMAP_JUXTAPOSITION,
        ]

    @staticmethod
    def _get_score_collection_types() -> List[TableComparisonScoreCollectionType]:
        return [
            TableComparisonScoreCollectionType.JS_DISTANCE,
        ]

    @staticmethod
    def _get_array_collection_types() -> List[TableComparisonArrayCollectionType]:
        return [
            TableComparisonArrayCollectionType.MEAN_JUXTAPOSITION,
            TableComparisonArrayCollectionType.STD_JUXTAPOSITION,
            TableComparisonArrayCollectionType.MIN_JUXTAPOSITION,
            TableComparisonArrayCollectionType.MAX_JUXTAPOSITION,
        ]

    @staticmethod
    def _get_plot_collection_types() -> List[TableComparisonPlotCollectionType]:
        return [
            TableComparisonPlotCollectionType.PDF,
            TableComparisonPlotCollectionType.CDF,
        ]
```

Note that you only declare the plan *class*: the framework builds it internally (against the experiment's resource spec and tracking queue) when the artifact routine is constructed. There is no user-side `.build(...)` call.

### Step 6: Routine Configuration

**What you need to do**: Configure all validation hooks you would like to inject into your training loop:

- `ArtifactRoutine`: domain-specific validation flow (in the context of tabular data synthesis this is the `TableComparisonRoutine`),
- `DataLoaderRoutine`: post-epoch per-data-loader monitoring,
- `TrainDiagnosticsRoutine`: monitoring of the training loop itself (model I/O, forward/backward hooks on training batches).

All routines are declared as classes and instantiated by the framework. Routine hooks are keyed by `DataSplit` (`TRAIN`, `VALIDATION`, `TEST`, `ALL` from `artifact_experiment.tracking`) where applicable, so the same routine class can prescribe different behavior per data split.

**Artifact Routine Configuration** (Suggested directory: `components/routines/artifact.py`)

```python
from typing import Optional, Type

from artifact_experiment.tracking import DataSplit
from artifact_torch.table_comparison import TableComparisonPlan, TableComparisonRoutine

from project.components.plans.artifact import TabularVAEComparisonPlan
from project.contracts.model import TabularVAEGenerationParams


class TabularVAEComparisonRoutine(TableComparisonRoutine[TabularVAEGenerationParams]):
    @classmethod
    def _get_period(cls, data_split: DataSplit) -> Optional[int]:
        # Configure how often validation runs (per data split; None disables the split)
        if data_split is DataSplit.TRAIN:
            return ARTIFACT_ROUTINE_PERIOD

    @classmethod
    def _get_generation_params(cls) -> TabularVAEGenerationParams:
        # Configure how to generate data for validation
        return TabularVAEGenerationParams(
            n_records=GENERATION_N_RECORDS, temperature=GENERATION_TEMPERATURE
        )

    @classmethod
    def _get_artifact_plan(cls, data_split: DataSplit) -> Optional[Type[TableComparisonPlan]]:
        # Configure which artifact plan (class) to use per data split
        if data_split is DataSplit.TRAIN:
            return TabularVAEComparisonPlan
```

**Model I/O Plan Configuration** (Suggested directory: `components/plans/model_io.py`)

`DataLoaderRoutine` and `TrainDiagnosticsRoutine` don't hold callbacks directly: they execute *plans* (`ModelIOPlan`, `ForwardHookPlan`, `BackwardHookPlan`), each grouping callbacks of a given kind. Plan hooks receive a build context exposing tracking writers.

```python
from typing import List

from artifact_torch.nn.callbacks.model_io import (
    LossCallback,
    ModelIOArrayCallback,
    ModelIOArrayCollectionCallback,
    ModelIOPlotCallback,
    ModelIOPlotCollectionCallback,
    ModelIOScoreCallback,
    ModelIOScoreCollectionCallback,
)
from artifact_torch.nn.plans import ModelIOPlan, ModelIOPlanBuildContext

from project.contracts.model import TabularVAEInput, TabularVAEOutput


class TabularVAEModelIOPlan(ModelIOPlan[TabularVAEInput, TabularVAEOutput]):
    @classmethod
    def _get_score_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCallback[TabularVAEInput, TabularVAEOutput]]:
        return [LossCallback(period=TRAIN_LOADER_ROUTINE_PERIOD, writer=context.score_writer)]

    @classmethod
    def _get_array_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCallback[TabularVAEInput, TabularVAEOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCallback[TabularVAEInput, TabularVAEOutput]]:
        _ = context
        return []

    @classmethod
    def _get_score_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCollectionCallback[TabularVAEInput, TabularVAEOutput]]:
        _ = context
        return []

    @classmethod
    def _get_array_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCollectionCallback[TabularVAEInput, TabularVAEOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCollectionCallback[TabularVAEInput, TabularVAEOutput]]:
        _ = context
        return []
```

**Forward Hook Plan Configuration** (Suggested directory: `components/plans/forward_hook.py`)

```python
from typing import Any, Sequence

from artifact_torch.nn import Model
from artifact_torch.nn.callbacks.forward_hook import (
    AllActivationsPDF,
    ForwardHookPlotCallback,
    ForwardHookScoreCallback,
)
from artifact_torch.nn.plans import ForwardHookPlan, ForwardHookPlanBuildContext


class TabularVAEForwardHookPlan(ForwardHookPlan[Model[Any, Any]]):
    @classmethod
    def _get_plot_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookPlotCallback[Model[Any, Any]]]:
        return [AllActivationsPDF(period=TRAIN_LOADER_ROUTINE_PERIOD, writer=context.plot_writer)]

    @classmethod
    def _get_score_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookScoreCallback[Model[Any, Any]]]:
        _ = context
        return []

    # ... analogous hooks for arrays and collections return [] ...
```

**Data Loader Routine Configuration** (Suggested directory: `components/routines/loader.py`)

```python
from typing import Any, Optional, Type

from artifact_experiment.tracking import DataSplit
from artifact_torch.nn import Model
from artifact_torch.nn.plans import ForwardHookPlan, ModelIOPlan
from artifact_torch.nn.routines import DataLoaderRoutine

from project.components.plans.forward_hook import TabularVAEForwardHookPlan
from project.components.plans.model_io import TabularVAEModelIOPlan
from project.contracts.model import TabularVAEInput, TabularVAEOutput


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

**Train Diagnostics Routine Configuration** (Suggested directory: `components/routines/train_diagnostics.py`)

Unlike the data loader routine (which re-runs the model over prescribed loaders after each epoch), the train diagnostics routine attaches its plans to the model *during* training batches and reports at epoch end. Its hooks are not split-keyed.

```python
from typing import Any, Optional, Type

from artifact_torch.nn import Model
from artifact_torch.nn.plans import BackwardHookPlan, ForwardHookPlan, ModelIOPlan
from artifact_torch.nn.routines import TrainDiagnosticsRoutine

from project.components.plans.model_io import TabularVAEModelIOPlan
from project.contracts.model import TabularVAEInput, TabularVAEOutput


class TabularVAETrainDiagnosticsRoutine(
    TrainDiagnosticsRoutine[Model[Any, TabularVAEOutput], TabularVAEInput, TabularVAEOutput]
):
    @classmethod
    def _get_model_io_plan(cls) -> Optional[Type[ModelIOPlan[TabularVAEInput, TabularVAEOutput]]]:
        return TabularVAEModelIOPlan

    @classmethod
    def _get_forward_hook_plan(cls) -> Optional[Type[ForwardHookPlan[Model[Any, Any]]]]:
        pass

    @classmethod
    def _get_backward_hook_plan(cls) -> Optional[Type[BackwardHookPlan[Model[Any, Any]]]]:
        pass
```

**Custom Callback Development**: For project-specific requirements, you can create custom callbacks tailored to your model's I/O profile by extending the appropriate base callback classes from `artifact_torch.nn.callbacks`. These custom callbacks seamlessly integrate with existing framework callbacks compatible with your model's I/O types, giving you both flexibility and access to the full ecosystem of pre-built functionality.

### Step 7: Trainer Configuration

**What you need to do**: Configure the trainer by extending `Trainer` and implementing its hook methods.

The trainer exposes configurable hooks governing core aspects of the training lifecycle:

- **Optimization Setup**  
  Configure standard PyTorch training components, including:
  - Optimizer selection and hyperparameters
  - Learning-rate scheduler policy
  - Device placement

- **Early Stopping & Model Selection**  
  Specify termination criteria based on validation signals and track the best-performing model state for subsequent use.

- **Model State Monitoring (Checkpointing)**  
  Implement `_get_checkpoint_period()` to enable periodic checkpointing; the framework builds the checkpoint callback itself when a tracking client (file writer) is available.

Note that routines are *not* configured on the trainer: they are injected via `Trainer.build(...)` by the experiment (see step 8).

Suggested directory: `trainer/trainer.py`

```python
from typing import Any, Optional

import torch
from artifact_torch.nn import Trainer
from artifact_torch.nn.early_stopping import EarlyStopper, EpochBoundStopper, StopperUpdateData
from artifact_torch.nn.model_tracking import ModelTracker, ModelTrackingCriterion
from artifact_torch.table_comparison import TableSynthesizer
from torch import optim

from project.contracts.model import TabularVAEInput, TabularVAEOutput


class TabularVAETrainer(
    Trainer[
        TableSynthesizer[Any, Any, Any],
        TabularVAEInput,
        TabularVAEOutput,
        StopperUpdateData,
        ModelTrackingCriterion,
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
    def _get_model_tracker() -> Optional[ModelTracker[ModelTrackingCriterion]]:
        pass

    def _get_model_tracking_criterion(self) -> Optional[ModelTrackingCriterion]:
        pass

    @staticmethod
    def _get_early_stopper() -> EarlyStopper[StopperUpdateData]:
        return EpochBoundStopper(max_n_epochs=MAX_N_EPOCHS)

    def _get_stopper_update_data(self) -> StopperUpdateData:
        return StopperUpdateData(n_epochs_elapsed=self.n_epochs_elapsed)
```

### Step 8: Experiment Orchestration

**What you need to do**: Tie everything together by extending the domain experiment and declaring—via classmethod hooks—the trainer and routine classes that make up your workflow.

Suggested directory: `experiment/experiment.py`

```python
from typing import Any, Optional, Type

from artifact_torch.nn import Trainer
from artifact_torch.nn.routines import DataLoaderRoutine, TrainDiagnosticsRoutine
from artifact_torch.table_comparison import (
    TableComparisonRoutine,
    TableSynthesizer,
    TabularSynthesisExperiment,
)

from project.components.routines.artifact import TabularVAEComparisonRoutine
from project.components.routines.loader import TabularVAELoaderRoutine
from project.components.routines.train_diagnostics import TabularVAETrainDiagnosticsRoutine
from project.contracts.model import (
    TabularVAEGenerationParams,
    TabularVAEInput,
    TabularVAEOutput,
)
from project.trainer.trainer import TabularVAETrainer


class TabularVAEExperiment(
    TabularSynthesisExperiment[
        TableSynthesizer[Any, TabularVAEOutput, TabularVAEGenerationParams],
        TabularVAEInput,
        TabularVAEOutput,
        TabularVAEGenerationParams,
    ]
):
    @classmethod
    def _get_trainer(
        cls,
    ) -> Type[
        Trainer[
            TableSynthesizer[Any, TabularVAEOutput, TabularVAEGenerationParams],
            TabularVAEInput,
            TabularVAEOutput,
            Any,
            Any,
        ]
    ]:
        return TabularVAETrainer

    @classmethod
    def _get_train_diagnostics_routine(
        cls,
    ) -> Optional[
        Type[
            TrainDiagnosticsRoutine[
                TableSynthesizer[Any, TabularVAEOutput, TabularVAEGenerationParams],
                TabularVAEInput,
                TabularVAEOutput,
            ]
        ]
    ]:
        return TabularVAETrainDiagnosticsRoutine

    @classmethod
    def _get_loader_routine(
        cls,
    ) -> Optional[
        Type[
            DataLoaderRoutine[
                TableSynthesizer[Any, TabularVAEOutput, TabularVAEGenerationParams],
                TabularVAEInput,
                TabularVAEOutput,
            ]
        ]
    ]:
        return TabularVAELoaderRoutine

    @classmethod
    def _get_artifact_routine(
        cls,
    ) -> Optional[Type[TableComparisonRoutine[TabularVAEGenerationParams]]]:
        return TabularVAEComparisonRoutine
```

### Step 9: Running the Experiment

**What you need to do**: Build the experiment with your model, `DataSplit`-keyed data, and an optional tracking client—then run it.

In the snippet below, `vae`, `discretizer` and `encoder` are your own architecture and preprocessing components (see step 3), and the uppercase names are project configuration constants.

```python
import pandas as pd
from artifact_experiment.tracking import DataSplit, FilesystemTrackingClient
from artifact_torch.nn import DataLoader
from artifact_torch.table_comparison import TableComparisonRoutineData, TabularDataSpec

from project.data.dataset import TabularVAEDataset
from project.experiment.experiment import TabularVAEExperiment
from project.model.synthesizer import TabularVAESynthesizer

df_real = pd.read_csv(TRAINING_DATASET_PATH)

data_spec = TabularDataSpec.from_df(
    df=df_real, cts_features=LS_CTS_FEATURES, cat_features=LS_CAT_FEATURES
)

data_loaders = {
    DataSplit.TRAIN: DataLoader(
        dataset=TabularVAEDataset(df_raw=df_real, discretizer=discretizer, encoder=encoder),
        batch_size=BATCH_SIZE,
        drop_last=DROP_LAST,
        shuffle=SHUFFLE,
    )
}

artifact_routine_data = {DataSplit.TRAIN: TableComparisonRoutineData(df_real=df_real)}

model = TabularVAESynthesizer(vae=vae, discretizer=discretizer, encoder=encoder)

tracking_client = FilesystemTrackingClient.build(experiment_id=EXPERIMENT_ID)

experiment = TabularVAEExperiment.build(
    model=model,
    data_loaders=data_loaders,
    artifact_routine_data=artifact_routine_data,
    artifact_routine_data_spec=data_spec,
    tracking_client=tracking_client,
)

experiment.run()
```

The experiment builds the routines from your declared classes, injects them into the trainer, and executes the training loop. Artifacts and scores are exported to the tracking client as training progresses.

After the run, per-epoch validation scores are available as a dataframe:

```python
experiment.epoch_scores
```
