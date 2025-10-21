# User Guide

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## End to End Demo Projects

For comprehensive usage examples and detailed implementation patterns, refer to our end-to-end demo projects:

- [synthetic tabular data demo project](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch/demos/table_comparison),
- [binary classification demo project](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch/demos/binary_classification).


## Step-by-Step Guide: Building a Project with Artifact-Torch

This section provides an **intuitive step-by-step guide** for building your own project using `artifact-torch`. Each step shows you how to configure the framework's interfaces for your specific use case - essentially, all implementations are configurations that tell the framework how to handle your particular ML task.

### Suggested Project Organization

The following template summarizes the various entities that need to be implemented when building a deep learning project with `artifact-torch`:

```
project_root/
├── model/
│   ├── io.py                    # ModelInput/ModelOutput type definitions
│   ├── model.py                 # Framework interface implementation
│   └── architectures/           # Neural network implementations
├── data/
│   ├── dataset.py              # Type-safe dataset implementation
│   └── preprocessing/          # Data transformation pipeline
├── trainer/
│   └── trainer.py              # CustomTrainer extension
├── routines/
│   ├── artifact.py             # Validation routine configuration
│   ├── batch.py                # Batch-level callback routines
│   └── loader.py               # DataLoader-level callback routines
└── config/
    └── configuration files
```

## Implementation Sequence

### Step 1: Application Domain (Domain Toolkit) Selection

The **first step** is identifying your ML task and checking if `artifact-torch` provides a domain toolkit to support it.

Currently, supported domains include:

- **tabular data synthesis** (used as an example in this demo)
- **binary classification**

For each supported domain, `artifact-torch` provides two **core interfaces** for you to implement:

- model,
- artifact validation routine: domain-specific validation workflows periodically injected into the training loop

For **tabular data synthesis**, these interfaces are:
- `TableSynthesizer` interface for models
- `TableComparisonRoutine` for validation

### Step 2: Model Input/Output Type Specification

**What you need to do**: Define strict type contracts for your model's forward pass signature

Suggested directory: `model/io.py`

```python
import torch

from artifact_torch.base.model.io import ModelInput, ModelOutput

class TabularVAEModelInput(ModelInput):
    batch: torch.Tensor                    # Input tensor for VAE

class TabularVAEModelOutput(ModelOutput):  
    reconstruction: torch.Tensor           # Reconstructed data
    mu: torch.Tensor                      # Latent mean
    logvar: torch.Tensor                  # Latent log variance
    loss: torch.Tensor                    # Computed loss
```
Keep in mind that all experiment workflows you'll build later on will be type-aware (with variance) and able to detect compatibility with your model given the above type contract specifications.

For this reason, it's beneficial to keep type requirements as lenient as possible (e.g. few standard inputs, many outputs).

Doing so expands the space of compatible workflows.

### Step 3: Model Implementation

**What you need to do**: Implement the model interface for your specific architecture while respecting the IO contracts.

In general, implementing one of the model interfaces (in this case the Tableynthesizer interface) might require implementing secondary configuration objects (in this case generation parameters).

Note that the above remark on lenient type requirements applies to these configuration objects as well (e.g. ask for a limited set of standard generation hyperparams like temperature).

**Secondary Configuration Objects** (Suggested directory: `model/synthesizer.py`)
```python
from dataclasses import dataclass

from artifact_torch.core.model.generative import GenerationParams

@dataclass
class TabularVAEGenerationParams(GenerationParams):
    n_records: int
    use_mean: bool = False
    temperature: float = 1.0
    sample: bool = True
```

**Model Implementation** (Suggested directory: `model/synthesizer.py`)
```python
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from artifact_torch.table_comparison.model import TableSynthesizer

class TabularVAESynthesizer(TableSynthesizer):
    def __init__(self, vae_model: VAE, discretizer: Discretizer, encoder: Encoder):
        self._vae_model = vae_model           # Your actual neural network
        self._discretizer = discretizer       # Your preprocessing
        self._encoder = encoder               # Your encoding logic
    
    def generate(self, params: TabularVAEGenerationParams) -> pd.DataFrame:
        # Implement the interface method the framework expects
        return self._postprocess_generated_data(raw_output)
```

### Step 4: Dataset Implementation

**What you need to do**: Implement the pipeline responsible for preparing individual training samples.

This is achieved by extending the Artifact-ML Dataset interface (type-aware torch-native Dataset wrapper) while respecting the expected type contracts.

Suggested directory: `data/dataset.py`.
```python
import pandas as pd
import torch

from artifact_torch.base.data.dataset import Dataset

class TabularVAEDataset(Dataset[TabularVAEModelInput]):
    def __init__(self, df_encoded: pd.DataFrame):
        self._df_encoded = df_encoded
        
    def __getitem__(self, idx: int) -> TabularVAEModelInput:
        # Define how to get individual samples - Dataset just returns single items
        return TabularVAEModelInput(
            batch=torch.tensor(self._df_encoded.iloc[idx].values, dtype=torch.float32)
        )
    
    def __len__(self) -> int:
        return len(self._df_encoded)
```

**Using Dataset with DataLoader**: Once you've implemented your dataset, you can use it with an Artifact-ML DataLoader (type-aware torch-native DataLoader wrapper) for batch preparation:

```python
from artifact_torch.base.data.data_loader import DataLoader

loader = DataLoader(
    dataset=dataset, 
    batch_size=BATCH_SIZE, 
    drop_last=DROP_LAST, 
    shuffle=SHUFFLE
)
```

### Step 5: Validation Plan Specification

**What you need to do**: Select the artifact collection you'd like to track as training progresses. Your customized plan will help configure the `ArtifactRoutine` injected into the training loop---see step 6. 

Suggested directory: `components/validation_plan.py`.

```python
from typing import List

from artifact_experiment.table_comparison.validation_plan import (
    TableComparisonPlan,
    TableComparisonScoreType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonArrayCollectionType,
    TableComparisonPlotCollectionType,
)

class DemoTableComparisonPlan(TableComparisonPlan):
    @staticmethod
    def _get_score_types() -> List[TableComparisonScoreType]:
        return [
            TableComparisonScoreType.MEAN_JS_DISTANCE,
            TableComparisonScoreType.PAIRWISE_CORRELATION_DISTANCE,
        ]

    @staticmethod
    def _get_plot_types() -> List[TableComparisonPlotType]:
        return [
            TableComparisonPlotType.PDF_PLOT,
            TableComparisonPlotType.CDF_PLOT,
            TableComparisonPlotType.DESCRIPTIVE_STATS_COMPARISON_PLOT,
            TableComparisonPlotType.PCA_PROJECTION_PLOT,
            TableComparisonPlotType.PAIRWISE_CORRELATION_COMPARISON_HEATMAP,
        ]

    @staticmethod
    def _get_score_collection_types() -> List[TableComparisonScoreCollectionType]:
        return [
            TableComparisonScoreCollectionType.JS_DISTANCE,
        ]

    @staticmethod
    def _get_array_collection_types() -> List[TableComparisonArrayCollectionType]:
        return [
            TableComparisonArrayCollectionType.MEANS,
            TableComparisonArrayCollectionType.STDS,
            TableComparisonArrayCollectionType.MINIMA,
            TableComparisonArrayCollectionType.MAXIMA,
        ]

    @staticmethod
    def _get_plot_collection_types() -> List[TableComparisonPlotCollectionType]:
        return [
            TableComparisonPlotCollectionType.PDF_PLOTS,
            TableComparisonPlotCollectionType.CDF_PLOTS,
        ]
```

### Step 6: Validation Routine Configuration

**What you need to do**: Configure all validation hooks you would like to inject into your training loop:

- `ArtifactRoutine`: domain specific validation flow (in the context of tabular data synthesis this is the `TableComparisonRoutine`).
- `DataLoaderRoutine`: post-epoch per-data loader monitoring,
- `BatchRoutine`: inter-epoch per-batch monitoring

**Artifact Routine Configuration** (Suggested directory: `components/routines/artifact.py`)
```python
from typing import Optional
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.table_comparison.validation_plan import TableComparisonPlan
from artifact_torch.table_comparison.routine import TableComparisonRoutine

class DemoTableComparisonRoutine(TableComparisonRoutine):
    @classmethod
    def _get_period(cls) -> int:
        # Configure how often validation runs
        return ARTIFACT_VALIDATION_PERIOD

    @classmethod
    def _get_generation_params(cls) -> TabularVAEGenerationParams:
        # Configure how to generate data for validation
        return TabularVAEGenerationParams(
            n_records=GENERATION_N_RECORDS,
            use_mean=GENERATION_USE_MEAN,
            temperature=GENERATION_TEMPERATURE,
            sample=True,
        )

    @classmethod
    def _get_validation_plan(cls, artifact_resource_spec, tracking_client) -> TableComparisonPlan:
        # Configure which validation plan to use
        return DemoTableComparisonPlan.build(
            resource_spec=artifact_resource_spec, tracking_client=tracking_client
        )
```

**Batch Routine Configuration** (Suggested directory: `components/routines/batch.py`)
```python
from typing import List
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.components.callbacks.batch import BatchCallback
from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.libs.components.callbacks.batch.loss import BatchLossCallback

class DemoBatchRoutine(BatchRoutine[TabularVAEInput, TabularVAEOutput]):
    @staticmethod
    def _get_batch_callbacks(tracking_client) -> List[BatchCallback]:
        # Configure which callbacks to execute on each batch
        return [BatchLossCallback(period=BATCH_LOSS_PERIOD, tracking_client=None)]
```

**Data Loader Routine Configuration** (Suggested directory: `components/routines/loader.py`)
```python
from typing import List
from artifact_torch.base.components.callbacks.data_loader import (
    DataLoaderScoreCallback,
    DataLoaderArrayCallback,
    DataLoaderPlotCallback,
)
from artifact_torch.base.components.routines.data_loader import DataLoaderRoutine
from artifact_torch.libs.components.callbacks.data_loader.loss import TrainLossCallback

class DemoLoaderRoutine(DataLoaderRoutine[TabularVAEInput, TabularVAEOutput]):
    @staticmethod
    def _get_score_callbacks() -> List[DataLoaderScoreCallback]:
        # Configure score callbacks for dataloader iteration
        return [TrainLossCallback(period=TRAIN_LOADER_CALLBACK_PERIOD)]
    
    @staticmethod
    def _get_array_callbacks() -> List[DataLoaderArrayCallback]:
        return []  # No array callbacks in this demo
    
    @staticmethod
    def _get_plot_callbacks() -> List[DataLoaderPlotCallback]:
        return []  # No plot callbacks in this demo
    
    # Similar methods for score_collection, array_collection, plot_collection callbacks
```

**Custom Callback Development**: For project-specific requirements, you can create custom callbacks tailored to your model's I/O profile by extending the appropriate base callback classes in your project's `libs/components/callbacks/` directory. These custom callbacks will seamlessly integrate with existing framework callbacks compatible with your model's I/O types, giving you both flexibility and access to the full ecosystem of pre-built functionality.

### Step 7: Trainer Configuration

**What you need to do**: Configure the trainer by extending CustomTrainer and implementing its hook methods.

The trainer exposes configurable hooks governing core aspects of the training lifecycle:

- **Optimization Setup**  
  Configure standard PyTorch training components, including:
  - Optimizer selection and hyperparameters
  - Learning-rate scheduler policy
  - Device placement

- **Early Stopping & Model Selection**  
  Specify termination criteria based on validation signals and track the best-performing model state for subsequent use.

- **Validation Routines**  
 Specify domain-agnostic validation logic at the batch/ data-laoder level as well as domain-specific evaluation hooks (e.g. comparing the real and synthetic data in a tabular synthesis experiment). This is achieved by implementing trainer hooks using the routines developed in the previous step.

- **Model State Monitoring (Checkpointing)**  
  Enable periodic and event-driven checkpointing to persist model weights, optimizer/scheduler state, and relevant metadata for recovery and reproducibility.

```python
from typing import Any, Optional
import torch
from torch import optim
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.components.callbacks.checkpoint import CheckpointCallback
from artifact_torch.base.components.early_stopping.stopper import EarlyStopper, StopperUpdateData
from artifact_torch.base.components.model_tracking.tracker import ModelTrackingCriterion
from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.base.components.routines.data_loader import DataLoaderRoutine
from artifact_torch.base.trainer.custom import CustomTrainer
from artifact_torch.libs.components.callbacks.checkpoint.standard import StandardCheckpointCallback
from artifact_torch.libs.components.early_stopping.epoch_bound import EpochBoundStopper
from artifact_torch.table_comparison.model import TableSynthesizer

class TabularVAETrainer(CustomTrainer[
    TableSynthesizer[TabularVAEInput, TabularVAEOutput, Any],
    TabularVAEInput,
    TabularVAEOutput,
    ModelTrackingCriterion,
    StopperUpdateData,
]):
    @staticmethod
    def _get_optimizer(model) -> optim.Optimizer:
        # Configure optimizer for your model
        return optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    @staticmethod
    def _get_scheduler(optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        # Configure learning rate scheduler (optional)
        return None  # No scheduler in this demo

    @staticmethod
    def _get_device() -> torch.device:
        # Configure training device
        return DEVICE

    @staticmethod
    def _get_early_stopper() -> EarlyStopper[StopperUpdateData]:
        # Configure early stopping criteria
        return EpochBoundStopper(max_n_epochs=MAX_N_EPOCHS)

    def _get_stopper_update_data(self) -> StopperUpdateData:
        # Configure data passed to early stopper
        return StopperUpdateData(n_epochs_elapsed=self.n_epochs_elapsed)

    @staticmethod
    def _get_checkpoint_callback(tracking_client) -> Optional[CheckpointCallback]:
        # Configure model checkpointing
        if tracking_client is not None:
            return StandardCheckpointCallback(
                period=CHECKPOINT_PERIOD, tracking_client=tracking_client
            )

    @staticmethod
    def _get_batch_routine(tracking_client) -> Optional[BatchRoutine]:
        # Configure batch processing routine
        return DemoBatchRoutine.build(tracking_client=tracking_client)

    @staticmethod
    def _get_train_loader_routine(data_loader, tracking_client) -> Optional[DataLoaderRoutine]:
        # Configure dataloader processing routine
        return DemoLoaderRoutine.build(data_loader=data_loader, tracking_client=tracking_client)
```

### (Optional) Step 8: Orchestrate All Components

**Optionally** create a high-level interface that orchestrates all your configured components into a clean, easy-to-use API.

```python
class TabularVAE:
    def __init__(self, discretizer: Discretizer, encoder: Encoder, config: TabularVAEConfig):
        self._discretizer = discretizer
        self._encoder = encoder
        self._config = config
        self._synthesizer: Optional[TabularVAESynthesizer] = None

    @classmethod
    def build(cls, data_spec: TabularDataSpecProtocol, config: TabularVAEConfig = TabularVAEConfig()) -> "TabularVAE":
        # Create and configure all preprocessing components
        discretizer = Discretizer(n_bins=config.n_bins_cts, ls_cts_features=data_spec.ls_cts_features)
        encoder = Encoder()
        return cls(discretizer=discretizer, encoder=encoder, config=config)

    def fit(self, df: pd.DataFrame, data_spec: TabularDataSpecProtocol, tracking_client: Optional[TrackingClient] = None) -> pd.DataFrame:
        # 1. Preprocess training data
        df_encoded = self._preprocess_training_data(df=df)
        
        # 2. Build synthesizer with all configured components
        self._synthesizer = TabularVAESynthesizer.build(
            data_spec=data_spec_encoded, discretizer=self._discretizer, 
            encoder=self._encoder, architecture_config=self._config.architecture
        )
        
        # 3. Create dataset and data loader
        dataset = TabularVAEDataset(df_encoded=df_encoded)
        loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, drop_last=DROP_LAST, shuffle=SHUFFLE)
        
        # 4. Set up artifact validation routine
        artifact_routine = DemoTableComparisonRoutine.build(
            df_real=df, data_spec=data_spec, tracking_client=tracking_client
        )
        
        # 5. Create and configure trainer
        trainer = TabularVAETrainer.build(
            model=self._synthesizer, train_loader=loader, 
            artifact_routine=artifact_routine, tracking_client=tracking_client
        )
        
        # 6. Execute training
        trainer.train()
        return trainer.epoch_scores

    def generate(self, n_records: int) -> pd.DataFrame:
        # Simple interface for generation
        params = TabularVAEGenerationParams(n_records=n_records, use_mean=GENERATION_USE_MEAN, temperature=GENERATION_TEMPERATURE, sample=True)
        return self._synthesizer.generate(params=params)
```