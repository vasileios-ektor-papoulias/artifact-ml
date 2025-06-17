# 🚀 Artifact-Torch Demo: Tabular Data Synthesis with VAE

> A comprehensive demonstration of the artifact-torch framework showcasing tabular data synthesis using a Variational Autoencoder.

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

---

## 📋 Overview

This demo showcases the full capabilities of `artifact-torch` through a production-ready tabular data synthesis pipeline.

It demonstrates how to:

1. **Build and train** a Variational Autoencoder (VAE) for tabular data synthesis
2. **Integrate seamlessly** with `artifact-core`'s validation artifacts and `artifact-experiment`'s validation plans through `artifact-torch`

## 🏗️ Architecture

The demo implements a VAE-based tabular synthesizer using `artifact-torch`:

### Core Components

- **`TabularVAE`**: High-level interface orchestrating the entire synthesis pipeline
- **`TabularVAESynthesizer`**: Implements `artifact-torch`'s `TableSynthesizer` interface for VAE-based synthesis
- **`TabularVAETrainer`**: Extends `artifact-torch`'s `CustomTrainer` to create a concrete training loop for the VAE model
- **`DemoTableComparisonRoutine`**: Configures `TableComparisonRoutine` (`ArtifactRoutine` subclass for tabular data synthesis) to integrate artifact-ML validation into the training loop
- **`DemoBatchRoutine`**: Configures `BatchRoutine` to provides batch-level performance evaluation callbacks
- **`DemoLoaderRoutine`**: Configures `DataLoaderRoutine` to handles epoch-end performance monitoring through dataloader iteration

### File Structure

```
demo/
├── demo.ipynb                     # Main demonstration notebook
├── tabular_vae.py                 # High-level TabularVAE interface
├── config/
│   ├── config.json                # Configuration parameters
│   └── constants.py               # Configuration loading utilities
├── model/
│   ├── synthesizer.py             # VAE model implementation
│   ├── io.py                      # Model I/O utilities
│   └── architectures/
│       └── vae.py                 # VAE architecture definition
├── trainer/
│   └── trainer.py                 # Extends CustomTrainer for VAE
├── components/
│   └── routines/
│       ├── artifact.py            # Integrates artifact-ML validation into training loop
│       ├── batch.py               # Provides batch-level performance evaluation callbacks
│       └── loader.py              # Handles epoch-end performance monitoring through dataloader iteration
├── data/
│   └── dataset.py                 # Dataset implementation for VAE
└── libs/
    ├── transformers/
    │   ├── discretizer.py         # Continuous feature discretization
    │   └── encoder.py             # Categorical feature encoding
    ├── layers/
    │   ├── mlp.py                 # Multi-layer perceptron implementation
    │   ├── diagonal_gaussian_latent.py  # Gaussian latent layer
    │   ├── embedder.py            # Feature embedding layer
    │   ├── multi_feature_predictor.py   # Multi-feature prediction
    │   └── lin_bn_drop.py         # Linear + BatchNorm + Dropout layer
    ├── losses/
    │   └── beta_loss.py           # Beta-VAE loss implementation
    └── utils/
        └── sampler.py             # Sampling utilities
```

## 🚀 Getting Started

### Prerequisites

Ensure you have the `artifact-ml` workspace properly set up:

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-ml.git
cd artifact-ml/artifact-torch
poetry install
```

### Running the Demo

#### Quick Start Example

```python
import pandas as pd
from artifact_core.libs.resource_spec.tabular.spec import TabularDataSpec
from artifact_experiment.libs.tracking.filesystem.client import FilesystemTrackingClient
from demo.tabular_vae import TabularVAE
from demo.config.constants import LS_CAT_FEATURES, LS_CTS_FEATURES, TRAINING_DATASET_PATH

# Load the dataset
df_real = pd.read_csv(TRAINING_DATASET_PATH)

# Create data specification
data_spec = TabularDataSpec.from_df(
    df=df_real,
    ls_cts_features=LS_CTS_FEATURES,
    ls_cat_features=LS_CAT_FEATURES,
)

# Initialize tracking
filesystem_tracker = FilesystemTrackingClient.build(experiment_id="demo")

# Build and train the VAE
model = TabularVAE.build(data_spec=data_spec)
epoch_scores = model.fit(
    df=df_real, 
    data_spec=data_spec, 
    tracking_client=filesystem_tracker
)

# Generate synthetic data
df_synthetic = model.generate(n_records=1000)
```

We've packaged this complete workflow in an interactive Jupyter notebook for easy exploration:

#### Running the Notebook

1. **Start Jupyter**: Launch Jupyter in the artifact-torch directory
2. **Open the notebook**: Navigate to `demo/demo.ipynb`
3. **Run all cells**: Execute the cells in sequence to see the complete workflow

#### Configuration

The demo is configurable through `config/config.json`:

```json
{
    "data": {
        "training_dataset_path": "assets/real.csv",
        "ls_cts_features": ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"],
        "ls_cat_features": ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope", "HeartDisease"]
    },
    "architecture": {
        "n_embd": 8,
        "ls_encoder_layer_sizes": [512, 256],
        "latent_dim": 128,
        "loss_beta": 0.1
    },
    "training": {
        "max_n_epochs": 200,
        "learning_rate": 0.001,
        "batch_size": 512,
        "checkpoint_period": 5
    },
    "validation": {
        "validation_plan_callback_period": 5,
        "generation_n_records": 1000
    }
}
```

#### Where Results Are Stored

The `FilesystemTrackingClient` saves all results to `~/artifact_ml/demo/<run_id>/` with this structure:

```
~/artifact_ml/demo/<run_id>/
├── artifacts/              # Validation plots, metrics, and statistics
├── tabular_data/          # Generated synthetic datasets  
└── torch_checkpoints/     # Model checkpoints
```

When you start training, the client prints the exact directory path where results are being saved

## 📊 Dataset

The demo uses the **Heart Disease dataset** (`../assets/real.csv`) with:

**Continuous Features:**
- `Age`: Patient age
- `RestingBP`: Resting blood pressure
- `Cholesterol`: Cholesterol level
- `MaxHR`: Maximum heart rate achieved
- `Oldpeak`: ST depression induced by exercise

**Categorical Features:**
- `Sex`: Patient gender
- `ChestPainType`: Type of chest pain
- `FastingBS`: Fasting blood sugar
- `RestingECG`: Resting electrocardiogram results
- `ExerciseAngina`: Exercise-induced angina
- `ST_Slope`: ST slope
- `HeartDisease`: Target variable (heart disease presence)

## 🎯 Model Architecture

The `TabularVAESynthesizer` implements a **β-VAE** (Beta Variational Autoencoder) specifically designed for tabular data:

### Network Components

1. **Encoder Network**: Transforms input data into latent mean and log-variance
   - Configurable layer sizes: `[512, 256]` (default)
   - Batch normalization and dropout for regularization
   - LeakyReLU activation functions

2. **Latent Space**: Gaussian latent representation
   - Latent dimension: `128` (configurable)
   - Reparameterization trick for differentiable sampling

3. **Decoder Network**: Reconstructs data from latent samples
   - Mirror architecture of encoder
   - Outputs reconstruction of original data

### Loss Function

Combines reconstruction and regularization terms:
- **Reconstruction Loss**: Cross-entropy for categorical features
- **KL Divergence**: Regularizes latent space distribution
- **β Parameter**: Controls regularization strength (`β = 0.1`)

## 🔧 Step-by-Step Guide: Building a Project with Artifact-Torch

This section provides an **intuitive step-by-step guide** for building your own project using `artifact-torch`. Each step shows you how to configure the framework's interfaces for your specific use case - essentially, all implementations are configurations that tell the framework how to handle your particular ML task.

### Step 1: Select Your ML Task and Framework Coverage

The **first step** is identifying your ML task and checking if `artifact-torch` provides framework support for it. Currently supported tasks include:

- **Tabular Data Synthesis**: Generate synthetic tabular data (our demo focus)

Once you select your project type, `artifact-torch` **always provides two interfaces** for you to implement:

1. **The model interface** - defines how your model integrates with the training framework
2. **The artifact validation routine** - leverages `artifact-core` and `artifact-experiment` to provide project-type specific validation capabilities that are periodically injected into the training loop

For **tabular data synthesis**, these interfaces are:
- `TableSynthesizer` interface for models
- `TableComparisonRoutine` for validation

### Step 2: Define the Model Input/Output Types

**What you need to do**: Define typed interfaces that describe your model's inputs and outputs.

#### **Model I/O Definition** (`model/io.py`)
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

**Configuration Purpose:**
- **Type Contracts**: Framework knows exactly what data flows through your pipeline
- **Callback Compatibility**: These I/O types determine which batch and dataloader callbacks you can use, as static type analysis can verify that your model I/O types satisfy the framework's callback type requirements through the type variance system

### Step 3: Implement the Model

**What you need to do**: Implement the model interface for your specific architecture while respecting the expected contracts. This allows you to build whatever architecture you want while maintaining full control over your model's internal structure and behavior.

In general, implementing one of the model interfaces—in this case for table synthesis—might require implementing secondary configuration objects, such as generation parameters.

#### **Model Implementation** (`model/synthesizer.py`)
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

#### **Secondary Configuration Objects** (`model/synthesizer.py`)
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

### Step 4: Implement the Dataset

**What you need to do**: Implement how to load individual training samples by extending the Dataset interface while respecting the expected contracts. The framework provides lightweight wrappers around PyTorch's native objects with type annotations and enhanced functionality.

#### **Dataset Implementation** (`data/dataset.py`)
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

**Using Dataset with DataLoader**: Once you've implemented your dataset, you can use it with the framework's DataLoader wrapper for batching during training:

```python
from artifact_torch.base.data.data_loader import DataLoader

loader = DataLoader(
    dataset=dataset, 
    batch_size=BATCH_SIZE, 
    drop_last=DROP_LAST, 
    shuffle=SHUFFLE
)
```

### Step 5: Configure your custom `Validation Plan`

**What you need to do**: Select the artifact collection you'd like to track as training progresses. Your customized plan will help configure the `ArtifactRoutine` injected into the training loop---see step 6. 

#### **Validation Plan Configuration** (`artifact_experiment/table_comparison/validation_plan.py`)
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

### Step 6: Configure Validation Routines

**What you need to do**: Configure all validation hooks you would like to inject into your training loop:
- `ArtifactRoutine`: Artifact-ML validation flow (in the context of tabular data synthesis this becomes `TableComparisonRoutine`).
- `DataLoaderRoutine`: Monitoring via post-epoch per-data loader execution,
- `BatchRoutine`: Monitoring via per-batch execution

#### **Validation Routine Configuration** (`components/routines/artifact.py`)
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

#### **Batch Routine Configuration** (`components/routines/batch.py`)
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

#### **Data Loader Routine Configuration** (`components/routines/loader.py`)
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

### Step 7: Configure the Trainer

**What you need to do**: Configure the trainer by extending CustomTrainer and implementing its hook methods. You're configuring the parameters and behavior of the training loop—optimization strategy, learning rate scheduling, early stopping criteria, and checkpointing behavior.

#### **Trainer Configuration** (`trainer/trainer.py`)
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

**Trainer Configuration Hooks:**
- **Optimization**: Configure optimizer, scheduler, and device
- **Early Stopping**: Configure when training should terminate
- **Checkpointing**: Configure model saving during training
- **Routines**: Wire up your batch and dataloader routines
- **Tracking**: Configure model tracking and metrics collection

**Framework Handles Automatically:**
- ✅ Training loop execution
- ✅ Device management and data transfer
- ✅ Gradient computation and backpropagation
- ✅ Metric aggregation and logging
- ✅ Checkpoint saving and loading
- ✅ Early stopping evaluation

### Step 8: Orchestrate All Components

**This step is optional - it's the approach we have followed in the demo, prompting us to include it in this guide**: Create a high-level interface that orchestrates all your configured components into a clean, easy-to-use API.

#### **Orchestration Class** (`tabular_vae.py`)
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

**Orchestration Benefits:**
- **Clean API**: Users interact with a simple, intuitive interface
- **Component Integration**: All configured pieces work together seamlessly  
- **Error Handling**: Central place to manage component dependencies and validation
- **Simplified Usage**: Complex configuration details are hidden from end users

**Usage Result:**
```python
# Simple 3-line usage instead of managing 7+ components manually
model = TabularVAE.build(data_spec=data_spec)
model.fit(df=df_real, data_spec=data_spec, tracking_client=tracking_client)
synthetic_data = model.generate(n_records=1000)
```

This orchestration pattern transforms your framework configurations into a production-ready, user-friendly interface.

