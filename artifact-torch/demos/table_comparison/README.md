# 🚀 Artifact-Torch Demo: Tabular Data Synthesis with VAE

> A comprehensive demonstration of the artifact-torch framework showcasing a tabular data synthesis experiment.

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
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
from demos.table_comparison.tabular_vae import TabularVAE
from demos.table_comparison.config.constants import LS_CAT_FEATURES, LS_CTS_FEATURES, TRAINING_DATASET_PATH

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
2. **Open the notebook**: Navigate to `demos/table_comparison/demo.ipynb`
3. **Run all cells**: Execute the cells in sequence to see the complete workflow

#### Configuration

The demo is configurable through `demos/table_comparison/config/config.json`:

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