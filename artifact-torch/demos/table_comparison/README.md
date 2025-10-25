# Artifact-Torch Demo: Tabular Data Synthesis with VAE

> A comprehensive demonstration of the artifact-torch framework showcasing a tabular data synthesis experiment.

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

---

## 📋 Overview

This demo showcases the full capabilities of `artifact-torch` through an end-to-end tabular data synthesis experiment.

It demonstrates how to:

1. Build a Variational Autoencoder (VAE) for tabular data synthesis.
2. Train the model with reusable [Artifact-ML](https://github.com/vasileios-ektor-papoulias/artifact-ml) experiment workflows.

## 🚀 Getting Started

### Prerequisites

Ensure you have the `artifact-ml` workspace properly set up:

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-ml.git
cd artifact-ml/artifact-torch
poetry install
```

### Execution: Script

The following code segment launches the tabular synthesizer training workflow.

```python
import pandas as pd
import seaborn as sns
from artifact_core.table_comparison import TabularDataSpec
from artifact_experiment.tracking import FilesystemTrackingClient
from matplotlib import pyplot as plt

from demos.table_comparison.config.constants import (
    EXPERIMENT_ID,
    LS_CAT_FEATURES,
    LS_CTS_FEATURES,
    TRAINING_DATASET_PATH,
)
from demos.table_comparison.tabular_vae import TabularVAE

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
```

To generate synthetic data run:

```python
df_synthetic = model.generate(n_records=1000)
```

### Execution: Notebook

We've packaged the full workflow in a Juyter notebook for convenience.

1. **Start Jupyter**: Launch Jupyter in the artifact-torch directory
2. **Open the notebook**: Navigate to `artifact_torch/demos/table_comparison/demo.ipynb`
3. **Run all cells**: Execute the cells in sequence to see the complete workflow

### Configuration

The demo is configurable through `artifact_torch/demos/table_comparison/config/config.json`:

```json
{
    "data": {
        "training_dataset_path": "assets/real.csv",
        "ls_cts_features": ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"],
        "ls_cat_features": ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope", "HeartDisease"]
    },
    "transformers": {
        "n_bins_cts": 10
    },
    "architecture": {
        "n_embd": 8,
        "ls_encoder_layer_sizes": [
            512,
            256
        ],
        "latent_dim": 128,
        "loss_beta": 0.1,
        "leaky_relu_slope": 0.1,
        "bn_momentum": 0.1,
        "bn_epsilon": 1e-5,
        "dropout_rate": 0
    },
    "training": {
        "device": "cpu",
        "max_n_epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 512,
        "drop_last": false,
        "shuffle": true,
        "checkpoint_period": 5,
        "batch_loss_period": 1
    },
    "validation": {
        "train_loader_callback_period": 1,
        "validation_plan_callback_period": 5,
        "generation_n_records": 1000,
        "generation_use_mean": false,
        "generation_temperature": 1
    },
    "tracking":{
        "experiment_id": "demo"
    }
}
```

### Export Directory

The `FilesystemTrackingClient` saves all results to `~/artifact_ml/demo/<run_id>/` with this structure:

```
~/artifact_ml/demo/<run_id>/
├── artifacts/              # Validation plots, metrics, and statistics
├── tabular_data/          # Generated synthetic datasets  
└── torch_checkpoints/     # Model checkpoints
```

When you start training, the client prints the exact directory path where results are being saved

## 📊 Dataset

The demo uses the **Heart Disease dataset** (`artifact_torch/assets/real.csv`) with:

**Continuous Features:**
- `Age`: Patient age.
- `RestingBP`: Resting blood pressure.
- `Cholesterol`: Cholesterol level.
- `MaxHR`: Maximum heart rate achieved.
- `Oldpeak`: ST depression induced by exercise.

**Categorical Features:**
- `Sex`: Patient gender.
- `ChestPainType`: Type of chest pain.
- `FastingBS`: Fasting blood sugar.
- `RestingECG`: Resting electrocardiogram results.
- `ExerciseAngina`: Exercise-induced angina.
- `ST_Slope`: slope (direction and angle) of the ST segment on ECG tracing.
- `HeartDisease`: heart disease presence (target variable).

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