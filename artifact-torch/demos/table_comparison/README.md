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
poetry install --with dev
```

### Execution: Script

The following code segment (run from the `artifact-torch` directory) launches the tabular synthesizer training workflow.

```python
import pandas as pd
from artifact_core.table_comparison import TabularDataSpec
from artifact_experiment.tracking import DataSplit, FilesystemTrackingClient

from demos.table_comparison.config.constants import (
    EXPERIMENT_ID,
    LS_CAT_FEATURES,
    LS_CTS_FEATURES,
    N_BINS_CTS,
    TRAINING_DATASET_PATH,
)
from demos.table_comparison.data.utils import DemoDataUtils
from demos.table_comparison.experiment.experiment import DemoTabularSynthesisExperiment
from demos.table_comparison.libs.transformers.discretizer import Discretizer
from demos.table_comparison.libs.transformers.encoder import Encoder
from demos.table_comparison.model.synthesizer import TabularVAESynthesizer

# Load the dataset and describe it with a spec
df_real = pd.read_csv(TRAINING_DATASET_PATH)
raw_data_spec = TabularDataSpec.from_df(
    df=df_real, cts_features=LS_CTS_FEATURES, cat_features=LS_CAT_FEATURES
)

# Fit the preprocessing transformers (discretize, then one-hot encode)
discretizer = Discretizer(n_bins=N_BINS_CTS, ls_cts_features=raw_data_spec.cts_features)
discretizer.fit(df=df_real)
df_discretized = discretizer.transform(df=df_real)

encoder = Encoder()
encoder.fit(df=df_discretized, ls_cat_features=list(df_discretized.columns))
df_encoded = encoder.transform(df=df_discretized)
encoded_data_spec = TabularDataSpec.from_df(df=df_encoded, cat_features=list(df_encoded.columns))

# Assemble the experiment inputs
data_loaders = {
    DataSplit.TRAIN: DemoDataUtils.build_data_loader(
        df=df_real, discretizer=discretizer, encoder=encoder
    )
}
artifact_routine_data = {
    DataSplit.TRAIN: DemoDataUtils.build_artifact_routine_data(df_real=df_real)
}
model = TabularVAESynthesizer.build(
    data_spec=encoded_data_spec, discretizer=discretizer, encoder=encoder
)
tracking_client = FilesystemTrackingClient.build(experiment_id=EXPERIMENT_ID)

# Build and run the experiment
experiment = DemoTabularSynthesisExperiment.build(
    model=model,
    data_loaders=data_loaders,
    artifact_routine_data=artifact_routine_data,
    artifact_routine_data_spec=raw_data_spec,
    tracking_client=tracking_client,
)
experiment.run()

experiment.epoch_scores
```

To generate synthetic data run:

```python
from demos.table_comparison.contracts.model import TabularVAEGenerationParams

df_synthetic = model.generate(params=TabularVAEGenerationParams(n_records=1000, temperature=1.0))
```

### Execution: Notebook

We've packaged the full workflow in a Jupyter notebook for convenience.

1. **Start Jupyter**: Launch Jupyter in the `artifact-torch` directory
2. **Open the notebook**: Navigate to `demos/table_comparison/demo.ipynb`
3. **Run all cells**: Execute the cells in sequence to see the complete workflow

### Configuration

The demo is configurable through `demos/table_comparison/config/config.json`:

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
        "checkpoint_period": 5
    },
    "validation": {
        "batch_routine_period": 1,
        "train_loader_routine_period": 5,
        "artifact_routine_period": 5,
        "generation_n_records": 1000,
        "generation_temperature": 1
    },
    "tracking":{
        "experiment_id": "demo"
    }
}
```

### Export Directory

The `FilesystemTrackingClient` saves all results (validation artifacts, generated data, and model checkpoints) under `~/artifact_ml/<experiment_id>/<run_id>/`.

When you start training, the client prints the exact directory path where results are being saved.

## 📊 Dataset

The demo uses the **Heart Disease dataset** (`artifact-torch/assets/real.csv`) with:

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