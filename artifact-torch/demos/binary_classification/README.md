# Artifact-Torch Demo: Binary Classification with MLP

> A comprehensive demonstration of the artifact-torch framework showcasing a binary classification experiment.

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

---

## 📋 Overview

This demo showcases the full capabilities of [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch) through an end-to-end binary classification experiment.

It demonstrates how to:

1. Build a Multilayer Perceptron (MLP) for binary classification.
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

The following code segment launches the binary classification training workflow.

```python
import pandas as pd
import seaborn as sns
from artifact_core.binary_classification import BinaryFeatureSpec
from artifact_experiment.tracking import FilesystemTrackingClient
from matplotlib import pyplot as plt

from demos.binary_classification.config.constants import (
    EXPERIMENT_ID,
    LABEL_FEATURE,
    LS_CATEGORIES,
    LS_FEATURES,
    POSITIVE_CATEGORY,
    TRAINING_DATASET_PATH,
)
from demos.binary_classification.mlp_classifier import MLPClassifier

# Load the dataset
df_train = pd.read_csv(artifact_torch_root / TRAINING_DATASET_PATH)

# Create data specification
data_spec = TabularDataSpec.from_df(
    df=df_real,
    ls_cts_features=LS_CTS_FEATURES,
    ls_cat_features=LS_CAT_FEATURES,
)

# Initialize tracking
filesystem_tracker = FilesystemTrackingClient.build(experiment_id=EXPERIMENT_ID)

# Build and train the MLP
model = MLPClassifier.build(class_spec=class_spec, ls_features=LS_FEATURES)
epoch_scores = model.fit(
    df=df_train,
    class_spec=class_spec,
    ls_features=LS_FEATURES,
    tracking_client=filesystem_tracker
)
```

### Execution: Notebook

We've packaged the full workflow in a Juyter notebook for convenience.

1. **Start Jupyter**: Launch Jupyter in the artifact-torch directory
2. **Open the notebook**: Navigate to `artifact_torch/demos/binary_classification/demo.ipynb`
3. **Run all cells**: Execute the cells in sequence to see the complete workflow

### Configuration

The demo is configurable through `artifact_torch/demos/binary_classification/config/config.json`:

```json
{
    "data": {
        "training_dataset_path": "assets/binary_classification.csv",
        "ls_features": ["weight", "height", "age", "bmi"],
        "label_feature": "arthritis_true",
        "ls_categories": ["0", "1"],
        "positive_category": "1"
    },
    "architecture": {
        "ls_hidden_sizes": [
            512,
            256
        ],
        "latent_dim": 128,
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
        "classification_threshold": 0.5
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
├── artifacts/              # Validation artifacts computed in periodic validation rounds 
├── metadata/          # Classification results obtained in periodic validation rounds 
└── torch_checkpoints/     # Model checkpoints
```

When you start training, the client prints the exact directory path where results are being saved

## 📊 Dataset

The demo uses the **Arthritis dataset** (`artifact_torch/assets/binary_classification.csv`) with:

**Continuous Features:**
- `weight`: Patient weight.
- `height`: Patient height.
- `age`: Patient age.
- `bmi`: Patient body/ mass index.

**Categorical Features:**
- `arthritis_true`: patient suffers from arthritis (target variable).

## 🎯 Model Architecture

The `MLPClassifier` implements a standard MLP architecture for binary classification:

### Network Components

1. **Encoder Network (`MLPEncoder`)**: Learns a latent representation of input data
   - Configurable layer sizes: `[512, 256]` (default)
   - Batch normalization and dropout for regularization
   - LeakyReLU activation functions

2. **Prediction Layer**: Maps the latent representation to logits over the labels

### Loss Function

Standard Categorical Cross Entropy (CCE) loss (negative log-likelihood).