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
poetry install --with dev
```

### Execution: Script

The following code segment (run from the `artifact-torch` directory) launches the binary classification training workflow.

```python
import pandas as pd
from artifact_experiment.tracking import DataSplit, FilesystemTrackingClient
from artifact_torch.binary_classification import BinaryClassSpec
from sklearn.model_selection import train_test_split

from demos.binary_classification.config.constants import (
    EXPERIMENT_ID,
    LABEL_FEATURE,
    LS_CLASS_NAMES,
    LS_FEATURES,
    POSITIVE_CLASS_NAME,
    TRAINING_DATASET_PATH,
    VAL_DATA_PROPORTION,
)
from demos.binary_classification.data.utils import DemoDataUtils
from demos.binary_classification.experiment.experiment import DemoBinaryClassificationExperiment
from demos.binary_classification.model.classifier import MLPClassifier

# Load the dataset and describe the classification task with a spec
df_all = pd.read_csv(TRAINING_DATASET_PATH)
class_spec = BinaryClassSpec(
    class_names=LS_CLASS_NAMES, positive_class=POSITIVE_CLASS_NAME, label_name=LABEL_FEATURE
)

# Split into train and validation sets
df_train, df_val = train_test_split(
    df_all, test_size=VAL_DATA_PROPORTION, random_state=42, shuffle=True
)

# Assemble the experiment inputs
data_loaders = {
    DataSplit.TRAIN: DemoDataUtils.build_data_loader(df=df_train, class_spec=class_spec),
    DataSplit.VALIDATION: DemoDataUtils.build_data_loader(df=df_val, class_spec=class_spec),
}
artifact_routine_data = {
    DataSplit.TRAIN: DemoDataUtils.build_artifact_routine_data(df=df_train, class_spec=class_spec),
    DataSplit.VALIDATION: DemoDataUtils.build_artifact_routine_data(df=df_val, class_spec=class_spec),
}
model = MLPClassifier.build(class_spec=class_spec, ls_features=LS_FEATURES)
tracking_client = FilesystemTrackingClient.build(experiment_id=EXPERIMENT_ID)

# Build and run the experiment
experiment = DemoBinaryClassificationExperiment.build(
    model=model,
    data_loaders=data_loaders,
    artifact_routine_data=artifact_routine_data,
    artifact_routine_data_spec=class_spec,
    tracking_client=tracking_client,
)
experiment.run()

experiment.epoch_scores
```

### Execution: Notebook

We've packaged the full workflow in a Jupyter notebook for convenience.

1. **Start Jupyter**: Launch Jupyter in the `artifact-torch` directory
2. **Open the notebook**: Navigate to `demos/binary_classification/demo.ipynb`
3. **Run all cells**: Execute the cells in sequence to see the complete workflow

### Configuration

The demo is configurable through `demos/binary_classification/config/config.json`:

```json
{
    "data": {
        "training_dataset_path": "assets/binary_classification.csv",
        "val_data_proportion": 0.2,
        "ls_features": ["weight", "height", "age", "bmi"],
        "label_feature": "arthritis_true",
        "ls_class_names": ["0", "1"],
        "positive_class_name": "1"
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
        "max_n_epochs": 50,
        "learning_rate": 0.001,
        "batch_size": 64,
        "drop_last": false,
        "shuffle": true,
        "checkpoint_period": 5
    },
    "validation": {
        "train_diagnostics_period": 1,
        "loader_validation_period": 5,
        "artifact_routine_period": 5,
        "classification_threshold": 0.5
    },
    "tracking":{
        "experiment_id": "demo"
    }
}
```

### Export Directory

The `FilesystemTrackingClient` saves all results (validation artifacts, classification metadata, and model checkpoints) under `~/artifact_ml/<experiment_id>/<run_id>/`.

When you start training, the client prints the exact directory path where results are being saved.

## 📊 Dataset

The demo uses the **Arthritis dataset** (`artifact-torch/assets/binary_classification.csv`) with:

**Continuous Features:**
- `weight`: Patient weight.
- `height`: Patient height.
- `age`: Patient age.
- `bmi`: Patient body mass index.

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