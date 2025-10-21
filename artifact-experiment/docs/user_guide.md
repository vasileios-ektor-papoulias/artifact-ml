# User Guide

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>
s

## Usage Sketch

### Validation Plan Configuration

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

class MyValidationPlan(TableComparisonPlan):
    @staticmethod
    def _get_score_types() -> List[TableComparisonScoreType]:
        return [
            TableComparisonScoreType.MEAN_JS_DISTANCE,
            TableComparisonScoreType.CORRELATION_DISTANCE,
        ]

    @staticmethod
    def _get_plot_types() -> List[TableComparisonPlotType]:
        return [
            TableComparisonPlotType.PDF,
            TableComparisonPlotType.CDF,
            TableComparisonPlotType.PCA_JUXTAPOSITION,
        ]

    @staticmethod
    def _get_score_collection_types() -> List[TableComparisonScoreCollectionType]:
        return [
            TableComparisonScoreCollectionType.JS_DISTANCE
            ]

    @staticmethod
    def _get_array_collection_types() -> List[TableComparisonArrayCollectionType]:
        return [
            TableComparisonArrayCollectionType.MEAN_JUXTAPOSITION,
            TableComparisonArrayCollectionType.STD_JUXTAPOSITION,
        ]

    @staticmethod  
    def _get_plot_collection_types() -> List[TableComparisonPlotCollectionType]:
        return [
            TableComparisonPlotCollectionType.PDF
            ]
```

### Validation Plan Execution

```python
import pandas as pd

from artifact_core.libs.resource_spec.tabular.spec import TabularDataSpec

# Load and prepare data
df_real = pd.read_csv("real_data.csv")
df_synthetic = pd.read_csv("synthetic_data.csv")

continuous_features = ["feature1", "feature2", "feature3"]
resource_spec = TabularDataSpec.from_df(
    df=df_real,
    ls_cts_features=continuous_features,
    ls_cat_features=[col for col in df_real.columns if col not in continuous_features]
)

# Execute validation plan
plan = MyValidationPlan.build(resource_spec=resource_spec)
plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)

# Access computed artifacts
js_distance = plan.scores.get("MEAN_JS_DISTANCE")
pca_plot = plan.plots.get("PCA_JUXTAPOSITION")
feature_means = plan.array_collections.get("MEAN_JUXTAPOSITION")
```

### Experiment Tracking Integration

#### MLflow Integration
```python
from artifact_experiment.libs.tracking.mlflow.client import MlflowTrackingClient

# Setup MLflow experiment
MLFLOW_EXPERIMENT_NAME = "artifact-experiment-demo"
experiment_id = MlflowTrackingClient.create_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)

# Create tracking client and build validation plan
mlflow_client = MlflowTrackingClient.build(experiment_id=experiment_id)
plan = MyValidationPlan.build(resource_spec=resource_spec, tracking_client=mlflow_client)

# Execute validation (results automatically logged to MLflow)
plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)

# Stop MLflow run
mlflow_client.run.stop()
```

#### ClearML Integration
```python
from artifact_experiment.libs.tracking.clear_ml.client import ClearMLTrackingClient

# Create ClearML tracking client
CLEAR_ML_PROJECT_NAME = "artifact-experiment-demo"
clearml_client = ClearMLTrackingClient.build(experiment_id=CLEAR_ML_PROJECT_NAME)

# Build and execute validation plan
plan = MyValidationPlan.build(resource_spec=resource_spec, tracking_client=clearml_client)
plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)
clearml_client.run.stop()
```

#### Neptune Integration
```python
from artifact_experiment.libs.tracking.neptune.client import NeptuneTrackingClient

# Create Neptune tracking client
NEPTUNE_PROJECT_NAME = "artifact-experiment-demo"
neptune_client = NeptuneTrackingClient.build(experiment_id=NEPTUNE_PROJECT_NAME)

# Build and execute validation plan
plan = MyValidationPlan.build(resource_spec=resource_spec, tracking_client=neptune_client)
plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)
neptune_client.run.stop()
```

#### Local Filesystem Integration
```python
from artifact_experiment.libs.tracking.filesystem.client import FilesystemTrackingClient

# Create filesystem tracking client (saves to ~/artifact_ml/)
EXPERIMENT_ID = "artifact-experiment-demo"
filesystem_client = FilesystemTrackingClient.build(experiment_id=EXPERIMENT_ID)

# Build and execute validation plan
plan = MyValidationPlan.build(resource_spec=resource_spec, tracking_client=filesystem_client)
plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)
filesystem_client.run.stop()

# Results saved to ~/artifact_ml/artifact-experiment-demo/<filesystem_client.run.run_id>
```