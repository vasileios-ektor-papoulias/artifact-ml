# User Guide

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## Usage Sketch

### Validation Plan Configuration

```python
from typing import List
from artifact_experiment.table_comparison import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonPlan,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)

class MyArtifactPlan(TableComparisonPlan):
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

**Note**: all six type getters are required abstract hooks. Returning an empty list (as `_get_array_types` does above) declares that no artifacts of that kind should be computed.

### Validation Plan Execution

```python
import pandas as pd

from artifact_experiment.table_comparison import TabularDataSpec

# Load and prepare data
df_real = pd.read_csv("real_data.csv")
df_synthetic = pd.read_csv("synthetic_data.csv")

continuous_features = ["feature1", "feature2", "feature3"]
resource_spec = TabularDataSpec.from_df(
    df=df_real,
    cts_features=continuous_features,
    cat_features=[col for col in df_real.columns if col not in continuous_features]
)

# Execute validation plan
plan = MyArtifactPlan.create(resource_spec=resource_spec)
plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)

# Access computed artifacts
js_distance = plan.scores.get("MEAN_JS_DISTANCE")
pca_plot = plan.plots.get("PCA_JUXTAPOSITION")
feature_means = plan.array_collections.get("MEAN_JUXTAPOSITION")
```

### Experiment Tracking Integration

#### MLflow Integration
```python
from artifact_experiment.tracking import MlflowTrackingClient

# Create tracking client: pass the experiment name as experiment_id;
# the MLflow experiment is created automatically if it doesn't exist
MLFLOW_EXPERIMENT_NAME = "artifact-experiment-demo"
mlflow_client = MlflowTrackingClient.build(experiment_id=MLFLOW_EXPERIMENT_NAME)

# Create validation plan with tracking enabled
plan = MyArtifactPlan.create(resource_spec=resource_spec, tracking_client=mlflow_client)

# Execute validation (results automatically logged to MLflow)
plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)

# Stop the client (flushes the background logging worker and terminates the MLflow run)
mlflow_client.stop()
```

#### ClearML Integration
```python
from artifact_experiment.tracking import ClearMLTrackingClient

# Create ClearML tracking client
CLEAR_ML_PROJECT_NAME = "artifact-experiment-demo"
clearml_client = ClearMLTrackingClient.build(experiment_id=CLEAR_ML_PROJECT_NAME)

# Create and execute validation plan
plan = MyArtifactPlan.create(resource_spec=resource_spec, tracking_client=clearml_client)
plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)
clearml_client.stop()
```

#### Neptune Integration
```python
from artifact_experiment.tracking import NeptuneTrackingClient

# Create Neptune tracking client
NEPTUNE_PROJECT_NAME = "artifact-experiment-demo"
neptune_client = NeptuneTrackingClient.build(experiment_id=NEPTUNE_PROJECT_NAME)

# Create and execute validation plan
plan = MyArtifactPlan.create(resource_spec=resource_spec, tracking_client=neptune_client)
plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)
neptune_client.stop()
```

#### Local Filesystem Integration
```python
from artifact_experiment.tracking import FilesystemTrackingClient

# Create filesystem tracking client (saves to ~/artifact_ml/)
EXPERIMENT_ID = "artifact-experiment-demo"
filesystem_client = FilesystemTrackingClient.build(experiment_id=EXPERIMENT_ID)

# Create and execute validation plan
plan = MyArtifactPlan.create(resource_spec=resource_spec, tracking_client=filesystem_client)
plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)
filesystem_client.stop()

# Results saved to ~/artifact_ml/artifact-experiment-demo/<filesystem_client.run.run_id>
```

### Binary Classification Validation

Alongside table comparison, the package ships a domain plan for classifier evaluation: `BinaryClassificationPlan`. It is used the same way — subclass it, implement the six artifact type getters (using the `BinaryClassification*` enums), and instantiate with `create`:

```python
from artifact_experiment.binary_classification import BinaryClassificationPlan

class MyClassificationPlan(BinaryClassificationPlan):
    # Implement _get_score_types, _get_array_types, _get_plot_types,
    # _get_score_collection_types, _get_array_collection_types,
    # _get_plot_collection_types
    ...

plan = MyClassificationPlan.create(resource_spec=class_spec, tracking_client=mlflow_client)
plan.execute_classifier_evaluation(true=y_true, predicted=y_pred, probs_pos=y_probs)
```
