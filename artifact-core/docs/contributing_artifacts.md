# Contributing Artifacts

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

The success of this project hinges on the availability of a rich corpus of validation artifacts.

Contributions are strongly encouraged and highly appreciated.

To contribute new artifacts to the `artifact-core` project:

1. Add a new value to the appropriate existing Enum (e.g., in `artifact_core/table_comparison/registries/scores/types.py`)
2. Create and register your hyperparameters class (inheriting from `ArtifactHyperparams`)
3. Add the default configuration values in the appropriate config file (e.g. in `artifact_core/table_comparison/config/raw.json`)
4. Create and register your artifact class (inheriting from `Artifact` with the appropriate generics matching the engine of interest)

## Example: Contributing a New Score Artifact to the TableComparisonEngine

First, add your new score type to the existing enum in: artifact_core/table_comparison/registries/scores/types.py.
```python
class TableComparisonScoreType(ArtifactType):
    MEAN_JS_DISTANCE = "mean_js_distance"
    CORRELATION_DISTANCE = "correlation_distance"
    # Add your new score type
    NEW_TABLE_COMPARISON_SCORE = "new_table_comparison_score"
```
Then implement and register your artifact's hyperparameters:

```python
from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.table_comparison.registries.scores.registry import TableComparisonScoreRegistry


@TableComparisonScoreRegistry.register_artifact_hyperparams(
    TableComparisonScoreType.MY_CUSTOM_SCORE
    )
@dataclass
class NewTableComparisonScoreHyperparams(ArtifactHyperparams):
    threshold: float,
    use_weights: bool
```

The corresponding contribution to the configuration file (`artifact_core/table_comparison/config/raw.json`) should then look like:

```json
{
  "scores": {
    "my_custom_score": {
      "threshold": 0.5,
      "use_weights": true
    }
  }
}
```

Should your contribution not require any hyperparameters, simply use the following as the generic parameter:

```python
from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
```

In this case no hyperparams class needs to be registered and no configuration params need to be added to the config file.

The appropriate generics for table comparison scores are as follows:

```python
import pandas as pd

from artifact_core.base.artifact import Artifact
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.core.dataset_comparison.artifact import DatasetComparisonResources

Artifact[
        DatasetComparisonResources[pd.DataFrame],
        float,
        <HyperparamsT>,
        TabularDataSpecProtocol
        ]
```
However, note that we've provided more refined abstractions than the general artifact base class.

You should work with these instead: they implement core logic tailored to the specific artifact group in question.

To illustrate: all table comparison scores should inherit the following base:

```python
import pandas as pd

from artifact_core.table_comparison.artifacts.base import TableComparisonScore
from artifact_core.table_comparison.registries.scores.types import TableComparisonScoreType

TableComparisonScore[<HyperparamsT>]
```

Finally implement and register your artifact (accessing the relevant hyperparameters and resource spec):

```python
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import pandas as pd

from artifact_core.table_comparison.artifacts.base import TableComparisonScore, TableComparisonArtifactResources
from artifact_core.table_comparison.registries.scores.registry import TableComparisonScoreRegistry
from artifact_core.table_comparison.registries.scores.types import TableComparisonScoreType
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.core.dataset_comparison.artifact import DatasetComparisonResources


@TableComparisonScoreRegistry.register_artifact(
    TableComparisonScoreType.MY_CUSTOM_SCORE
    )
class NewTableComparisonScore(
    TableComparisonScore[
        NewTableComparisonScoreHyperparams
        ]
    ):
    def _validate(
        self,
        resources: TableComparisonArtifactResources
        ) -> TableComparisonArtifactResources:
        if resources.dataset_real is None or resources.dataset_synthetic is None:
            raise ValueError(
                "Both real and synthetic datasets must be provided"
                )
        return resources
        
    def _compare_datasets(
        self,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame
        ) -> float:
        dataset_real = dataset_real[self._resource_spec.ls_cts_features]
        dataset_synthetic = dataset_synthetic[self._resource_spec.ls_cts_features]
        score = 1.0
        if score > self._hyperparams.threshold and self._hyperparams.use_weights:
            score = 2*score
        return score
```