# Contributing Artifacts

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

The success of this project hinges on the availability of a rich corpus of validation artifacts.

Contributions are strongly encouraged and highly appreciated.

To contribute new artifacts to the `artifact-core` project:

1. Add a new value to the appropriate existing Enum (e.g., in `artifact_core/table_comparison/_types/scores.py`)
2. Create and register your hyperparameters class (inheriting from `ArtifactHyperparams`)
3. Add the default configuration values in the appropriate config file (e.g. in `artifact_core/table_comparison/_config/raw.json`)
4. Create and register your artifact class (inheriting from `Artifact` with the appropriate generics matching the engine of interest)

## Example: Contributing a New Score Artifact to the TableComparisonEngine

First, add your new score type to the existing enum in: `artifact_core/table_comparison/_types/scores.py`.
```python
class TableComparisonScoreType(ArtifactType):
    MEAN_JS_DISTANCE = "mean_js_distance"
    CORRELATION_DISTANCE = "correlation_distance"
    # Add your new score type
    MY_CUSTOM_SCORE = "my_custom_score"
```
Then implement and register your artifact's hyperparameters:

```python
from dataclasses import dataclass

from artifact_core.spi.artifact import ArtifactHyperparams
from artifact_core.table_comparison.spi import TableComparisonScoreRegistry
from artifact_core.table_comparison._types.scores import TableComparisonScoreType


@TableComparisonScoreRegistry.register_artifact_hyperparams(
    TableComparisonScoreType.MY_CUSTOM_SCORE
    )
@dataclass(frozen=True)
class MyCustomScoreHyperparams(ArtifactHyperparams):
    threshold: float
    use_weights: bool
```

The corresponding contribution to the configuration file (`artifact_core/table_comparison/_config/raw.json`) should then look like the following---note that configuration keys are the enum member *names*:

```json
{
  "scores": {
    "MY_CUSTOM_SCORE": {
      "threshold": 0.5,
      "use_weights": true
    }
  }
}
```

Should your contribution not require any hyperparameters, simply use the following as the generic parameter:

```python
from artifact_core.spi.artifact import NO_ARTIFACT_HYPERPARAMS
```

In this case no hyperparams class needs to be registered and no configuration params need to be added to the config file.

The appropriate generics for table comparison scores are as follows:

```python
import pandas as pd

from artifact_core.spi.artifact import Artifact
from artifact_core.spi.resources import DatasetComparisonArtifactResources
from artifact_core.table_comparison.spi import TabularDataSpecProtocol

Artifact[
        DatasetComparisonArtifactResources[pd.DataFrame],
        TabularDataSpecProtocol,
        <HyperparamsT>,
        float
        ]
```
However, note that we've provided more refined abstractions than the general artifact base class.

You should work with these instead: they implement core logic tailored to the specific artifact group in question.

To illustrate: all table comparison scores should inherit the following base:

```python
from artifact_core.table_comparison.spi import TableComparisonScore

TableComparisonScore[<HyperparamsT>]
```

Finally implement and register your artifact (accessing the relevant hyperparameters and resource spec).

Note that the table comparison base classes already implement resource validation (`_validate`): you only need to implement `_compare_datasets`:

```python
import pandas as pd

from artifact_core.table_comparison.spi import (
    TableComparisonScore,
    TableComparisonScoreRegistry,
)
from artifact_core.table_comparison._types.scores import TableComparisonScoreType


@TableComparisonScoreRegistry.register_artifact(
    TableComparisonScoreType.MY_CUSTOM_SCORE
    )
class MyCustomScore(
    TableComparisonScore[
        MyCustomScoreHyperparams
        ]
    ):
    def _compare_datasets(
        self,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame
        ) -> float:
        dataset_real = dataset_real[self._resource_spec.cts_features]
        dataset_synthetic = dataset_synthetic[self._resource_spec.cts_features]
        score = 1.0
        if score > self._hyperparams.threshold and self._hyperparams.use_weights:
            score = 2*score
        return score
```
