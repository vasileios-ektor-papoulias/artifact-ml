from typing import Type, TypeVar

import pandas as pd
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonArtifactResources,
)

from artifact_experiment.base.callbacks.artifact import ArtifactCallbackResources

tableComparisonCallbackResourcesT = TypeVar(
    "tableComparisonCallbackResourcesT", bound="TableComparisonCallbackResources"
)


class TableComparisonCallbackResources(ArtifactCallbackResources[TableComparisonArtifactResources]):
    @classmethod
    def build(
        cls: Type[tableComparisonCallbackResourcesT],
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
    ) -> tableComparisonCallbackResourcesT:
        artifact_resources = TableComparisonArtifactResources(
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        callback_resources = cls(artifact_resources=artifact_resources)
        return callback_resources
