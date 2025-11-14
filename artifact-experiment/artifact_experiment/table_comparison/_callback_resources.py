from dataclasses import dataclass
from typing import Type, TypeVar

import pandas as pd
from artifact_core.table_comparison.spi import TableComparisonArtifactResources

from artifact_experiment._domains.dataset_comparison.callback_resources import (
    DatasetComparisonCallbackResources,
)

TableComparisonCallbackResourcesT = TypeVar(
    "TableComparisonCallbackResourcesT", bound="TableComparisonCallbackResources"
)


@dataclass(frozen=True)
class TableComparisonCallbackResources(DatasetComparisonCallbackResources[pd.DataFrame]):
    @classmethod
    def build(
        cls: Type[TableComparisonCallbackResourcesT],
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
    ) -> TableComparisonCallbackResourcesT:
        artifact_resources = TableComparisonArtifactResources(
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        callback_resources = cls(artifact_resources=artifact_resources)
        return callback_resources
