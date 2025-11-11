from abc import abstractmethod
from typing import Dict, Generic, Tuple, TypeVar

import pandas as pd
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core._base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResult,
)
from artifact_core._core.dataset_comparison.artifact import (
    DatasetComparisonArtifact,
    DatasetComparisonArtifactResources,
)
from artifact_core._libs.resource_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)
from artifact_core._libs.resource_validation.tabular.table_validator import (
    TableValidator,
)

ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


TableComparisonArtifactResources = DatasetComparisonArtifactResources[pd.DataFrame]


class TableComparisonArtifact(
    DatasetComparisonArtifact[
        pd.DataFrame, TabularDataSpecProtocol, ArtifactHyperparamsT, ArtifactResultT
    ],
    Generic[ArtifactHyperparamsT, ArtifactResultT],
):
    @abstractmethod
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> ArtifactResultT: ...

    def _validate_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ls_features = [
            feature
            for feature in self._resource_spec.ls_features
            if feature in self._resource_spec.ls_cat_features
            or feature in self._resource_spec.ls_cts_features
        ]
        dataset_real_validated = TableValidator.validate(
            df=dataset_real,
            ls_features=ls_features,
            ls_cat_features=self._resource_spec.ls_cat_features,
            ls_cts_features=self._resource_spec.ls_cts_features,
        )
        dataset_synthetic_validated = TableValidator.validate(
            df=dataset_synthetic,
            ls_features=ls_features,
            ls_cat_features=self._resource_spec.ls_cat_features,
            ls_cts_features=self._resource_spec.ls_cts_features,
        )
        return dataset_real_validated, dataset_synthetic_validated


TableComparisonScore = TableComparisonArtifact[ArtifactHyperparamsT, float]
TableComparisonArray = TableComparisonArtifact[ArtifactHyperparamsT, ndarray]
TableComparisonPlot = TableComparisonArtifact[ArtifactHyperparamsT, Figure]
TableComparisonScoreCollection = TableComparisonArtifact[ArtifactHyperparamsT, Dict[str, float]]
TableComparisonArrayCollection = TableComparisonArtifact[ArtifactHyperparamsT, Dict[str, ndarray]]
TableComparisonPlotCollection = TableComparisonArtifact[ArtifactHyperparamsT, Dict[str, Figure]]
