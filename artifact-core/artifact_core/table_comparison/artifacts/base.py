from abc import abstractmethod
from typing import Dict, Generic, Optional, Tuple, TypeVar

import pandas as pd
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResult,
)
from artifact_core.core.dataset_comparison.artifact import (
    DatasetComparisonArtifact,
    DatasetComparisonArtifactResources,
)
from artifact_core.libs.data_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)
from artifact_core.libs.validation.table_validator import (
    TableValidator,
)

artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)
artifactHyperparamsT = TypeVar("artifactHyperparamsT", bound="ArtifactHyperparams")


class TableComparisonArtifact(
    DatasetComparisonArtifact[
        pd.DataFrame,
        artifactResultT,
        artifactHyperparamsT,
        TabularDataSpecProtocol,
    ],
    Generic[artifactResultT, artifactHyperparamsT],
):
    def __init__(
        self,
        data_spec: TabularDataSpecProtocol,
        hyperparams: Optional[artifactHyperparamsT] = None,
    ):
        self._data_spec = data_spec
        self._hyperparams = hyperparams

    @abstractmethod
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> artifactResultT: ...

    def _validate_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ls_features = [
            feature
            for feature in self._data_spec.ls_features
            if feature in self._data_spec.ls_cat_features
            or feature in self._data_spec.ls_cts_features
        ]
        dataset_real_validated = TableValidator.validate(
            df=dataset_real,
            ls_features=ls_features,
            ls_cat_features=self._data_spec.ls_cat_features,
            ls_cts_features=self._data_spec.ls_cts_features,
        )
        dataset_synthetic_validated = TableValidator.validate(
            df=dataset_synthetic,
            ls_features=ls_features,
            ls_cat_features=self._data_spec.ls_cat_features,
            ls_cts_features=self._data_spec.ls_cts_features,
        )
        return dataset_real_validated, dataset_synthetic_validated


TableComparisonArtifactResources = DatasetComparisonArtifactResources[pd.DataFrame]

TableComparisonScore = TableComparisonArtifact[float, artifactHyperparamsT]
TableComparisonArray = TableComparisonArtifact[ndarray, artifactHyperparamsT]
TableComparisonPlot = TableComparisonArtifact[Figure, artifactHyperparamsT]
TableComparisonScoreCollection = TableComparisonArtifact[Dict[str, float], artifactHyperparamsT]
TableComparisonArrayCollection = TableComparisonArtifact[Dict[str, ndarray], artifactHyperparamsT]
TableComparisonPlotCollection = TableComparisonArtifact[Dict[str, Figure], artifactHyperparamsT]
