from abc import abstractmethod
from typing import Generic, Tuple, TypeVar

import pandas as pd

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)
from artifact_core._domains.dataset_comparison.artifact import (
    DatasetComparisonArtifact,
    DatasetComparisonArtifactResources,
)
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core._libs.validation.table_comparison.table_validator import TableValidator

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
            for feature in self._resource_spec.features
            if feature in self._resource_spec.cat_features
            or feature in self._resource_spec.cts_features
        ]
        dataset_real_validated = TableValidator.validate(
            df=dataset_real,
            ls_features=ls_features,
            ls_cat_features=self._resource_spec.cat_features,
            ls_cts_features=self._resource_spec.cts_features,
        )
        dataset_synthetic_validated = TableValidator.validate(
            df=dataset_synthetic,
            ls_features=ls_features,
            ls_cat_features=self._resource_spec.cat_features,
            ls_cts_features=self._resource_spec.cts_features,
        )
        return dataset_real_validated, dataset_synthetic_validated


TableComparisonScore = TableComparisonArtifact[ArtifactHyperparamsT, Score]
TableComparisonArray = TableComparisonArtifact[ArtifactHyperparamsT, Array]
TableComparisonPlot = TableComparisonArtifact[ArtifactHyperparamsT, Plot]
TableComparisonScoreCollection = TableComparisonArtifact[ArtifactHyperparamsT, ScoreCollection]
TableComparisonArrayCollection = TableComparisonArtifact[ArtifactHyperparamsT, ArrayCollection]
TableComparisonPlotCollection = TableComparisonArtifact[ArtifactHyperparamsT, PlotCollection]
