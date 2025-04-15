from dataclasses import dataclass

import pandas as pd

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.data_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)
from artifact_core.libs.implementation.pairwsie_correlation.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
    PairwiseCorrelationCalculator,
)
from artifact_core.libs.utils.vector_distance_calculator import (
    VectorDistanceMetric,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonScore,
)
@dataclass(frozen=True)
class PairwiseCorrelationDistanceConfig(ArtifactHyperparams):
    categorical_association_type: CategoricalAssociationType
    continuous_association_type: ContinuousAssociationType
    vector_distance_metric: VectorDistanceMetric

    def __post_init__(self):
        if isinstance(self.categorical_association_type, str):
            object.__setattr__(
                self,
                "categorical_association_type",
                CategoricalAssociationType[self.categorical_association_type],
            )
        if isinstance(self.continuous_association_type, str):
            object.__setattr__(
                self,
                "continuous_association_type",
                ContinuousAssociationType[self.continuous_association_type],
            )
        if isinstance(self.vector_distance_metric, str):
            object.__setattr__(
                self,
                "vector_distance_metric",
                VectorDistanceMetric[self.vector_distance_metric],
            )
class PairwiseCorrelationDistance(TableComparisonScore[PairwiseCorrelationDistanceConfig]):
    def __init__(
        self,
        data_spec: TabularDataSpecProtocol,
        hyperparams: PairwiseCorrelationDistanceConfig,
    ):
        self._data_spec = data_spec
        self._hyperparams = hyperparams

    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> float:
        pairwise_correlation_distance = PairwiseCorrelationCalculator.compute_correlation_distance(
            categorical_correlation_type=self._hyperparams.categorical_association_type,
            continuous_correlation_type=self._hyperparams.continuous_association_type,
            distance_metric=self._hyperparams.vector_distance_metric,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cat_features=self._data_spec.ls_cat_features,
        )
        return pairwise_correlation_distance
