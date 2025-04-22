from dataclasses import dataclass
from typing import Literal, Type, TypeVar, Union

import pandas as pd

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.implementation.tabular.pairwise_correlation.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
    PairwiseCorrelationCalculator,
)
from artifact_core.libs.resource_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)
from artifact_core.libs.utils.vector_distance_calculator import (
    VectorDistanceMetric,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonScore,
)
from artifact_core.table_comparison.registries.scores.registry import (
    TableComparisonScoreRegistry,
    TableComparisonScoreType,
)

pairwiseCorrelationDistanceConfigT = TypeVar(
    "pairwiseCorrelationDistanceConfigT", bound="PairwiseCorrelationDistanceConfig"
)


@TableComparisonScoreRegistry.register_artifact_config(
    TableComparisonScoreType.PAIRWISE_CORRELATION_DISTANCE
)
@dataclass(frozen=True)
class PairwiseCorrelationDistanceConfig(ArtifactHyperparams):
    categorical_association_type: CategoricalAssociationType
    continuous_association_type: ContinuousAssociationType
    vector_distance_metric: VectorDistanceMetric

    @classmethod
    def build(
        cls: Type[pairwiseCorrelationDistanceConfigT],
        categorical_association_type: Union[
            CategoricalAssociationType, Literal["THEILS_U"], Literal["CRAMERS_V"]
        ],
        continuous_association_type: Union[
            ContinuousAssociationType, Literal["PEARSON"], Literal["SPEARMAN"], Literal["KENDALL"]
        ],
        vector_distance_metric: Union[
            VectorDistanceMetric,
            Literal["L2"],
            Literal["MAE"],
            Literal["RMSE"],
            Literal["COSINE_SIMILARITY"],
        ],
    ) -> pairwiseCorrelationDistanceConfigT:
        if isinstance(categorical_association_type, str):
            categorical_association_type = CategoricalAssociationType[categorical_association_type]
        if isinstance(continuous_association_type, str):
            continuous_association_type = ContinuousAssociationType[continuous_association_type]
        if isinstance(vector_distance_metric, str):
            vector_distance_metric = VectorDistanceMetric[vector_distance_metric]
        correlation_comparison_heatmap_hyperparams = cls(
            categorical_association_type=categorical_association_type,
            continuous_association_type=continuous_association_type,
            vector_distance_metric=vector_distance_metric,
        )
        return correlation_comparison_heatmap_hyperparams


@TableComparisonScoreRegistry.register_artifact(
    TableComparisonScoreType.PAIRWISE_CORRELATION_DISTANCE
)
class PairwiseCorrelationDistance(TableComparisonScore[PairwiseCorrelationDistanceConfig]):
    def __init__(
        self,
        resource_spec: TabularDataSpecProtocol,
        hyperparams: PairwiseCorrelationDistanceConfig,
    ):
        self._resource_spec = resource_spec
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
            ls_cat_features=self._resource_spec.ls_cat_features,
        )
        return pairwise_correlation_distance
