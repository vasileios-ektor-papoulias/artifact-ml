from dataclasses import dataclass
from typing import Type, TypeVar, Union

import pandas as pd

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import Score
from artifact_core._libs.artifacts.table_comparison.correlations.calculator import (
    CategoricalAssociationType,
    CategoricalAssociationTypeLiteral,
    ContinuousAssociationType,
    ContinuousAssociationTypeLiteral,
    CorrelationCalculator,
)
from artifact_core._libs.tools.calculators.vector_distance_calculator import (
    VectorDistanceMetric,
    VectorDistanceMetricLiteral,
)
from artifact_core.table_comparison._artifacts.base import TableComparisonScore
from artifact_core.table_comparison._registries.scores import TableComparisonScoreRegistry
from artifact_core.table_comparison._types.scores import TableComparisonScoreType

PairwiseCorrelationDistanceHyperparamsT = TypeVar(
    "PairwiseCorrelationDistanceHyperparamsT", bound="CorrelationDistanceScoreHyperparams"
)


@TableComparisonScoreRegistry.register_artifact_hyperparams(
    TableComparisonScoreType.CORRELATION_DISTANCE
)
@dataclass(frozen=True)
class CorrelationDistanceScoreHyperparams(ArtifactHyperparams):
    categorical_association_type: CategoricalAssociationType
    continuous_association_type: ContinuousAssociationType
    vector_distance_metric: VectorDistanceMetric

    @classmethod
    def build(
        cls: Type[PairwiseCorrelationDistanceHyperparamsT],
        categorical_association_type: Union[
            CategoricalAssociationType, CategoricalAssociationTypeLiteral
        ],
        continuous_association_type: Union[
            ContinuousAssociationType, ContinuousAssociationTypeLiteral
        ],
        vector_distance_metric: Union[VectorDistanceMetric, VectorDistanceMetricLiteral],
    ) -> PairwiseCorrelationDistanceHyperparamsT:
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


@TableComparisonScoreRegistry.register_artifact(TableComparisonScoreType.CORRELATION_DISTANCE)
class CorrelationDistanceScore(TableComparisonScore[CorrelationDistanceScoreHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Score:
        pairwise_correlation_distance = CorrelationCalculator.compute_correlation_distance(
            categorical_correlation_type=self._hyperparams.categorical_association_type,
            continuous_correlation_type=self._hyperparams.continuous_association_type,
            distance_metric=self._hyperparams.vector_distance_metric,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cat_features=self._resource_spec.cat_features,
        )
        return pairwise_correlation_distance
