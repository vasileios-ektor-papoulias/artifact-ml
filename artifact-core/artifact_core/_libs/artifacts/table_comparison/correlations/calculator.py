from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal, Sequence, Union

import pandas as pd
from dython.nominal import associations
from matplotlib.axes import Axes

from artifact_core._libs.tools.calculators.vector_distance_calculator import (
    VectorDistanceCalculator,
    VectorDistanceMetric,
)

CategoricalAssociationTypeLiteral = Literal["THEILS_U", "CRAMERS_V"]
ContinuousAssociationTypeLiteral = Literal["PEARSON", "SPEARMAN", "KENDALL"]


class CategoricalAssociationType(Enum):
    THEILS_U = "theil"
    CRAMERS_V = "cramer"


class ContinuousAssociationType(Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


@dataclass
class DythonAssociationConfig:
    categorical_continuous_correlation: (
        Callable[[pd.Series, pd.Series], float] | Literal["correlation_ratio"]
    ) = "correlation_ratio"
    mark_columns_with_feature_type: bool = True
    compute_only: bool = True


class CorrelationCalculator:
    _dict_assoc_df_key: str = "corr"
    _dict_assoc_ax_key: str = "ax"
    _dython_association_config = DythonAssociationConfig()

    @classmethod
    def compute_df_correlations(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset: pd.DataFrame,
        cat_features: Sequence[str],
    ) -> pd.DataFrame:
        dict_assoc = cls._compute(
            dataset=dataset,
            cat_features=cat_features,
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
        )
        df_correlations = dict_assoc[cls._dict_assoc_df_key]
        assert isinstance(df_correlations, pd.DataFrame)
        return df_correlations

    @classmethod
    def compute_df_correlation_difference(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        cat_features: Sequence[str],
    ) -> pd.DataFrame:
        df_correlations_real = cls.compute_df_correlations(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_real,
            cat_features=cat_features,
        )
        df_correlations_synthetic = cls.compute_df_correlations(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_synthetic,
            cat_features=cat_features,
        )
        df_correlation_difference = abs(df_correlations_real - df_correlations_synthetic)
        return df_correlation_difference

    @classmethod
    def compute_correlation_distance(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        distance_metric: VectorDistanceMetric,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        cat_features: Sequence[str],
    ) -> float:
        arr_correlations_real = cls.compute_df_correlations(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_real,
            cat_features=cat_features,
        ).values
        arr_correlations_synthetic = (
            cls.compute_df_correlations(
                categorical_correlation_type=categorical_correlation_type,
                continuous_correlation_type=continuous_correlation_type,
                dataset=dataset_synthetic,
                cat_features=cat_features,
            )
        ).values
        correlation_distance = VectorDistanceCalculator.compute(
            metric=distance_metric,
            v_1=arr_correlations_real,
            v_2=arr_correlations_synthetic,
        )
        return correlation_distance

    @classmethod
    def _compute(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset: pd.DataFrame,
        cat_features: Sequence[str],
    ) -> Dict[str, Union[pd.DataFrame, Axes]]:
        dict_assoc = associations(
            dataset=dataset,
            nominal_columns=list(cat_features),
            nom_nom_assoc=categorical_correlation_type.value,
            num_num_assoc=continuous_correlation_type.value,
            nom_num_assoc=cls._dython_association_config.categorical_continuous_correlation,
            mark_columns=cls._dython_association_config.mark_columns_with_feature_type,
            compute_only=cls._dython_association_config.compute_only,
        )
        return dict_assoc
