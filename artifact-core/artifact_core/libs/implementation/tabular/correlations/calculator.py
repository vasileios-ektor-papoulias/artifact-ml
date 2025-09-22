from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Union

import pandas as pd
from dython.nominal import associations
from matplotlib.axes import Axes

from artifact_core.libs.utils.vector_distance_calculator import (
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
    dict_assoc_df_key = "corr"
    dict_assoc_ax_key = "ax"
    categorical_continuous_correlation = "correlation_ratio"
    mark_columns_with_feature_type = True


class CorrelationCalculator:
    _dython_association_config = DythonAssociationConfig()

    @classmethod
    def compute_df_correlations(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset: pd.DataFrame,
        ls_cat_features: List[str],
    ) -> pd.DataFrame:
        dict_assoc = cls._compute(
            dataset=dataset,
            ls_cat_features=ls_cat_features,
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
        )
        df_correlations = dict_assoc[cls._dython_association_config.dict_assoc_df_key]
        assert isinstance(df_correlations, pd.DataFrame)
        return df_correlations

    @classmethod
    def compute_df_correlation_difference(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_cat_features: List[str],
    ) -> pd.DataFrame:
        df_correlations_real = cls.compute_df_correlations(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_real,
            ls_cat_features=ls_cat_features,
        )
        df_correlations_synthetic = cls.compute_df_correlations(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_synthetic,
            ls_cat_features=ls_cat_features,
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
        ls_cat_features: List[str],
    ) -> float:
        arr_correlations_real = cls.compute_df_correlations(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_real,
            ls_cat_features=ls_cat_features,
        ).values
        arr_correlations_synthetic = (
            cls.compute_df_correlations(
                categorical_correlation_type=categorical_correlation_type,
                continuous_correlation_type=continuous_correlation_type,
                dataset=dataset_synthetic,
                ls_cat_features=ls_cat_features,
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
        ls_cat_features: List[str],
    ) -> Dict[str, Union[pd.DataFrame, Axes]]:
        dict_assoc = associations(
            dataset=dataset,
            nominal_columns=ls_cat_features,
            nom_nom_assoc=categorical_correlation_type.value,
            num_num_assoc=continuous_correlation_type.value,
            nom_num_assoc=cls._dython_association_config.categorical_continuous_correlation,
            mark_columns=cls._dython_association_config.mark_columns_with_feature_type,
            compute_only=True,
        )
        return dict_assoc
