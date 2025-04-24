from enum import Enum
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from artifact_core.libs.utils.score_juxtaposition import (
    JuxtapositionArrayCalculator,
)


class DescriptiveStatistic(Enum):
    MEAN = "mean"
    STD = "std"
    VARIANCE = "variance"
    MEDIAN = "median"
    Q1 = "q1"
    Q3 = "q3"
    MIN = "min"
    MAX = "max"


class DescriptiveStatisticsCalculator:
    @classmethod
    def compute_juxtaposition(
        cls,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        ls_cts_features: List[str],
        stat: DescriptiveStatistic,
    ) -> Dict[str, np.ndarray]:
        dict_stats_real = cls._compute_stat_for_cts_features(
            df=df_real, ls_cts_features=ls_cts_features, stat=stat
        )
        dict_stats_synthetic = cls._compute_stat_for_cts_features(
            df=df_synthetic,
            ls_cts_features=ls_cts_features,
            stat=stat,
        )
        dict_juxtaposition_arrays = JuxtapositionArrayCalculator.juxtapose_score_collections(
            dict_scores_real=dict_stats_real,
            dict_scores_synthetic=dict_stats_synthetic,
            ls_keys=ls_cts_features,
        )
        return dict_juxtaposition_arrays

    @classmethod
    def compute(
        cls,
        df: pd.DataFrame,
        ls_cts_features: List[str],
        stat: DescriptiveStatistic,
    ) -> Dict[str, float]:
        dict_stats = cls._compute_stat_for_cts_features(df, ls_cts_features, stat)
        return dict_stats

    @classmethod
    def _compute_stat_for_cts_features(
        cls,
        df: pd.DataFrame,
        ls_cts_features: List[str],
        stat: DescriptiveStatistic,
    ) -> Dict[str, float]:
        cls._validate_no_missing_cols(
            set_cols=set(df.columns), set_cts_features=set(ls_cts_features)
        )
        dict_stats = {}
        for feature in ls_cts_features:
            dict_stats[feature] = DescriptiveStatisticsCalculator._compute_stat(df[feature], stat)
        return dict_stats

    @staticmethod
    def _compute_stat(sr_cts_data: pd.Series, stat: DescriptiveStatistic) -> float:
        if stat == DescriptiveStatistic.MEAN:
            return sr_cts_data.mean()
        elif stat == DescriptiveStatistic.STD:
            return sr_cts_data.std()
        elif stat == DescriptiveStatistic.VARIANCE:
            return sr_cts_data.var()
        elif stat == DescriptiveStatistic.MEDIAN:
            return sr_cts_data.median()
        elif stat == DescriptiveStatistic.Q1:
            return sr_cts_data.quantile(0.25)
        elif stat == DescriptiveStatistic.Q3:
            return sr_cts_data.quantile(0.75)
        elif stat == DescriptiveStatistic.MIN:
            return sr_cts_data.min()
        elif stat == DescriptiveStatistic.MAX:
            return sr_cts_data.max()
        else:
            raise ValueError(f"Unsupported statistic: {stat}")

    @staticmethod
    def _validate_no_missing_cols(
        set_cols: Set[str],
        set_cts_features: Set[str],
    ):
        difference = set_cts_features.difference(set_cols)
        if difference:
            raise ValueError(f"Missing columns: {difference}.")
