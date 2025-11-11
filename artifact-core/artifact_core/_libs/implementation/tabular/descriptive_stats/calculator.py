from typing import Dict, List, Set

import numpy as np
import pandas as pd

from artifact_core._libs.utils.calculators.descriptive_stats_calculator import (
    DescriptiveStatistic,
    DescriptiveStatsCalculator,
)
from artifact_core._libs.utils.calculators.score_juxtaposition_calculator import (
    ScoreJuxtapositionCalculator,
)


class TableStatsCalculator:
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
        dict_juxtaposition_arrays = ScoreJuxtapositionCalculator.juxtapose_score_collections(
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
        dict_stats = cls._compute_stat_for_cts_features(
            df=df, ls_cts_features=ls_cts_features, stat=stat
        )
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
            dict_stats[feature] = DescriptiveStatsCalculator.compute_stat(
                sr_cts_data=df[feature], stat=stat
            )
        return dict_stats

    @staticmethod
    def _validate_no_missing_cols(
        set_cols: Set[str],
        set_cts_features: Set[str],
    ):
        difference = set_cts_features.difference(set_cols)
        if difference:
            raise ValueError(f"Missing columns: {difference}.")
