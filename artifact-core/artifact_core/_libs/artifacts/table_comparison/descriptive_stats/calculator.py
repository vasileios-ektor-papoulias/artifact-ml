from typing import Dict, Mapping, Sequence, Set

import numpy as np
import pandas as pd

from artifact_core._libs.tools.calculators.descriptive_stats_calculator import (
    DescriptiveStatistic,
    DescriptiveStatsCalculator,
)
from artifact_core._libs.tools.calculators.score_juxtaposition_calculator import (
    ScoreJuxtapositionCalculator,
)


class TableStatsCalculator:
    @classmethod
    def compute_juxtaposition(
        cls,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        cts_features: Sequence[str],
        stat: DescriptiveStatistic,
    ) -> Mapping[str, np.ndarray]:
        dict_stats_real = cls._compute_stat_for_cts_features(
            df=df_real, cts_features=cts_features, stat=stat
        )
        dict_stats_synthetic = cls._compute_stat_for_cts_features(
            df=df_synthetic,
            cts_features=cts_features,
            stat=stat,
        )
        dict_juxtaposition_arrays = ScoreJuxtapositionCalculator.juxtapose_score_collections(
            scores_real=dict_stats_real,
            scores_synthetic=dict_stats_synthetic,
            keys=cts_features,
        )
        return dict_juxtaposition_arrays

    @classmethod
    def compute(
        cls,
        df: pd.DataFrame,
        cts_features: Sequence[str],
        stat: DescriptiveStatistic,
    ) -> Dict[str, float]:
        dict_stats = cls._compute_stat_for_cts_features(df=df, cts_features=cts_features, stat=stat)
        return dict_stats

    @classmethod
    def _compute_stat_for_cts_features(
        cls,
        df: pd.DataFrame,
        cts_features: Sequence[str],
        stat: DescriptiveStatistic,
    ) -> Dict[str, float]:
        cls._validate_no_missing_cols(set_cols=set(df.columns), set_cts_features=set(cts_features))
        dict_stats = {}
        for feature in cts_features:
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
