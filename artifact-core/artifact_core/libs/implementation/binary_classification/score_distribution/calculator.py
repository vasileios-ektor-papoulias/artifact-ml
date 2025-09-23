from typing import Dict, Hashable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from artifact_core.libs.implementation.binary_classification.score_distribution.partitioner import (
    BinarySampleSplit,
)
from artifact_core.libs.implementation.binary_classification.score_distribution.sampler import (
    ScoreDistributionSampler,
)
from artifact_core.libs.utils.descriptive_stats_calculator import (
    DescriptiveStatistic,
    DescriptiveStatsCalculator,
)


class ScoreDistributionStatsCalculator:
    @classmethod
    def compute(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        stats: Sequence[DescriptiveStatistic],
        splits: List[BinarySampleSplit],
    ) -> pd.DataFrame:
        ls_split_stats: List[pd.Series] = [
            cls._compute_for_split(
                id_to_is_pos=id_to_is_pos,
                id_to_prob_pos=id_to_prob_pos,
                stats=stats,
                split=split,
            )
            for split in splits
        ]
        if not ls_split_stats:
            df = cls._format_result_empty(stats=stats)
        else:
            df = cls._format_result(ls_split_stats=ls_split_stats, splits=splits)
        return df

    @classmethod
    def _compute_for_split(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        stats: Sequence[DescriptiveStatistic],
        split: BinarySampleSplit,
    ) -> pd.Series:
        samples = ScoreDistributionSampler.get_sample(
            id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos, split=split
        )
        sr_probs = pd.Series(samples, name=f"prob_pos[{split.name}]")
        sr_stats = cls._compute_sr_stats(sr_cts_data=sr_probs, stats=stats)
        return sr_stats

    @staticmethod
    def _compute_sr_stats(
        sr_cts_data: pd.Series, stats: Sequence[DescriptiveStatistic]
    ) -> pd.Series:
        dict_stats: Dict[str, float] = {}
        for stat in stats:
            value = DescriptiveStatsCalculator.compute_stat(sr_cts_data=sr_cts_data, stat=stat)
            dict_stats[stat.name] = float(value) if pd.notna(value) else np.nan
        sr_stats = pd.Series(data=dict_stats)
        return sr_stats

    @staticmethod
    def _format_result(
        ls_split_stats: List[pd.Series],
        splits: List[BinarySampleSplit],
    ) -> pd.DataFrame:
        df = pd.concat(ls_split_stats, axis=1)
        df.columns = [split.name for split in splits]
        return df

    @staticmethod
    def _format_result_empty(
        stats: Sequence[DescriptiveStatistic],
    ) -> pd.DataFrame:
        idx = [stat.name for stat in stats]
        df = pd.DataFrame(index=idx)
        return df
