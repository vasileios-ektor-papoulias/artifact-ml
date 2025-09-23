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
    def compute_as_df(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        stats: Sequence[DescriptiveStatistic],
        splits: Sequence[BinarySampleSplit],
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
    def compute_as_dict(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        stats: Sequence[DescriptiveStatistic],
        splits: Sequence[BinarySampleSplit],
    ) -> Dict[DescriptiveStatistic, Dict[BinarySampleSplit, float]]:
        if not stats or not splits:
            return {}
        df = cls.compute_as_df(
            id_to_is_pos=id_to_is_pos,
            id_to_prob_pos=id_to_prob_pos,
            stats=stats,
            splits=list(splits),
        )
        dict_result: Dict[DescriptiveStatistic, Dict[BinarySampleSplit, float]] = {}
        for stat in stats:
            sr_stats_by_split = df[stat.name].astype(float)
            dict_stat_by_split = {
                split: float(sr_stats_by_split.loc[split.name]) for split in splits
            }
            dict_result[stat] = dict_stat_by_split
        return dict_result

    @classmethod
    def compute_by_split(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        stats: Sequence[DescriptiveStatistic],
        split: BinarySampleSplit,
    ) -> Dict[DescriptiveStatistic, float]:
        if not stats:
            return {}
        df = cls.compute_as_df(
            id_to_is_pos=id_to_is_pos,
            id_to_prob_pos=id_to_prob_pos,
            stats=stats,
            splits=[split],
        )
        sr_stats = df[split.name].astype(float)
        dict_stats = {stat: float(sr_stats.loc[stat.name]) for stat in stats}
        return dict_stats

    @classmethod
    def compute_by_stat(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        stat: DescriptiveStatistic,
        splits: Sequence[BinarySampleSplit],
    ) -> Dict[BinarySampleSplit, float]:
        if not splits:
            return {}
        df = cls.compute_as_df(
            id_to_is_pos=id_to_is_pos,
            id_to_prob_pos=id_to_prob_pos,
            stats=[stat],
            splits=list(splits),
        )
        sr_stats = df.loc[stat.name].astype(float)
        dict_stats = {split: float(sr_stats.loc[split.name]) for split in splits}
        return dict_stats

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
        ls_split_stats: Sequence[pd.Series],
        splits: Sequence[BinarySampleSplit],
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
