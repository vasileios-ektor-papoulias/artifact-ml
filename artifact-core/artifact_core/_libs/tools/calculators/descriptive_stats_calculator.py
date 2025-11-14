from enum import Enum
from typing import Literal, Mapping, Sequence

import pandas as pd

DescriptiveStatisticLiteral = Literal["MEAN", "STD", "VARIANCE", "MEDIAN", "Q1", "Q3", "MIN", "MAX"]


class DescriptiveStatistic(Enum):
    MEAN = "mean"
    STD = "std"
    VARIANCE = "variance"
    MEDIAN = "median"
    Q1 = "q1"
    Q3 = "q3"
    MIN = "min"
    MAX = "max"


class DescriptiveStatsCalculator:
    @staticmethod
    def compute_stat(sr_cts_data: pd.Series, stat: DescriptiveStatistic) -> float:
        if stat == DescriptiveStatistic.MEAN:
            return sr_cts_data.mean()
        elif stat == DescriptiveStatistic.STD:
            return sr_cts_data.std()
        elif stat == DescriptiveStatistic.VARIANCE:
            return sr_cts_data.var()  # type: ignore
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

    @classmethod
    def compute_dict_stats(
        cls, sr_cts_data: pd.Series, stats: Sequence[DescriptiveStatistic]
    ) -> Mapping[DescriptiveStatistic, float]:
        dict_stats: Mapping[DescriptiveStatistic, float] = {}
        for stat in stats:
            val = cls.compute_stat(sr_cts_data=sr_cts_data, stat=stat)
            dict_stats[stat] = float(val) if pd.notna(val) else float("nan")
        return dict_stats

    @staticmethod
    def compute_sr_stats(
        sr_cts_data: pd.Series, stats: Sequence[DescriptiveStatistic]
    ) -> pd.Series:
        dict_stats = DescriptiveStatsCalculator.compute_dict_stats(
            sr_cts_data=sr_cts_data, stats=stats
        )
        sr_stats = pd.Series(data={stat.name: val for stat, val in dict_stats.items()})
        return sr_stats
