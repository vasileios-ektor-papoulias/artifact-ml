from enum import Enum
from typing import Literal

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
