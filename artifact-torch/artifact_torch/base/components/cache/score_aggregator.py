from enum import Enum

import pandas as pd


class AggregationType(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


class ScoreAggregator:
    @classmethod
    def aggregate_multiple(
        cls,
        df_histories: pd.DataFrame,
        aggregation: AggregationType = AggregationType.MEAN,
    ) -> pd.Series:
        sr_aggregated = df_histories.apply(
            lambda col: cls.aggregate(col, aggregation=aggregation), axis=0
        )
        return sr_aggregated

    @staticmethod
    def aggregate(
        sr_history: pd.Series,
        aggregation: AggregationType = AggregationType.MEAN,
    ) -> float:
        if aggregation == AggregationType.MEAN:
            aggregated = float(sr_history.mean())
        elif aggregation == AggregationType.MEDIAN:
            aggregated = float(sr_history.median())
        elif aggregation == AggregationType.SUM:
            aggregated = float(sr_history.sum())
        elif aggregation == AggregationType.MIN:
            aggregated = float(sr_history.min())
        elif aggregation == AggregationType.MAX:
            aggregated = float(sr_history.max())
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        return aggregated
