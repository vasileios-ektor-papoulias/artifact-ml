from enum import Enum

import pandas as pd

from artifact_core._libs.tools.schema.feature_partition.feature_partition import FeaturePartition


class FeaturePartitionStrategy(Enum):
    NATIVE_PANDAS = "NATIVE_PANDAS"


class FeaturePartitionInferenceEngine:
    @classmethod
    def infer(
        cls,
        df: pd.DataFrame,
        strategy: FeaturePartitionStrategy = FeaturePartitionStrategy.NATIVE_PANDAS,
    ) -> FeaturePartition:
        if strategy is FeaturePartitionStrategy.NATIVE_PANDAS:
            return cls._infer_native_pandas(df=df)
        raise ValueError(f"Unsupported partition strategy: {strategy}")

    @staticmethod
    def _infer_native_pandas(df: pd.DataFrame) -> FeaturePartition:
        set_all_features = set(df.columns)
        set_cts_features = {col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])}
        set_cat_features = {col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])}
        return FeaturePartition.build(
            seq_all_features=list(set_all_features),
            seq_cts_features=list(set_cts_features),
            seq_cat_features=list(set_cat_features),
        )
