import warnings
from typing import Optional, Sequence, Set, Tuple

import pandas as pd

from artifact_core._libs.tools.schema.feature_partition.feature_partition import FeaturePartition
from artifact_core._libs.tools.schema.feature_partition.inference_engine import (
    FeaturePartitionInferenceEngine,
    FeaturePartitionStrategy,
)


class FeaturePartitioner:
    @classmethod
    def partition_features(
        cls,
        df: pd.DataFrame,
        ls_cts_features: Optional[Sequence[str]] = None,
        ls_cat_features: Optional[Sequence[str]] = None,
        strategy: FeaturePartitionStrategy = FeaturePartitionStrategy.NATIVE_PANDAS,
        warn_on_missing: bool = True,
    ) -> FeaturePartition:
        prescribed_feature_partition = cls._get_prescribed_feature_partition(
            ls_cts_features=ls_cts_features, ls_cat_features=ls_cat_features
        )
        inferred_feature_partition = cls._infer_feature_partition(df=df, strategy=strategy)
        merged_feature_partition = cls._merge_feature_partitions(
            prescribed_feature_partition=prescribed_feature_partition,
            inferred_feature_partition=inferred_feature_partition,
            prescribed_precedence=True,
            warn_on_missing=warn_on_missing,
        )

        return merged_feature_partition

    @classmethod
    def _merge_feature_partitions(
        cls,
        prescribed_feature_partition: FeaturePartition,
        inferred_feature_partition: FeaturePartition,
        prescribed_precedence: bool = True,
        warn_on_missing: bool = True,
    ) -> FeaturePartition:
        set_all_features = set(inferred_feature_partition.ls_all_features)
        set_prescribed_features = set(prescribed_feature_partition.ls_all_features)
        set_missing_features = set_prescribed_features - set_all_features
        if warn_on_missing and set_missing_features:
            warnings.warn(
                f"Ignoring missing prescribed features: {sorted(set_missing_features)}. "
                f"Available dataset columns: {sorted(set_all_features)}",
                UserWarning,
            )
        set_cts_features, set_cat_features = cls._merge_prescribed_and_inferred(
            set_cts_prescribed=set(prescribed_feature_partition.ls_cts_features),
            set_cat_prescribed=set(prescribed_feature_partition.ls_cat_features),
            set_cts_inferred=set(inferred_feature_partition.ls_cts_features),
            set_cat_inferred=set(inferred_feature_partition.ls_cat_features),
            prescribed_precedence=prescribed_precedence,
        )
        set_cts_features = set_cts_features.intersection(set_all_features)
        set_cat_features = set_cat_features.intersection(set_all_features)
        merged_feature_partition = FeaturePartition.build(
            seq_all_features=list(set_all_features),
            seq_cts_features=list(set_cts_features),
            seq_cat_features=list(set_cat_features),
        )
        return merged_feature_partition

    @staticmethod
    def _merge_prescribed_and_inferred(
        set_cts_prescribed: Set[str],
        set_cat_prescribed: Set[str],
        set_cts_inferred: Set[str],
        set_cat_inferred: Set[str],
        prescribed_precedence: bool = True,
    ) -> Tuple[Set[str], Set[str]]:
        if prescribed_precedence:
            set_cts = (set_cts_inferred - set_cat_prescribed) | set_cts_prescribed
            set_cat = (set_cat_inferred - set_cts_prescribed) | set_cat_prescribed
        else:
            set_cts = (set_cts_prescribed - set_cat_inferred) | set_cts_inferred
            set_cat = (set_cat_prescribed - set_cts_inferred) | set_cat_inferred

        return set_cts, set_cat

    @staticmethod
    def _get_prescribed_feature_partition(
        ls_cts_features: Optional[Sequence[str]],
        ls_cat_features: Optional[Sequence[str]],
    ) -> FeaturePartition:
        set_cts_prescribed = set(ls_cts_features or [])
        set_cat_prescribed = set(ls_cat_features or [])
        set_prescribed = set_cts_prescribed.union(set_cat_prescribed)
        prescribed_partition = FeaturePartition.build(
            seq_all_features=list(set_prescribed),
            seq_cts_features=list(set_cts_prescribed),
            seq_cat_features=list(set_cat_prescribed),
        )
        return prescribed_partition

    @classmethod
    def _infer_feature_partition(
        cls, df: pd.DataFrame, strategy: FeaturePartitionStrategy
    ) -> FeaturePartition:
        return FeaturePartitionInferenceEngine.infer(df=df, strategy=strategy)
