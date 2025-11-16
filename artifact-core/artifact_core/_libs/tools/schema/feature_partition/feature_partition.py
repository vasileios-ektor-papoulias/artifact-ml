from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple, Type, TypeVar

from artifact_core._utils.collections.deduplicator import Deduplicator


class FeatureType(Enum):
    UNKNOWN = "UNKNOWN"
    CONTINUOUS = "CONTINUOUS"
    CATEGORICAL = "CATEGORICAL"


FeaturePartitionT = TypeVar("FeaturePartitionT", bound="FeaturePartition")


@dataclass(frozen=True)
class FeaturePartition:
    _dict_partition: Dict[str, FeatureType]

    @property
    def dict_partition(self) -> Dict[str, FeatureType]:
        return deepcopy(self._dict_partition)

    @classmethod
    def build(
        cls: Type[FeaturePartitionT],
        seq_all_features: Sequence[str],
        seq_cts_features: Optional[Sequence[str]] = None,
        seq_cat_features: Optional[Sequence[str]] = None,
    ) -> FeaturePartitionT:
        if seq_cts_features is None:
            seq_cts_features = []
        if seq_cat_features is None:
            seq_cat_features = []
        set_cts_features, set_cat_features, set_unknown_features = cls._validate(
            seq_all_features=seq_all_features,
            seq_cts_features=seq_cts_features,
            seq_cat_features=seq_cat_features,
        )
        dict_partition: Dict[str, FeatureType] = {}
        for feature in set_cts_features:
            dict_partition[feature] = FeatureType.CONTINUOUS
        for feature in set_cat_features:
            dict_partition[feature] = FeatureType.CATEGORICAL
        for feature in set_unknown_features:
            dict_partition[feature] = FeatureType.UNKNOWN
        return cls(_dict_partition=dict_partition)

    @classmethod
    def from_dict(
        cls: Type[FeaturePartitionT], dict_partition: Dict[str, FeatureType]
    ) -> FeaturePartitionT:
        seq_all_features = cls._get_features_group_from_dict(dict_partition=dict_partition)
        seq_cts_features = cls._get_features_group_from_dict(
            dict_partition=dict_partition, feature_type=FeatureType.CONTINUOUS
        )
        seq_cat_features = cls._get_features_group_from_dict(
            dict_partition=dict_partition, feature_type=FeatureType.CATEGORICAL
        )
        return cls.build(
            seq_all_features=seq_all_features,
            seq_cts_features=seq_cts_features,
            seq_cat_features=seq_cat_features,
        )

    @classmethod
    def build_empty(
        cls: Type[FeaturePartitionT],
    ) -> FeaturePartitionT:
        return cls.from_dict(dict_partition={})

    @classmethod
    def build_all_cts(
        cls: Type[FeaturePartitionT], seq_all_features: Sequence[str]
    ) -> FeaturePartitionT:
        return cls.build(
            seq_all_features=seq_all_features,
            seq_cts_features=seq_all_features,
            seq_cat_features=[],
        )

    @classmethod
    def build_all_cat(
        cls: Type[FeaturePartitionT], seq_all_features: Sequence[str]
    ) -> FeaturePartitionT:
        return cls.build(
            seq_all_features=seq_all_features,
            seq_cts_features=[],
            seq_cat_features=seq_all_features,
        )

    @classmethod
    def build_all_unknown(
        cls: Type[FeaturePartitionT], seq_all_features: Sequence[str]
    ) -> FeaturePartitionT:
        return cls.build(
            seq_all_features=seq_all_features, seq_cts_features=[], seq_cat_features=[]
        )

    @property
    def ls_all_features(self) -> List[str]:
        return self._get_features_group_from_dict(dict_partition=self._dict_partition)

    @property
    def ls_cts_features(self) -> List[str]:
        return self._get_features_group_from_dict(
            dict_partition=self._dict_partition, feature_type=FeatureType.CONTINUOUS
        )

    @property
    def ls_cat_features(self) -> List[str]:
        return self._get_features_group_from_dict(
            dict_partition=self._dict_partition, feature_type=FeatureType.CATEGORICAL
        )

    @property
    def unknown_features(self) -> List[str]:
        return self._get_features_group_from_dict(
            dict_partition=self._dict_partition, feature_type=FeatureType.UNKNOWN
        )

    @staticmethod
    def _get_features_group_from_dict(
        dict_partition: Dict[str, FeatureType], feature_type: Optional[FeatureType] = None
    ) -> List[str]:
        if feature_type is None:
            ls_features = list(dict_partition.keys())
        else:
            ls_features = [
                feature
                for feature, stored_type in dict_partition.items()
                if stored_type == feature_type
            ]
        return ls_features

    @staticmethod
    def _validate(
        seq_all_features: Sequence[str],
        seq_cts_features: Sequence[str],
        seq_cat_features: Sequence[str],
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        set_all_features = Deduplicator.to_set(seq=seq_all_features)
        set_cts_features = Deduplicator.to_set(seq=seq_cts_features)
        set_cat_features = Deduplicator.to_set(seq=seq_cat_features)
        set_missing_features = (set_cts_features | set_cat_features) - set_all_features
        if set_missing_features:
            raise ValueError(
                "Prescribed features not present in the full feature set: "
                f"{sorted(set_missing_features)}. All features: {sorted(set_all_features)}"
            )
        set_overlapping_features = set_cts_features & set_cat_features
        if set_overlapping_features:
            raise ValueError(
                "Features cannot be both continuous and categorical: "
                f"{sorted(set_overlapping_features)}"
            )
        set_unknown_features = set_all_features - set_cts_features - set_cat_features
        return set_cts_features, set_cat_features, set_unknown_features
