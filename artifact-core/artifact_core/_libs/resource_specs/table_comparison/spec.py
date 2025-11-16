from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Type, TypeVar

import pandas as pd

from artifact_core._interfaces.serializable import Serializable
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core._libs.tools.schema.data_types.typing import TabularDataDType
from artifact_core._libs.tools.schema.feature_partition.feature_partition import FeaturePartition
from artifact_core._libs.tools.schema.feature_partition.feature_partitioner import (
    FeaturePartitioner,
)
from artifact_core._libs.tools.schema.feature_partition.inference_engine import (
    FeaturePartitionStrategy,
)
from artifact_core._libs.tools.schema.feature_spec.categorical import CategoricalFeatureSpec
from artifact_core._libs.tools.schema.feature_spec.continuous import ContinuousFeatureSpec
from artifact_core._utils.collections.sequence_concatenator import SequenceConcatenator

TabularDataSpecT = TypeVar("TabularDataSpecT", bound="TabularDataSpec")


class TabularDataSpec(Serializable, TabularDataSpecProtocol):
    _feature_partition_strategy = FeaturePartitionStrategy.NATIVE_PANDAS
    _cts_specs_key = "continuous"
    _cat_specs_key = "categorical"
    _features_order_key = "features_order"

    def __init__(
        self,
        cts_feature_specs: Mapping[str, ContinuousFeatureSpec],
        cat_feature_specs: Mapping[str, CategoricalFeatureSpec],
        features_order: Sequence[str],
    ):
        self._cts_feature_specs = cts_feature_specs
        self._cat_feature_specs = cat_feature_specs
        self._features_order = features_order

    @classmethod
    def from_df(
        cls: Type[TabularDataSpecT],
        df: pd.DataFrame,
        cts_features: Optional[Sequence[str]] = None,
        cat_features: Optional[Sequence[str]] = None,
    ) -> TabularDataSpecT:
        spec = cls.build(
            cts_features=cts_features,
            cat_features=cat_features,
        )
        spec.fit(df=df)
        return spec

    @classmethod
    def build(
        cls: Type[TabularDataSpecT],
        cts_features: Optional[Sequence[str]] = None,
        cat_features: Optional[Sequence[str]] = None,
    ) -> TabularDataSpecT:
        if cts_features is None:
            cts_features = []
        if cat_features is None:
            cat_features = []
        hardcoded_spec = cls._build_hardcoded(
            cts_features=cts_features,
            cat_features=cat_features,
        )
        spec = cls(
            cts_feature_specs=hardcoded_spec[cls._cts_specs_key],
            cat_feature_specs=hardcoded_spec[cls._cat_specs_key],
            features_order=hardcoded_spec[cls._features_order_key],
        )
        return spec

    @property
    def features(self) -> Sequence[str]:
        return list(self._features_order)

    @property
    def n_features(self) -> int:
        return len(self._features_order)

    @property
    def cts_features(self) -> Sequence[str]:
        return [feat for feat in self._features_order if feat in self._cts_feature_specs]

    @property
    def n_cts_features(self) -> int:
        return len(self._cts_feature_specs)

    @property
    def cts_dtypes(self) -> Mapping[str, TabularDataDType]:
        return {feat: self._cts_feature_specs[feat].dtype for feat in self.cts_features}

    @property
    def cat_features(self) -> Sequence[str]:
        return [feat for feat in self._features_order if feat in self._cat_feature_specs]

    @property
    def n_cat_features(self) -> int:
        return len(self._cat_feature_specs)

    @property
    def cat_dtypes(self) -> Mapping[str, TabularDataDType]:
        return {feat: self._cat_feature_specs[feat].dtype for feat in self.cat_features}

    @property
    def cat_unique_map(self) -> Mapping[str, List[str]]:
        return {feat: self._cat_feature_specs[feat].ls_categories for feat in self.cat_features}

    @property
    def cat_unique_count_map(self) -> Mapping[str, int]:
        return {
            feat: len(self._cat_feature_specs[feat].ls_categories) for feat in self.cat_features
        }

    @property
    def seq_n_cat(self) -> Sequence[int]:
        return [len(self._cat_feature_specs[feat].ls_categories) for feat in self.cat_features]

    def fit(self, df: pd.DataFrame) -> None:
        self._cts_feature_specs = dict(self._cts_feature_specs)
        self._cat_feature_specs = dict(self._cat_feature_specs)
        self._assert_prescribed_features_exist(
            df=df,
            prescribed_features=set(self._cts_feature_specs.keys())
            | self._cat_feature_specs.keys(),
        )
        feature_partition = self._partition_features(
            df=df,
            set_cts_features=set(self._cts_feature_specs.keys()),
            set_cat_features=set(self._cat_feature_specs.keys()),
        )
        set_cts_features = set(feature_partition.ls_cts_features)
        set_cat_features = set(feature_partition.ls_cat_features)
        for feature in set_cts_features:
            self._cts_feature_specs[feature] = self._get_cts_spec_from_data(sr_data=df[feature])
        for feature in set_cat_features:
            self._cat_feature_specs[feature] = self._get_cat_spec_from_data(sr_data=df[feature])
            self._assert_known_categories(
                sr_data=df[feature], cat_feature_spec=self._cat_feature_specs[feature], name=feature
            )
        self._features_order = self._get_features_order_from_data(
            df_data=df, cont_set=set_cts_features, cat_set=set_cat_features
        )
        self._drop_missing_features(cont_set=set_cts_features, cat_set=set_cat_features)

    def get_unique_categories(self, feature_name: str) -> List[str]:
        if feature_name not in self._cat_feature_specs:
            raise ValueError(f"Feature '{feature_name}' is not a categorical feature.")
        return self._cat_feature_specs[feature_name].ls_categories

    def get_n_unique_categories(self, feature_name: str) -> int:
        return len(self.get_unique_categories(feature_name))

    def export(self, filepath: Path):
        json_str = self.serialize()
        if filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as file:
            file.write(json_str)

    @classmethod
    def load(cls: Type[TabularDataSpecT], filepath: Path) -> TabularDataSpecT:
        if filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")
        with open(filepath, "r") as json_file:
            json_str = json_file.read()
        return cls.deserialize(json_str=json_str)

    @classmethod
    def from_dict(cls: Type[TabularDataSpecT], data: Dict[str, Any]) -> TabularDataSpecT:
        dict_cts_raw: Dict[str, Any] = cls._get_from_data(key=cls._cts_specs_key, data=data)
        dict_cts_specs = {
            name: ContinuousFeatureSpec.from_dict(data=raw_spec)
            for name, raw_spec in dict_cts_raw.items()
        }
        dict_cat_raw: Dict[str, Any] = cls._get_from_data(key=cls._cat_specs_key, data=data)
        dict_cat_specs = {
            name: CategoricalFeatureSpec.from_dict(data=raw_spec)
            for name, raw_spec in dict_cat_raw.items()
        }
        features_order = cls._get_from_data(key=cls._features_order_key, data=data)
        return cls(
            cts_feature_specs=dict_cts_specs,
            cat_feature_specs=dict_cat_specs,
            features_order=list(features_order),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            self._cts_specs_key: {
                name: spec.to_dict() for name, spec in self._cts_feature_specs.items()
            },
            self._cat_specs_key: {
                name: spec.to_dict() for name, spec in self._cat_feature_specs.items()
            },
            self._features_order_key: list(self._features_order),
        }

    def _drop_missing_features(self, cont_set: Set[str], cat_set: Set[str]) -> None:
        self._cts_feature_specs = {
            feature_name: feature_spec
            for feature_name, feature_spec in self._cts_feature_specs.items()
            if feature_name in cont_set
        }
        self._cat_feature_specs = {
            feature_name: feature_spec
            for feature_name, feature_spec in self._cat_feature_specs.items()
            if feature_name in cat_set
        }
        self._features_order = [
            feature_name
            for feature_name in self._features_order
            if feature_name in cont_set or feature_name in cat_set
        ]

    def _assert_known_categories(
        self, sr_data: pd.Series, cat_feature_spec: CategoricalFeatureSpec, name: str
    ) -> None:
        idx_unknown = pd.Index(sr_data.dropna().astype(str).unique()).difference(
            pd.Index(cat_feature_spec.ls_categories)
        )
        if len(idx_unknown):
            raise ValueError(f"Categorical '{name}' has unseen values: {list(idx_unknown)}")

    @staticmethod
    def _assert_prescribed_features_exist(
        df: pd.DataFrame,
        prescribed_features: Iterable[str],
    ) -> None:
        prescribed: Set[str] = set(prescribed_features)
        missing = prescribed - set(df.columns)
        if missing:
            raise ValueError(f"Prescribed features {missing} not found in dataset columns")

    @staticmethod
    def _get_features_order_from_data(
        df_data: pd.DataFrame, cont_set: Set[str], cat_set: Set[str]
    ) -> List[str]:
        return [c for c in df_data.columns if c in cont_set or c in cat_set]

    @staticmethod
    def _get_cts_spec_from_data(sr_data: pd.Series) -> ContinuousFeatureSpec:
        spec = ContinuousFeatureSpec(dtype=sr_data.dtype.type)
        return spec

    @staticmethod
    def _get_cat_spec_from_data(sr_data: pd.Series) -> CategoricalFeatureSpec:
        ls_categories = list(sr_data.dropna().astype(str).unique())
        spec = CategoricalFeatureSpec(dtype=sr_data.dtype.type, ls_categories=ls_categories)
        return spec

    @classmethod
    def _partition_features(
        cls,
        df: pd.DataFrame,
        set_cts_features: Set[str],
        set_cat_features: Set[str],
    ) -> FeaturePartition:
        return FeaturePartitioner.partition_features(
            df=df,
            ls_cts_features=list(set_cts_features),
            ls_cat_features=list(set_cat_features),
            strategy=cls._feature_partition_strategy,
            warn_on_missing=True,
        )

    @classmethod
    def _build_hardcoded(
        cls,
        cts_features: Optional[Sequence[str]] = None,
        cat_features: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if cts_features is None:
            cts_features = []
        if cat_features is None:
            cat_features = []
        cont_set = set(cts_features)
        cat_set = set(cat_features)
        intersection = cont_set.intersection(cat_set)
        if intersection:
            raise ValueError(f"Categorical and continuous features overlap: {intersection}")
        hardcoded_spec: Dict[str, Any] = {
            cls._cts_specs_key: {feat: cls._build_null_cts_spec() for feat in cts_features},
            cls._cat_specs_key: {feat: cls._build_null_cat_spec() for feat in cat_features},
            cls._features_order_key: SequenceConcatenator.concatenate(cts_features, cat_features),
        }
        return hardcoded_spec

    @staticmethod
    def _build_null_cts_spec() -> ContinuousFeatureSpec:
        return ContinuousFeatureSpec(dtype=float)

    @staticmethod
    def _build_null_cat_spec() -> CategoricalFeatureSpec:
        return CategoricalFeatureSpec(dtype=str, ls_categories=[])
