from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, TypeVar

import pandas as pd

from artifact_core.libs.resource_spec.tabular.categorical_column_spec import CategoricalColumnSpec
from artifact_core.libs.resource_spec.tabular.continuous_column_spec import ContinuousColumnSpec
from artifact_core.libs.resource_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)
from artifact_core.libs.resource_spec.tabular.types import (
    TabularDataDType,
)
from artifact_core.libs.types.serializable import Serializable
from artifact_core.libs.utils.feature_partition.feature_partition import FeaturePartition
from artifact_core.libs.utils.feature_partition.feature_partitioner import FeaturePartitioner
from artifact_core.libs.utils.feature_partition.inference_engine import FeaturePartitionStrategy

TabularDataSpecT = TypeVar("TabularDataSpecT", bound="TabularDataSpec")


class TabularDataSpec(Serializable, TabularDataSpecProtocol):
    _feature_partition_strategy = FeaturePartitionStrategy.NATIVE_PANDAS
    _cts_specs_key = "continuous"
    _cat_specs_key = "categorical"
    _features_order_key = "features_order"

    def __init__(
        self,
        dict_cts_specs: Dict[str, ContinuousColumnSpec],
        dict_cat_specs: Dict[str, CategoricalColumnSpec],
        features_order: List[str],
    ):
        self._dict_cts_specs = dict_cts_specs
        self._dict_cat_specs = dict_cat_specs
        self._features_order = features_order

    @classmethod
    def from_df(
        cls: Type[TabularDataSpecT],
        df: pd.DataFrame,
        ls_cts_features: Optional[List[str]] = None,
        ls_cat_features: Optional[List[str]] = None,
    ) -> TabularDataSpecT:
        spec = cls.build(
            ls_cts_features=ls_cts_features,
            ls_cat_features=ls_cat_features,
        )
        spec.fit(df=df)
        return spec

    @classmethod
    def build(
        cls: Type[TabularDataSpecT],
        ls_cts_features: Optional[List[str]] = None,
        ls_cat_features: Optional[List[str]] = None,
    ) -> TabularDataSpecT:
        if ls_cts_features is None:
            ls_cts_features = []
        if ls_cat_features is None:
            ls_cat_features = []
        hardcoded_spec = cls._build_hardcoded(
            ls_cts_features=ls_cts_features,
            ls_cat_features=ls_cat_features,
        )
        spec = cls(
            dict_cts_specs=hardcoded_spec[cls._cts_specs_key],
            dict_cat_specs=hardcoded_spec[cls._cat_specs_key],
            features_order=hardcoded_spec[cls._features_order_key],
        )
        return spec

    @property
    def ls_features(self) -> List[str]:
        return self._ls_features.copy()

    @property
    def n_features(self) -> int:
        return len(self._ls_features)

    @property
    def ls_cts_features(self) -> List[str]:
        return [feat for feat in self._ls_features if feat in self._continuous]

    @property
    def n_cts_features(self) -> int:
        return len(self._continuous)

    @property
    def dict_cts_dtypes(self) -> Dict[str, TabularDataDType]:
        return {feat: self._continuous[feat].dtype for feat in self.ls_cts_features}

    @property
    def ls_cat_features(self) -> List[str]:
        return [feat for feat in self._ls_features if feat in self._categorical]

    @property
    def n_cat_features(self) -> int:
        return len(self._categorical)

    @property
    def dict_cat_dtypes(self) -> Dict[str, TabularDataDType]:
        return {feat: self._categorical[feat].dtype for feat in self.ls_cat_features}

    @property
    def cat_unique_map(self) -> Dict[str, List[str]]:
        return {feat: self._categorical[feat].ls_categories for feat in self.ls_cat_features}

    @property
    def cat_unique_count_map(self) -> Dict[str, int]:
        return {feat: len(self._categorical[feat].ls_categories) for feat in self.ls_cat_features}

    @property
    def ls_n_cat(self) -> List[int]:
        return [len(self._categorical[feat].ls_categories) for feat in self.ls_cat_features]

    @property
    def _ls_features(self) -> List[str]:
        return self._features_order

    @_ls_features.setter
    def _ls_features(self, features_order: List[str]):
        self._features_order = features_order

    @property
    def _continuous(self) -> Dict[str, ContinuousColumnSpec]:
        return self._dict_cts_specs

    @_continuous.setter
    def _continuous(self, continuous: Dict[str, ContinuousColumnSpec]):
        self._dict_cts_specs = continuous

    @property
    def _categorical(self) -> Dict[str, CategoricalColumnSpec]:
        return self._dict_cat_specs

    @_categorical.setter
    def _categorical(self, categorical: Dict[str, CategoricalColumnSpec]):
        self._dict_cat_specs = categorical

    def fit(self, df: pd.DataFrame) -> None:
        feature_partition = self._partition_features(
            df=df,
            set_cts_features=set(self._dict_cts_specs.keys()),
            set_cat_features=set(self._dict_cat_specs.keys()),
        )
        set_cts_features = set(feature_partition.ls_cts_features)
        set_cat_features = set(feature_partition.ls_cat_features)
        for feature in set_cts_features:
            stored_spec = self._continuous.get(feature)
            if stored_spec is None:
                self._continuous[feature] = self._get_cts_spec_from_data(sr_data=df[feature])
        for feature in set_cat_features:
            stored_spec = self._categorical.get(feature)
            if stored_spec is None:
                self._categorical[feature] = self._get_cat_spec_from_data(sr_data=df[feature])
            self._assert_known_categories(
                sr_data=df[feature], cat_feature_spec=self._categorical[feature], name=feature
            )
        self._features_order = self._get_features_order_from_data(
            df_data=df, cont_set=set_cts_features, cat_set=set_cat_features
        )
        self._drop_missing_features(cont_set=set_cts_features, cat_set=set_cat_features)

    def get_unique_categories(self, feature_name: str) -> List[str]:
        if feature_name not in self._categorical:
            raise ValueError(f"Feature '{feature_name}' is not a categorical feature.")
        return self._categorical[feature_name].ls_categories

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
            name: ContinuousColumnSpec.from_dict(data=raw_spec)
            for name, raw_spec in dict_cts_raw.items()
        }
        dict_cat_raw: Dict[str, Any] = cls._get_from_data(key=cls._cat_specs_key, data=data)
        dict_cat_specs = {
            name: CategoricalColumnSpec.from_dict(data=raw_spec)
            for name, raw_spec in dict_cat_raw.items()
        }
        features_order = cls._get_from_data(key=cls._features_order_key, data=data)
        return cls(
            dict_cts_specs=dict_cts_specs,
            dict_cat_specs=dict_cat_specs,
            features_order=list(features_order),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            self._cts_specs_key: {name: spec.to_dict() for name, spec in self._continuous.items()},
            self._cat_specs_key: {name: spec.to_dict() for name, spec in self._categorical.items()},
            self._features_order_key: list(self._ls_features),
        }

    def _drop_missing_features(self, cont_set: Set[str], cat_set: Set[str]) -> None:
        self._continuous = {
            feature_name: feature_spec
            for feature_name, feature_spec in self._continuous.items()
            if feature_name in cont_set
        }
        self._categorical = {
            feature_name: feature_spec
            for feature_name, feature_spec in self._categorical.items()
            if feature_name in cat_set
        }
        self._features_order = [
            feature_name
            for feature_name in self._features_order
            if feature_name in cont_set or feature_name in cat_set
        ]

    def _assert_known_categories(
        self, sr_data: pd.Series, cat_feature_spec: CategoricalColumnSpec, name: str
    ) -> None:
        idx_not_in_range = pd.Index(sr_data.dropna().astype(str).unique()).difference(
            pd.Index(cat_feature_spec.ls_categories)
        )
        if len(idx_not_in_range):
            raise ValueError(f"Categorical '{name}' has unseen values: {list(idx_not_in_range)}")

    @staticmethod
    def _get_features_order_from_data(
        df_data: pd.DataFrame, cont_set: Set[str], cat_set: Set[str]
    ) -> List[str]:
        return [c for c in df_data.columns if c in cont_set or c in cat_set]

    @staticmethod
    def _get_cts_spec_from_data(sr_data: pd.Series) -> ContinuousColumnSpec:
        spec = ContinuousColumnSpec(dtype=sr_data.dtype.type)
        return spec

    @staticmethod
    def _get_cat_spec_from_data(sr_data: pd.Series) -> CategoricalColumnSpec:
        ls_categories = list(map(str, sr_data.dropna().unique().tolist()))
        spec = CategoricalColumnSpec(dtype=str, ls_categories=ls_categories)
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
        ls_cts_features: Optional[List[str]] = None,
        ls_cat_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if ls_cts_features is None:
            ls_cts_features = []
        if ls_cat_features is None:
            ls_cat_features = []
        cont_set = set(ls_cts_features)
        cat_set = set(ls_cat_features)
        intersection = cont_set.intersection(cat_set)
        if intersection:
            raise ValueError(f"Features cannot be both continuous and categorical: {intersection}")
        hardcoded_spec: Dict[str, Any] = {
            cls._cts_specs_key: {feat: cls._build_null_cts_spec() for feat in ls_cts_features},
            cls._cat_specs_key: {feat: cls._build_null_cat_spec() for feat in ls_cat_features},
            cls._features_order_key: ls_cts_features + ls_cat_features,
        }
        return hardcoded_spec

    @staticmethod
    def _build_null_cts_spec() -> ContinuousColumnSpec:
        return ContinuousColumnSpec(dtype=float)

    @staticmethod
    def _build_null_cat_spec() -> CategoricalColumnSpec:
        return CategoricalColumnSpec(dtype=str, ls_categories=[])
