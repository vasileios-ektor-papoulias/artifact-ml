import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar

import numpy as np
import pandas as pd

from artifact_core.libs.data_spec.tabular.protocol import (
    TabularDataDType,
    TabularDataSpecProtocol,
)

tabular_dataset_types: Dict[str, TabularDataDType] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "object": object,
    # Date and time types (Python)
    "datetime.date": datetime.date,
    "datetime.datetime": datetime.datetime,
    # NumPy types
    "numpy.object_": np.object_,
    "numpy.int8": np.int8,
    "numpy.int16": np.int16,
    "numpy.int32": np.int32,
    "numpy.int64": np.int64,
    "numpy.uint8": np.uint8,
    "numpy.uint16": np.uint16,
    "numpy.uint32": np.uint32,
    "numpy.uint64": np.uint64,
    "numpy.float16": np.float16,
    "numpy.float32": np.float32,
    "numpy.float64": np.float64,
    "numpy.bool": np.bool,
    "numpy.bool_": np.bool_,
    "numpy.str_": np.str_,
    "numpy.datetime64": np.datetime64,
    # Pandas types
    "pandas.CategoricalDtypeType": pd.CategoricalDtype.type,
    "pandas.CategoricalDtype": pd.CategoricalDtype.type,
    "pandas.DatetimeTZDtype": pd.DatetimeTZDtype.type,
    "pandas.PeriodDtype": pd.PeriodDtype.type,
    "pandas.StringDtype": pd.StringDtype.type,
    "pandas.Timestamp": pd.Timestamp,
    "pandas.DatetimeIndex": pd.DatetimeIndex,
    "pandas.Timedelta": pd.Timedelta,
    "pandas.TimedeltaIndex": pd.TimedeltaIndex,
}


tabularDataSpecT = TypeVar("tabularDataSpecT", bound="TabularDataSpec")


class TabularDataSpec(TabularDataSpecProtocol):
    _types = tabular_dataset_types

    def __init__(self, internal_spec: Dict[str, Any]):
        self._internal_spec = internal_spec

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
        return {feat: self._continuous[feat]["dtype"] for feat in self.ls_cts_features}

    @property
    def ls_cat_features(self) -> List[str]:
        return [feat for feat in self._ls_features if feat in self._categorical]

    @property
    def n_cat_features(self) -> int:
        return len(self._categorical)

    @property
    def dict_cat_dtypes(self) -> Dict[str, TabularDataDType]:
        return {feat: self._categorical[feat]["dtype"] for feat in self.ls_cat_features}

    @property
    def cat_unique_map(self) -> Dict[str, List[str]]:
        return {feat: self._categorical[feat]["unique_categories"] for feat in self.ls_cat_features}

    @cat_unique_map.setter
    def cat_unique_map(self, unique_map: Dict[str, List[str]]) -> None:
        if not isinstance(unique_map, dict):
            raise ValueError("categorical_unique_map must be a dictionary.")
        for key, value in unique_map.items():
            if key not in self._categorical:
                raise ValueError(f"Feature '{key}' is not a categorical feature.")
            if not isinstance(value, list) or not all(
                isinstance(item, (str, int)) for item in value
            ):
                raise ValueError(
                    f"Unique categories for '{key}' must be a list of strings or integers."
                )
            self._categorical[key]["unique_categories"] = value

    @property
    def cat_unique_count_map(self) -> Dict[str, int]:
        return {
            feat: len(self._categorical[feat]["unique_categories"]) for feat in self.ls_cat_features
        }

    @property
    def ls_n_cat(self) -> List[int]:
        return [len(self._categorical[feat]["unique_categories"]) for feat in self.ls_cat_features]

    @property
    def _ls_features(self) -> List[str]:
        return self._internal_spec["features_order"]

    @_ls_features.setter
    def _ls_features(self, features_order: List[str]):
        self._internal_spec["features_order"] = features_order

    @property
    def _continuous(self) -> Dict[str, Any]:
        return self._internal_spec["continuous"]

    @property
    def _categorical(self) -> Dict[str, Any]:
        return self._internal_spec["categorical"]

    @classmethod
    def from_df(
        cls: Type[tabularDataSpecT],
        df: pd.DataFrame,
        ls_cts_features: Optional[List[str]] = None,
        ls_cat_features: Optional[List[str]] = None,
    ) -> tabularDataSpecT:
        data_spec = cls.build(
            ls_cts_features=ls_cts_features,
            ls_cat_features=ls_cat_features,
        )
        data_spec.fit(df=df)
        return data_spec

    @classmethod
    def build(
        cls: Type[tabularDataSpecT],
        ls_cts_features: Optional[List[str]] = None,
        ls_cat_features: Optional[List[str]] = None,
    ) -> tabularDataSpecT:
        if ls_cts_features is None:
            ls_cts_features = []
        if ls_cat_features is None:
            ls_cat_features = []
        cls._validate(ls_cat_features=ls_cat_features, ls_cts_features=ls_cts_features)
        internal_spec = cls._build_internal_spec(
            ls_cts_features=ls_cts_features,
            ls_cat_features=ls_cat_features,
        )
        data_spec = cls(internal_spec=internal_spec)
        return data_spec

    def fit(self, df: pd.DataFrame) -> None:
        set_cts_features, set_cat_features = self._partition_features(
            df=df,
            set_cts_features=set(self._internal_spec["continuous"].keys()),
            set_cat_features=set(self._internal_spec["categorical"].keys()),
        )
        for feature in set_cts_features:
            spec = self._get_continuous_feature_spec(sr_data=df[feature])
            self._continuous[feature] = spec
        for feature in set_cat_features:
            spec = self._get_categorical_feature_spec(sr_data=df[feature])
            self._categorical[feature] = spec
        self._ls_features = list(df.columns)

    def get_unique_categories(self, feature_name: str) -> List[str]:
        if feature_name not in self._categorical:
            raise ValueError(f"Feature '{feature_name}' is not a categorical feature.")
        return self._categorical[feature_name]["unique_categories"]

    def get_num_unique_categories(self, feature_name: str) -> int:
        return len(self.get_unique_categories(feature_name))

    def set_unique_categories(self, feature_name: str, unique_categories: List[str]) -> None:
        if feature_name not in self._categorical:
            raise ValueError(f"Feature '{feature_name}' is not a categorical feature.")
        self._categorical[feature_name]["unique_categories"] = unique_categories

    def export(self, filepath: Path):
        json_str = self.serialize()
        if filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as file:
            file.write(json_str)

    def serialize(self) -> str:
        serializable_dict = self._convert_to_serializable(self._internal_spec)
        return json.dumps(serializable_dict)

    @classmethod
    def load(cls: Type[tabularDataSpecT], filepath: Path) -> tabularDataSpecT:
        if filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")
        with open(filepath, "r") as json_file:
            json_str = json_file.read()
        return cls.deserialize(json_str=json_str)

    @classmethod
    def deserialize(cls: Type[tabularDataSpecT], json_str: str) -> tabularDataSpecT:
        loaded_dict = json.loads(json_str)
        internal_spec = cls._convert_from_serializable(loaded_dict)
        return cls(internal_spec=internal_spec)

    @classmethod
    def _partition_features(
        cls,
        df: pd.DataFrame,
        set_cts_features: Set[str],
        set_cat_features: Set[str],
    ) -> Tuple[Set[str], Set[str]]:
        set_cts_features = set_cts_features.copy()
        set_cat_features = set_cat_features.copy()
        prescribed_features_set = set_cts_features.union(set_cat_features)
        all_features_set = set(df.columns)
        if not prescribed_features_set.issubset(all_features_set):
            raise ValueError(
                f"Prescribed features {prescribed_features_set} "
                + "not found in dataset columns {all_features_set}"
            )
        continuous_features_default = cls._get_continuous_features_default(df=df)
        categorical_features_default = cls._get_categorical_features_default(df=df)
        for feature in continuous_features_default:
            if feature not in set_cat_features:
                set_cts_features.add(feature)
        for feature in categorical_features_default:
            if feature not in set_cts_features:
                set_cat_features.add(feature)
        return set_cts_features, set_cat_features

    @staticmethod
    def _get_continuous_feature_spec(sr_data: pd.Series) -> Dict[str, Any]:
        spec = {"dtype": sr_data.dtype.type}
        return spec

    @staticmethod
    def _get_categorical_feature_spec(sr_data: pd.Series) -> Dict[str, Any]:
        cats = sr_data.astype(str).dropna().unique().tolist()
        spec = {
            "dtype": sr_data.dtype.type,
            "unique_categories": cats,
        }
        return spec

    @staticmethod
    def _get_continuous_features_default(df: pd.DataFrame) -> List[str]:
        ls_cts_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        return ls_cts_features

    @staticmethod
    def _get_categorical_features_default(df: pd.DataFrame) -> List[str]:
        ls_cat_features = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        return ls_cat_features

    @classmethod
    def _build_internal_spec(
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
        internal_spec: Dict[str, Any] = {
            "continuous": {feat: cls._get_default_cts_spec() for feat in ls_cts_features},
            "categorical": {feat: cls._get_default_cat_spec() for feat in ls_cat_features},
            "features_order": ls_cts_features + ls_cat_features,
        }
        return internal_spec

    @staticmethod
    def _get_default_cts_spec() -> Dict[str, Any]:
        spec = {"dtype": float}
        return spec

    @staticmethod
    def _get_default_cat_spec() -> Dict[str, Any]:
        spec = {
            "dtype": str,
            "unique_categories": [],
        }
        return spec

    @staticmethod
    def _validate(ls_cat_features: List[str], ls_cts_features: List[str]):
        overlap = set(ls_cat_features).intersection(ls_cts_features)
        if overlap:
            raise ValueError(f"Categorical and continuous features overlap: {overlap}")

    @classmethod
    def _convert_to_serializable(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: cls._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, type):
            return cls._serialize_type(obj)
        else:
            return obj

    @classmethod
    def _convert_from_serializable(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if k == "dtype" and isinstance(v, str):
                    new_obj[k] = cls._deserialize_type(v)
                else:
                    new_obj[k] = cls._convert_from_serializable(v)
            return new_obj
        elif isinstance(obj, list):
            return [cls._convert_from_serializable(item) for item in obj]
        else:
            return obj

    @classmethod
    def _serialize_type(cls, type_obj: type) -> str:
        if type_obj.__module__ == "numpy":
            return f"numpy.{type_obj.__name__}"
        elif type_obj.__module__.startswith("pandas"):
            return f"pandas.{type_obj.__name__}"
        else:
            return type_obj.__name__

    @classmethod
    def _deserialize_type(cls, type_str: str) -> Any:
        if type_str in cls._types:
            return cls._types[type_str]
        return type_str
