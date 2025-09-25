import datetime
from typing import Dict, Type, TypeVar, Union

import numpy as np
import pandas as pd

TabularDataDType = Union[
    Type[int],
    Type[float],
    Type[str],
    Type[bool],
    Type[object],
    Type[np.generic],
    Type[np.dtype],
    Type[pd.api.extensions.ExtensionDtype],
    Type[datetime.date],
    Type[datetime.datetime],
    Type[pd.Timestamp],
    Type[pd.DatetimeIndex],
    Type[pd.Timedelta],
    Type[pd.TimedeltaIndex],
]

TABULAR_DATA_TYPE_REGISTRY: Dict[str, TabularDataDType] = {
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

TabularDataTypeSerializerT = TypeVar(
    "TabularDataTypeSerializerT", bound="TabularDataTypeSerializer"
)


class TabularDataTypeSerializer:
    _type_registry = TABULAR_DATA_TYPE_REGISTRY

    @classmethod
    def get_dtype(cls: Type[TabularDataTypeSerializerT], dtype_name: str) -> TabularDataDType:
        if dtype_name not in cls._type_registry:
            raise ValueError(f"Unknown dtype during deserialization: {dtype_name}")
        dtype = cls._type_registry[dtype_name]
        return dtype

    @classmethod
    def get_dtype_name(cls: Type[TabularDataTypeSerializerT], dtype: TabularDataDType) -> str:
        for registered_dtype_name, registered_dtype in cls._type_registry.items():
            if dtype is registered_dtype:
                return registered_dtype_name
        raise ValueError(f"Unsupported dtype for serialization: {dtype}")
