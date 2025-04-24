import datetime
from typing import Dict, List, Protocol, Type, Union

import numpy as np
import pandas as pd

from artifact_core.base.artifact_dependencies import ResourceSpecProtocol

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


class TabularDataSpecProtocol(ResourceSpecProtocol, Protocol):
    @property
    def ls_features(self) -> List[str]: ...

    @property
    def n_features(self) -> int: ...

    @property
    def ls_cts_features(self) -> List[str]: ...

    @property
    def n_cts_features(self) -> int: ...

    @property
    def dict_cts_dtypes(self) -> Dict[str, TabularDataDType]: ...

    @property
    def ls_cat_features(self) -> List[str]: ...

    @property
    def n_cat_features(self) -> int: ...

    @property
    def dict_cat_dtypes(self) -> Dict[str, TabularDataDType]: ...

    @property
    def cat_unique_map(self) -> Dict[str, List[str]]: ...

    @property
    def cat_unique_count_map(self) -> Dict[str, int]: ...
