import datetime
from typing import Type, Union

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
