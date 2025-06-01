from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from artifact_torch.base.components.cache.cache import AlignedCache


class ScoreCache(AlignedCache[float]):
    def __init__(self):
        super().__init__()
        self._df_scores: pd.DataFrame = self._build_df(self._cache)

    @property
    def scores(self) -> pd.DataFrame:
        df = self._build_df(self._cache)
        self._df_scores = df
        return df

    def clear(self) -> None:
        super().clear()
        self._df_scores = self._build_df(cache=self._cache)

    def append(self, items: Dict[str, float]) -> None:
        super().append(items)
        self._df_scores = self._build_df(self._cache)

    @staticmethod
    def _build_df(cache: Dict[str, List[Optional[float]]]) -> pd.DataFrame:
        series_dict: Dict[str, pd.Series] = {k: pd.Series(v) for k, v in cache.items()}
        df = pd.DataFrame(series_dict).replace({None: np.nan}).astype(pd.SparseDtype("float"))
        return df
