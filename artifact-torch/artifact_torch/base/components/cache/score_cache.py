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
        return self._df_scores

    def clear(self) -> None:
        super().clear()
        self._df_scores = self._build_df(cache=self._cache)

    def append(self, items: Dict[str, float]) -> None:
        super().append(items)
        self._df_scores = self._build_df(self._cache)

    def get_latest_non_null(self, key: str) -> Optional[float]:
        if key not in self._df_scores.columns:
            return None
        series = self._df_scores[key].dropna()
        if series.empty:
            return None
        return float(series.iloc[-1])

    @staticmethod
    def _build_df(cache: Dict[str, List[Optional[float]]]) -> pd.DataFrame:
        sanitized: Dict[str, List[float]] = {
            k: [np.nan if v_item is None else v_item for v_item in v_list]
            for k, v_list in cache.items()
        }
        df = pd.DataFrame(sanitized).astype(pd.SparseDtype("float"))
        return df
