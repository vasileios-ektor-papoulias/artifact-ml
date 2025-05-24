from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class ScoreCache:
    def __init__(self):
        self._scores: Dict[str, List[float]] = {}
        self._keys_in_new_entry: List[str] = []
        self._n_entries = 0

    @property
    def is_empty(self) -> bool:
        return len(self._scores) == 0

    @property
    def n_entries(self) -> int:
        return self._n_entries

    @property
    def scores(self) -> pd.DataFrame:
        dict_scores = {
            score_key: pd.Series(score_values) for score_key, score_values in self._scores.items()
        }
        df_scores = pd.DataFrame(data=dict_scores).astype(pd.SparseDtype("float"))
        return df_scores

    def get_full_history(self, score_key: str) -> Optional[pd.Series]:
        ls_history = self._get_full_history(score_key=score_key)
        if ls_history is not None:
            return pd.Series(ls_history)

    def get_latest_value(self, score_key: str, default: Optional[float] = None) -> Optional[float]:
        full_history = self.get_full_history(score_key=score_key)
        if full_history is not None:
            full_history = full_history.dropna()
            if not full_history.empty:
                return full_history.iloc[-1]
        return default

    def clear(self):
        self._scores.clear()

    def finalize_entry(self):
        self._pad_keys(
            scores=self._scores,
            score_keys=[key for key in self._scores.keys() if key not in self._keys_in_new_entry],
        )
        self._keys_in_new_entry.clear()
        self._n_entries += 1

    def append(self, score_key: str, score_value: float):
        self._append(score_key=score_key, score_value=score_value)

    def append_multiple(self, scores: Dict[str, float]):
        for score_key, score_value in scores.items():
            self._append(score_key, score_value)

    def _get_full_history(self, score_key: str) -> Optional[List[float]]:
        return self._scores.get(score_key, None)

    def _append(self, score_key: str, score_value: float):
        if score_key not in self._scores.keys():
            self._scores[score_key] = self._get_empty_column(n_entries=self.n_entries)
        self._scores[score_key].append(score_value)
        self._keys_in_new_entry.append(score_key)

    @classmethod
    def _pad_keys(cls, scores: Dict[str, List[float]], score_keys: List[str]):
        for score_key in score_keys:
            ls_scores = scores[score_key]
            ls_scores.append(np.nan)

    @classmethod
    def _get_empty_column(cls, n_entries: int):
        return n_entries * [np.nan]
