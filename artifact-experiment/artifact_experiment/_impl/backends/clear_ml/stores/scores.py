from copy import deepcopy
from typing import Dict, List, Mapping, Sequence, Type, TypeVar

from artifact_experiment._impl.backends.clear_ml.stores.store import ClearMLStore

ClearMLScoreT = TypeVar("ClearMLScoreT", bound="ClearMLScore")
ClearMLScoreStoreT = TypeVar("ClearMLScoreStoreT", bound="ClearMLScoreStore")


class ClearMLScoreSeries:
    def __init__(self, raw_series_data: Mapping[str, Sequence[float]]):
        self._data = raw_series_data

    @property
    def x(self) -> List[float]:
        return list(self._data["x"])

    @property
    def y(self) -> List[float]:
        return list(self._data["y"])

    @property
    def n_entries(self) -> int:
        return len(self.x)

    def __len__(self) -> int:
        return self.n_entries


class ClearMLScore:
    def __init__(self, dict_score_data: Dict[str, ClearMLScoreSeries]):
        self._dict_score_data = dict_score_data

    @classmethod
    def build(
        cls: Type[ClearMLScoreT], raw_score_data: Mapping[str, Mapping[str, Sequence[float]]]
    ) -> ClearMLScoreT:
        dict_score_data = {
            series_name: ClearMLScoreSeries(raw_series_data=raw_series_data)
            for series_name, raw_series_data in raw_score_data.items()
        }
        score = cls(dict_score_data=dict_score_data)
        return score

    @property
    def n_series(self) -> int:
        return len(self._dict_score_data)

    @property
    def dict_series(self) -> Dict[str, ClearMLScoreSeries]:
        return deepcopy(self._dict_score_data)

    def __len__(self) -> int:
        return self.n_series

    def __getitem__(self, series_name: str) -> ClearMLScoreSeries:
        return self.get_series(series_name=series_name)

    def get_series(self, series_name: str) -> ClearMLScoreSeries:
        try:
            return self._dict_score_data[series_name]
        except KeyError:
            raise KeyError(f"series: {series_name} not found in score.")


class ClearMLScoreStore(ClearMLStore[ClearMLScore]):
    def __init__(self, root_dir: str, dict_store_data: Dict[str, ClearMLScore]):
        super().__init__(root_dir=root_dir)
        self._dict_store_data = dict_store_data

    @classmethod
    def build(
        cls: Type[ClearMLScoreStoreT],
        raw_store_data: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
        root_dir: str = "",
    ) -> ClearMLScoreStoreT:
        if root_dir is None:
            root_dir = ""
        dict_store_data = {
            score_name: ClearMLScore.build(raw_score_data=raw_score_data)
            for score_name, raw_score_data in raw_store_data.items()
            if score_name.startswith(root_dir)
        }
        store = cls(root_dir=root_dir, dict_store_data=dict_store_data)
        return store

    @property
    def n_entries(self) -> int:
        return len(self._dict_store_data)

    def __getitem__(self, path: str) -> ClearMLScore:
        return self.get(path=path)

    def _get(self, path: str) -> ClearMLScore:
        try:
            return self._dict_store_data[path]
        except KeyError:
            raise KeyError(f"score: {path} not found in store.")
