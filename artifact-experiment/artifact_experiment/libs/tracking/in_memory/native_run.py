from typing import Dict, List

from matplotlib.figure import Figure
from numpy import ndarray


class InMemoryNativeRun:
    def __init__(self, experiment_id: str, run_id: str):
        self._is_active = True
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._dict_scores: Dict[str, float] = {}
        self._dict_arrays: Dict[str, ndarray] = {}
        self._dict_plots: Dict[str, Figure] = {}
        self._dict_score_collections: Dict[str, Dict[str, float]] = {}
        self._dict_array_collections: Dict[str, Dict[str, ndarray]] = {}
        self._dict_plot_collections: Dict[str, Dict[str, Figure]] = {}
        self._uploaded_files: List[Dict[str, str]] = []

    @property
    def experiment_id(self) -> str:
        return str(self._experiment_id)

    @property
    def run_id(self) -> str:
        return str(self._run_id)

    @property
    def is_active(self) -> bool:
        return self._is_active

    @is_active.setter
    def is_active(self, is_active: bool):
        self._is_active = is_active

    @property
    def dict_scores(self) -> Dict[str, float]:
        return self._dict_scores

    @property
    def dict_arrays(self) -> Dict[str, ndarray]:
        return self._dict_arrays

    @property
    def dict_plots(self) -> Dict[str, Figure]:
        return self._dict_plots

    @property
    def dict_score_collections(self) -> Dict[str, Dict[str, float]]:
        return self._dict_score_collections

    @property
    def dict_array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return self._dict_array_collections

    @property
    def dict_plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return self._dict_plot_collections

    @property
    def uploaded_files(self) -> List[Dict[str, str]]:
        return self._uploaded_files
