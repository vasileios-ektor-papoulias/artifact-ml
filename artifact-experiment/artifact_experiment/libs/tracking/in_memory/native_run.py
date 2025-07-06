from typing import Dict, List


class InMemoryNativeRun:
    def __init__(self, experiment_id: str, run_id: str):
        self._is_active = True
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._ls_scores = []
        self._ls_arrays = []
        self._ls_plots = []
        self._ls_score_collections = []
        self._ls_array_collections = []
        self._ls_plot_collections = []
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
    def ls_scores(self) -> List[float]:
        return self._ls_scores

    @property
    def ls_arrays(self) -> List:
        return self._ls_arrays

    @property
    def ls_plots(self) -> List:
        return self._ls_plots

    @property
    def ls_score_collections(self) -> List:
        return self._ls_score_collections

    @property
    def ls_array_collections(self) -> List:
        return self._ls_array_collections

    @property
    def ls_plot_collections(self) -> List:
        return self._ls_plot_collections

    @property
    def uploaded_files(self) -> List[Dict[str, str]]:
        return self._uploaded_files
