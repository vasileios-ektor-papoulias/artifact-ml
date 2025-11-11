from copy import deepcopy
from typing import Any, Dict, List

from artifact_core._base.artifact_dependencies import ArtifactResult
from matplotlib.figure import Figure
from numpy import ndarray


class InMemoryRun:
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
        return deepcopy(self._experiment_id)

    @property
    def run_id(self) -> str:
        return deepcopy(self._run_id)

    @property
    def is_active(self) -> bool:
        return self._is_active

    @is_active.setter
    def is_active(self, is_active: bool):
        self._is_active = is_active

    @property
    def dict_scores(self) -> Dict[str, float]:
        return deepcopy(self._dict_scores)

    @property
    def dict_arrays(self) -> Dict[str, ndarray]:
        return deepcopy(self._dict_arrays)

    @property
    def dict_plots(self) -> Dict[str, Figure]:
        return deepcopy(self._dict_plots)

    @property
    def dict_score_collections(self) -> Dict[str, Dict[str, float]]:
        return deepcopy(self._dict_score_collections)

    @property
    def dict_array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return deepcopy(self._dict_array_collections)

    @property
    def dict_plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return deepcopy(self._dict_plot_collections)

    @property
    def uploaded_files(self) -> List[Dict[str, str]]:
        return deepcopy(self._uploaded_files)

    def log_score(self, key: str, score: float):
        self._log(key=key, value=score, store=self._dict_scores)

    def log_array(self, key: str, array: ndarray):
        self._log(key=key, value=array, store=self._dict_arrays)

    def log_plot(self, key: str, plot: Figure):
        self._log(key=key, value=plot, store=self._dict_plots)

    def log_score_collection(self, key: str, score_collection: Dict[str, float]):
        self._log(key=key, value=score_collection, store=self._dict_score_collections)

    def log_array_collection(self, key: str, array_collection: Dict[str, ndarray]):
        self._log(key=key, value=array_collection, store=self._dict_array_collections)

    def log_plot_collection(self, key: str, plot_collection: Dict[str, Figure]):
        self._log(key=key, value=plot_collection, store=self._dict_plot_collections)

    def upload(self, path_source: str, dir_target: str):
        upload_entry = self._format_upload_entry(path_source=path_source, dir_target=dir_target)
        self._uploaded_files.append(upload_entry)

    @staticmethod
    def _log(key: str, value: ArtifactResult, store: Dict[str, Any]):
        if key in store.keys():
            raise ValueError(f"Artifact already registered at path {key}")
        store[key] = value

    @staticmethod
    def _format_upload_entry(path_source: str, dir_target: str) -> Dict[str, str]:
        dict_upload_entry = {"path_source": path_source, "dir_target": dir_target}
        return dict_upload_entry
