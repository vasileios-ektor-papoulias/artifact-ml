from copy import deepcopy
from typing import Any, Dict, List, Mapping, Sequence

from artifact_core.typing import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)


class InMemoryRun:
    def __init__(self, experiment_id: str, run_id: str):
        self._is_active = True
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._dict_scores: Dict[str, Score] = {}
        self._dict_arrays: Dict[str, Array] = {}
        self._dict_plots: Dict[str, Plot] = {}
        self._dict_score_collections: Dict[str, ScoreCollection] = {}
        self._dict_array_collections: Dict[str, ArrayCollection] = {}
        self._dict_plot_collections: Dict[str, PlotCollection] = {}
        self._ls_uploaded_files: List[Dict[str, str]] = []

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
    def scores(self) -> Mapping[str, float]:
        return deepcopy(self._dict_scores)

    @property
    def arrays(self) -> Mapping[str, Array]:
        return deepcopy(self._dict_arrays)

    @property
    def plots(self) -> Mapping[str, Plot]:
        return deepcopy(self._dict_plots)

    @property
    def score_collections(self) -> Mapping[str, ScoreCollection]:
        return deepcopy(self._dict_score_collections)

    @property
    def array_collections(self) -> Mapping[str, ArrayCollection]:
        return deepcopy(self._dict_array_collections)

    @property
    def plot_collections(self) -> Mapping[str, PlotCollection]:
        return deepcopy(self._dict_plot_collections)

    @property
    def uploaded_files(self) -> Sequence[Mapping[str, str]]:
        return deepcopy(self._ls_uploaded_files)

    def log_score(self, key: str, score: Score):
        self._log(key=key, value=score, store=self._dict_scores)

    def log_array(self, key: str, array: Array):
        self._log(key=key, value=array, store=self._dict_arrays)

    def log_plot(self, key: str, plot: Plot):
        self._log(key=key, value=plot, store=self._dict_plots)

    def log_score_collection(self, key: str, score_collection: ScoreCollection):
        self._log(key=key, value=score_collection, store=self._dict_score_collections)

    def log_array_collection(self, key: str, array_collection: ArrayCollection):
        self._log(key=key, value=array_collection, store=self._dict_array_collections)

    def log_plot_collection(self, key: str, plot_collection: PlotCollection):
        self._log(key=key, value=plot_collection, store=self._dict_plot_collections)

    def upload(self, path_source: str, dir_target: str):
        upload_entry = self._format_upload_entry(path_source=path_source, dir_target=dir_target)
        self._ls_uploaded_files.append(upload_entry)

    @staticmethod
    def _log(key: str, value: ArtifactResult, store: Dict[str, Any]):
        if key in store.keys():
            raise ValueError(f"Artifact already registered at path {key}")
        store[key] = value

    @staticmethod
    def _format_upload_entry(path_source: str, dir_target: str) -> Dict[str, str]:
        upload_entry = {"path_source": path_source, "dir_target": dir_target}
        return upload_entry
