from pathlib import Path
from typing import Any, Dict, List

from artifact_experiment.base.tracking.backend.adapter import RunAdapter
from artifact_experiment.libs.tracking.in_memory.native_run import InMemoryRun


class InMemoryRunAdapter(RunAdapter[InMemoryRun]):
    @property
    def experiment_id(self) -> str:
        return self._native_run.experiment_id

    @property
    def run_id(self) -> str:
        return self._native_run.run_id

    @property
    def is_active(self) -> bool:
        return self._native_run.is_active

    @property
    def uploaded_files(self):
        return self._native_run.uploaded_files

    @property
    def dict_scores(self) -> Dict[str, float]:
        return self._native_run.dict_scores

    @property
    def ls_scores(self) -> List[float]:
        return list(self.dict_scores.values())

    @property
    def n_scores(self) -> int:
        return len(self.ls_scores)

    @property
    def dict_arrays(self) -> Dict[str, Array]:
        return self._native_run.dict_arrays

    @property
    def ls_arrays(self) -> List[Array]:
        return list(self.dict_arrays.values())

    @property
    def n_arrays(self) -> int:
        return len(self.ls_arrays)

    @property
    def dict_plots(self) -> Dict[str, Figure]:
        return self._native_run.dict_plots

    @property
    def ls_plots(self) -> List[Figure]:
        return list(self.dict_plots.values())

    @property
    def n_plots(self) -> int:
        return len(self.ls_plots)

    @property
    def dict_score_collections(self) -> Dict[str, Dict[str, float]]:
        return self._native_run.dict_score_collections

    @property
    def ls_score_collections(self) -> List[Dict[str, float]]:
        return list(self.dict_score_collections.values())

    @property
    def n_score_collections(self) -> int:
        return len(self.ls_score_collections)

    @property
    def dict_array_collections(self) -> Dict[str, Dict[str, Array]]:
        return self._native_run.dict_array_collections

    @property
    def ls_array_collections(self) -> List[Dict[str, Array]]:
        return list(self.dict_array_collections.values())

    @property
    def n_array_collections(self) -> int:
        return len(self.ls_array_collections)

    @property
    def dict_plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return self._native_run.dict_plot_collections

    @property
    def ls_plot_collections(self) -> List[Dict[str, Figure]]:
        return list(self.dict_plot_collections.values())

    @property
    def n_plot_collections(self) -> int:
        return len(self.ls_plot_collections)

    def stop(self):
        self._native_run.is_active = False

    def log_score(self, path: str, score: float):
        key = self._get_store_key(path=path)
        self._native_run.log_score(key=key, score=score)

    def log_array(self, path: str, array: Array):
        key = self._get_store_key(path=path)
        self._native_run.log_array(key=key, array=array)

    def log_plot(self, path: str, plot: Figure):
        key = self._get_store_key(path=path)
        self._native_run.log_plot(key=key, plot=plot)

    def log_score_collection(self, path: str, score_collection: Dict[str, float]):
        key = self._get_store_key(path=path)
        self._native_run.log_score_collection(key=key, score_collection=score_collection)

    def log_array_collection(self, path: str, array_collection: Dict[str, Array]):
        key = self._get_store_key(path=path)
        self._native_run.log_array_collection(key=key, array_collection=array_collection)

    def log_plot_collection(self, path: str, plot_collection: Dict[str, Figure]):
        key = self._get_store_key(path=path)
        self._native_run.log_plot_collection(key=key, plot_collection=plot_collection)

    def upload(self, path_source: str, dir_target: str):
        self._native_run.upload(path_source=path_source, dir_target=dir_target)

    def search_score_store(self, artifact_path: str) -> List[str]:
        return self._search_store(artifact_path=artifact_path, store=self.dict_scores)

    def search_array_store(self, artifact_path: str) -> List[str]:
        return self._search_store(artifact_path=artifact_path, store=self.dict_arrays)

    def search_plot_store(self, artifact_path: str) -> List[str]:
        return self._search_store(artifact_path=artifact_path, store=self.dict_plots)

    def search_score_collection_store(self, artifact_path: str) -> List[str]:
        return self._search_store(artifact_path=artifact_path, store=self.dict_score_collections)

    def search_array_collection_store(self, artifact_path: str) -> List[str]:
        return self._search_store(artifact_path=artifact_path, store=self.dict_array_collections)

    def search_plot_collection_store(self, artifact_path: str) -> List[str]:
        return self._search_store(artifact_path=artifact_path, store=self.dict_plot_collections)

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> InMemoryRun:
        return InMemoryRun(experiment_id=experiment_id, run_id=run_id)

    def _start(self, run_id: str):
        if self.run_id != run_id:
            self._native_run = self._build_native_run(
                experiment_id=self.experiment_id, run_id=run_id
            )
        else:
            self._native_run.is_active = True

    @staticmethod
    def _search_store(artifact_path: str, store: Dict[str, Any]) -> List[str]:
        artifact_path = Path(artifact_path).as_posix()
        return [key for key in store.keys() if key.startswith(artifact_path)]

    @staticmethod
    def _get_store_key(path: str) -> str:
        key = Path(path).as_posix()
        return key
