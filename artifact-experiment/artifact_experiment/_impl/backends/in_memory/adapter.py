from pathlib import Path
from typing import Any, List, Mapping, Sequence

from artifact_core.typing import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.primitives.file import File
from artifact_experiment._base.tracking.backend.adapter import RunAdapter
from artifact_experiment._impl.backends.in_memory.native_run import InMemoryRun


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
    def scores(self) -> Mapping[str, Score]:
        return self._native_run.scores

    @property
    def ls_scores(self) -> Sequence[Score]:
        return list(self.scores.values())

    @property
    def n_scores(self) -> int:
        return len(self.ls_scores)

    @property
    def arrays(self) -> Mapping[str, Array]:
        return self._native_run.arrays

    @property
    def ls_arrays(self) -> List[Array]:
        return list(self.arrays.values())

    @property
    def n_arrays(self) -> int:
        return len(self.ls_arrays)

    @property
    def plots(self) -> Mapping[str, Plot]:
        return self._native_run.plots

    @property
    def ls_plots(self) -> Sequence[Plot]:
        return list(self.plots.values())

    @property
    def n_plots(self) -> int:
        return len(self.ls_plots)

    @property
    def score_collections(self) -> Mapping[str, ScoreCollection]:
        return self._native_run.score_collections

    @property
    def ls_score_collections(self) -> Sequence[ScoreCollection]:
        return list(self.score_collections.values())

    @property
    def n_score_collections(self) -> int:
        return len(self.ls_score_collections)

    @property
    def array_collections(self) -> Mapping[str, ArrayCollection]:
        return self._native_run.array_collections

    @property
    def ls_array_collections(self) -> Sequence[ArrayCollection]:
        return list(self.array_collections.values())

    @property
    def n_array_collections(self) -> int:
        return len(self.ls_array_collections)

    @property
    def plot_collections(self) -> Mapping[str, PlotCollection]:
        return self._native_run.plot_collections

    @property
    def ls_plot_collections(self) -> Sequence[PlotCollection]:
        return list(self.plot_collections.values())

    @property
    def n_plot_collections(self) -> int:
        return len(self.ls_plot_collections)

    @property
    def files(self) -> Mapping[str, File]:
        return self._native_run.files

    @property
    def ls_files(self) -> Sequence[File]:
        return list(self.files.values())

    @property
    def n_files(self) -> int:
        return len(self.ls_files)

    def stop(self):
        self._native_run.is_active = False

    def log_score(self, path: str, score: Score):
        key = self._get_store_key(path=path)
        self._native_run.log_score(key=key, score=score)

    def log_array(self, path: str, array: Array):
        key = self._get_store_key(path=path)
        self._native_run.log_array(key=key, array=array)

    def log_plot(self, path: str, plot: Plot):
        key = self._get_store_key(path=path)
        self._native_run.log_plot(key=key, plot=plot)

    def log_score_collection(self, path: str, score_collection: ScoreCollection):
        key = self._get_store_key(path=path)
        self._native_run.log_score_collection(key=key, score_collection=score_collection)

    def log_array_collection(self, path: str, array_collection: ArrayCollection):
        key = self._get_store_key(path=path)
        self._native_run.log_array_collection(key=key, array_collection=array_collection)

    def log_plot_collection(self, path: str, plot_collection: PlotCollection):
        key = self._get_store_key(path=path)
        self._native_run.log_plot_collection(key=key, plot_collection=plot_collection)

    def log_file(self, key: str, file: File):
        self._native_run.log_file(key=key, file=file)

    def search_score_store(self, store_path: str) -> Sequence[str]:
        return self._search_store(store_path=store_path, store=self.scores)

    def search_array_store(self, store_path: str) -> Sequence[str]:
        return self._search_store(store_path=store_path, store=self.arrays)

    def search_plot_store(self, store_path: str) -> Sequence[str]:
        return self._search_store(store_path=store_path, store=self.plots)

    def search_score_collection_store(self, store_path: str) -> Sequence[str]:
        return self._search_store(store_path=store_path, store=self.score_collections)

    def search_array_collection_store(self, store_path: str) -> Sequence[str]:
        return self._search_store(store_path=store_path, store=self.array_collections)

    def search_plot_collection_store(self, store_path: str) -> Sequence[str]:
        return self._search_store(store_path=store_path, store=self.plot_collections)

    def search_file_store(self, store_path: str) -> Sequence[str]:
        return self._search_store(store_path=store_path, store=self.files)

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
    def _search_store(store_path: str, store: Mapping[str, Any]) -> Sequence[str]:
        store_path = Path(store_path).as_posix()
        return [key for key in store.keys() if key.startswith(store_path)]

    @staticmethod
    def _get_store_key(path: str) -> str:
        key = Path(path).as_posix()
        return key
