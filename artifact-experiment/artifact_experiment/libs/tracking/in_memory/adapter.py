from typing import Dict, List

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.tracking.adapter import RunAdapter
from artifact_experiment.libs.tracking.in_memory.native_run import (
    InMemoryRun,
)


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
    def dict_arrays(self) -> Dict[str, ndarray]:
        return self._native_run.dict_arrays

    @property
    def ls_arrays(self) -> List[ndarray]:
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
    def dict_array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return self._native_run.dict_array_collections

    @property
    def ls_array_collections(self) -> List[Dict[str, ndarray]]:
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

    def upload(self, path_source: str, dir_target: str):
        upload_entry = self._format_upload_entry(path_source=path_source, dir_target=dir_target)
        self._native_run.uploaded_files.append(upload_entry)

    @staticmethod
    def _format_upload_entry(path_source: str, dir_target: str) -> Dict[str, str]:
        dict_upload_entry = {"path_source": path_source, "dir_target": dir_target}
        return dict_upload_entry

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
