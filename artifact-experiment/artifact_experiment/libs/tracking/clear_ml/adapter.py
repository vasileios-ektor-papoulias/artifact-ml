import time
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence

from clearml import Task, TaskTypes
from clearml.binding.artifacts import Artifact
from matplotlib.figure import Figure

from artifact_experiment.base.tracking.adapter import InactiveRunError, RunAdapter


class InactiveClearMLRunError(InactiveRunError):
    pass


class ClearMLRunAdapter(RunAdapter[Task]):
    _time_to_wait_before_stopping_seconds: int = 1
    _default_series_name: str = ""
    _tup_active_statuses = (
        Task.TaskStatusEnum.queued,
        Task.TaskStatusEnum.in_progress,
    )
    _new_task_type = TaskTypes.testing

    def __init__(self, native_run: Task):
        super().__init__(native_run=native_run)
        self._experiment_id: Optional[str] = None
        self._run_id: Optional[str] = None

    @property
    def experiment_id(self) -> str:
        if self._experiment_id is None:
            self._experiment_id = self._native_run.project
        return self._experiment_id

    @property
    def run_id(self) -> str:
        if self._run_id is None:
            self._run_id = self._native_run.id
        assert self._run_id is not None
        return self._run_id

    @property
    def run_status(self) -> str:
        status = self._native_run.get_status()
        return status

    @property
    def is_active(self) -> bool:
        return self.run_status.lower() in (status.value for status in self._tup_active_statuses)

    def start(self):
        if not self.is_active:
            self._native_run = self._build_native_run(
                experiment_id=self.experiment_id,
                run_id=self.run_id,
            )

    def stop(self):
        if self.is_active:
            time.sleep(self._time_to_wait_before_stopping_seconds)
            self._native_run.close()

    def log_score(
        self,
        value: float,
        title: str,
        series: Optional[str] = None,
        iteration: int = 0,
    ):
        if not self.is_active:
            raise InactiveClearMLRunError("Run is inactive")
        if series is None:
            series = deepcopy(self._default_series_name)
        logger = self._native_run.get_logger()
        logger.report_scalar(
            title=title,
            series=series,
            iteration=iteration,
            value=value,
        )

    def log_plot(
        self,
        plot: Figure,
        title: str,
        series: Optional[str] = None,
        iteration: int = 0,
    ):
        if not self.is_active:
            raise InactiveClearMLRunError("Run is inactive")
        if series is None:
            series = deepcopy(self._default_series_name)
        logger = self._native_run.get_logger()
        logger.report_matplotlib_figure(
            title=title,
            series=series,
            iteration=iteration,
            figure=plot,
        )

    def upload_artifact(self, name: str, filepath: str, delete_after_upload: bool = False):
        if not self.is_active:
            raise InactiveClearMLRunError("Run is inactive")
        self._native_run.upload_artifact(
            name=name,
            artifact_object=filepath,
            delete_after_upload=delete_after_upload,
            auto_pickle=False,
            wait_on_upload=True,
        )

    def get_exported_scores(
        self, max_iterations: int = 0
    ) -> Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]]:
        score_data = self._native_run.get_reported_scalars(max_samples=max_iterations)
        return score_data

    def get_exported_plots(self, max_iterations: int = 0) -> List[Dict[str, Any]]:
        plot_data = self._native_run.get_reported_plots(max_iterations=max_iterations)
        return plot_data

    def get_uploaded_files(self) -> Dict[str, Artifact]:
        return self._native_run.artifacts

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> Task:
        return Task.init(
            project_name=experiment_id,
            task_name=run_id,
            task_type=cls._new_task_type,
            reuse_last_task_id=False,
        )
