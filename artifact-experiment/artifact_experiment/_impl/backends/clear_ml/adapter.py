import os
import time
from typing import Optional

from artifact_core.typing import Plot
from clearml import Task, TaskTypes

from artifact_experiment._base.primitives.file import File
from artifact_experiment._base.tracking.backend.adapter import InactiveRunError, RunAdapter
from artifact_experiment._impl.backends.clear_ml.setup_validator import ClearMLSetupValidator
from artifact_experiment._impl.backends.clear_ml.stores.files import ClearMLFileStore
from artifact_experiment._impl.backends.clear_ml.stores.plots import ClearMLPlotStore
from artifact_experiment._impl.backends.clear_ml.stores.scores import ClearMLScoreStore


class InactiveClearMLRunError(InactiveRunError):
    pass


class ClearMLRunAdapter(RunAdapter[Task]):
    _root_dir = "artifact_ml"
    _time_to_wait_before_stopping_seconds: int = 1
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

    def stop(self):
        if self.is_active:
            time.sleep(self._time_to_wait_before_stopping_seconds)
            self._native_run.close()

    def log_score(self, value: float, title: str, series: str, iteration: int = 0):
        if not self.is_active:
            raise InactiveClearMLRunError()
        title = self._prepend_root_dir(path=title)
        logger = self._native_run.get_logger()
        logger.report_scalar(
            title=title,
            series=series,
            iteration=iteration,
            value=value,
        )

    def log_plot(self, plot: Plot, title: str, series: str, iteration: int = 0):
        if not self.is_active:
            raise InactiveClearMLRunError()
        title = self._prepend_root_dir(path=title)
        logger = self._native_run.get_logger()
        logger.report_matplotlib_figure(
            title=title,
            series=series,
            iteration=iteration,
            figure=plot,
        )

    def log_file(self, backend_dir: str, file: File, delete_after_upload: bool = False):
        if not self.is_active:
            raise InactiveClearMLRunError()
        backend_dir = self._prepend_root_dir(path=backend_dir)
        self._native_run.upload_artifact(
            name=backend_dir,
            artifact_object=file.path_source,
            delete_after_upload=delete_after_upload,
            auto_pickle=False,
            wait_on_upload=True,
        )

    def get_exported_scores(self, max_iterations: int = 0) -> ClearMLScoreStore:
        raw_store_data = self._native_run.get_reported_scalars(max_samples=max_iterations)
        store = ClearMLScoreStore.build(raw_store_data=raw_store_data, root_dir=self._root_dir)
        return store

    def get_exported_plots(self, max_iterations: int = 0) -> ClearMLPlotStore:
        raw_plot_data = self._native_run.get_reported_plots(max_iterations=max_iterations)
        store = ClearMLPlotStore.build(raw_plot_data=raw_plot_data, root_dir=self._root_dir)
        return store

    def get_exported_files(self) -> ClearMLFileStore:
        dict_all_files = self._native_run.artifacts
        store = ClearMLFileStore.build(dict_all_files=dict_all_files, root_dir=self._root_dir)
        return store

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> Task:
        if not ClearMLSetupValidator.is_configured():
            setup_instructions = ClearMLSetupValidator.get_setup_instructions()
            raise RuntimeError(setup_instructions)
        native_run = Task.init(
            project_name=experiment_id,
            task_name=run_id,
            task_type=cls._new_task_type,
            reuse_last_task_id=False,
        )
        return native_run

    @classmethod
    def _prepend_root_dir(cls, path: str) -> str:
        return os.path.join(cls._root_dir, path)
