from typing import Optional, Type, TypeVar

from artifact_experiment.base.tracking.backend.client import TrackingClient
from artifact_experiment.base.tracking.backend.logger_worker import LoggerWorker
from artifact_experiment.base.tracking.background.tracking_queue import TrackingQueue
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRun, FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.logger_worker import FilesystemLoggerWorker

FilesystemTrackingClientT = TypeVar("FilesystemTrackingClientT", bound="FilesystemTrackingClient")


class FilesystemTrackingClient(TrackingClient[FilesystemRunAdapter]):
    @classmethod
    def build(
        cls: Type[FilesystemTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> FilesystemTrackingClientT:
        run = FilesystemRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls._build(run=run)
        return client

    @classmethod
    def from_native_run(
        cls: Type[FilesystemTrackingClientT], native_run: FilesystemRun
    ) -> FilesystemTrackingClientT:
        run = FilesystemRunAdapter.from_native_run(native_run=native_run)
        client = cls._build(run=run)
        return client

    @property
    def experiment_dir(self) -> str:
        return self._run.experiment_dir

    @property
    def run_dir(self) -> str:
        return self._run.run_dir

    @staticmethod
    def _get_logger_worker(
        run: FilesystemRunAdapter,
        tracking_queue: TrackingQueue,
    ) -> LoggerWorker[FilesystemRunAdapter]:
        return FilesystemLoggerWorker.build(
            run=run, queue=tracking_queue.queue, managed_temp_dir=tracking_queue.managed_temp_dir
        )
