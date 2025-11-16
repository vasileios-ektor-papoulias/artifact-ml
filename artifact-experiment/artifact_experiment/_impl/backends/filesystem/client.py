from typing import Optional, Type, TypeVar

from artifact_experiment._base.tracking.backend.client import TrackingClient
from artifact_experiment._base.tracking.backend.worker import BackendLoggingWorker
from artifact_experiment._base.tracking.background.queue import TrackingQueue
from artifact_experiment._impl.backends.filesystem.adapter import (
    FilesystemRun,
    FilesystemRunAdapter,
)
from artifact_experiment._impl.backends.filesystem.worker import FilesystemLoggingWorker

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
    def _get_worker(
        run: FilesystemRunAdapter,
        tracking_queue: TrackingQueue,
    ) -> BackendLoggingWorker[FilesystemRunAdapter]:
        return FilesystemLoggingWorker.build(
            run=run, queue=tracking_queue.queue, temp_dir=tracking_queue.temp_dir
        )
