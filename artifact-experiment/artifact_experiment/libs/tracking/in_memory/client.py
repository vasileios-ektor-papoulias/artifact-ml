from typing import Optional, Type, TypeVar

from artifact_experiment.base.tracking.backend.client import TrackingClient
from artifact_experiment.base.tracking.backend.logger_worker import LoggerWorker
from artifact_experiment.base.tracking.background.tracking_queue import TrackingQueue
from artifact_experiment.libs.tracking.in_memory.adapter import InMemoryRun, InMemoryRunAdapter
from artifact_experiment.libs.tracking.in_memory.logger_worker import InMemoryLoggerWorker

InMemoryTrackingClientT = TypeVar("InMemoryTrackingClientT", bound="InMemoryTrackingClient")


class InMemoryTrackingClient(TrackingClient[InMemoryRunAdapter]):
    @classmethod
    def build(
        cls: Type[InMemoryTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> InMemoryTrackingClientT:
        run = InMemoryRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls._build(run=run)
        return client

    @classmethod
    def from_native_run(
        cls: Type[InMemoryTrackingClientT], native_run: InMemoryRun
    ) -> InMemoryTrackingClientT:
        run = InMemoryRunAdapter.from_native_run(native_run=native_run)
        client = cls._build(run=run)
        return client

    @property
    def uploaded_files(self):
        return self._run.uploaded_files

    @staticmethod
    def _get_logger_worker(
        run: InMemoryRunAdapter,
        tracking_queue: TrackingQueue,
    ) -> LoggerWorker[InMemoryRunAdapter]:
        return InMemoryLoggerWorker.build(
            queue=tracking_queue.queue, run=run, managed_temp_dir=tracking_queue.managed_temp_dir
        )
