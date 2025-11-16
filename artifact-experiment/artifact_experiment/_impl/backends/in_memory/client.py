from typing import Optional, Type, TypeVar

from artifact_experiment._base.tracking.backend.client import TrackingClient
from artifact_experiment._base.tracking.backend.worker import BackendLoggingWorker
from artifact_experiment._base.tracking.background.queue import TrackingQueue
from artifact_experiment._impl.backends.in_memory.adapter import InMemoryRun, InMemoryRunAdapter
from artifact_experiment._impl.backends.in_memory.worker import InMemoryLoggingWorker

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
        return self._run.files

    @staticmethod
    def _get_worker(
        run: InMemoryRunAdapter,
        tracking_queue: TrackingQueue,
    ) -> BackendLoggingWorker[InMemoryRunAdapter]:
        return InMemoryLoggingWorker.build(
            queue=tracking_queue.queue, run=run, temp_dir=tracking_queue.temp_dir
        )
