from typing import Optional, Type, TypeVar

import neptune

from artifact_experiment._base.tracking.backend.client import TrackingClient
from artifact_experiment._base.tracking.backend.worker import BackendLoggingWorker
from artifact_experiment._base.tracking.background.queue import TrackingQueue
from artifact_experiment._impl.backends.neptune.adapter import NeptuneRunAdapter
from artifact_experiment._impl.backends.neptune.worker import NeptuneLoggingWorker

NeptuneTrackingClientT = TypeVar("NeptuneTrackingClientT", bound="NeptuneTrackingClient")


class NeptuneTrackingClient(TrackingClient[NeptuneRunAdapter]):
    @classmethod
    def build(
        cls: Type[NeptuneTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> NeptuneTrackingClientT:
        run = NeptuneRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls._build(run=run)
        return client

    @classmethod
    def from_native_run(
        cls: Type[NeptuneTrackingClientT], native_run: neptune.Run
    ) -> NeptuneTrackingClientT:
        run = NeptuneRunAdapter.from_native_run(native_run=native_run)
        client = cls._build(run=run)
        return client

    @staticmethod
    def _get_worker(
        run: NeptuneRunAdapter, tracking_queue: TrackingQueue
    ) -> BackendLoggingWorker[NeptuneRunAdapter]:
        return NeptuneLoggingWorker.build(
            run=run, queue=tracking_queue.queue, temp_dir=tracking_queue.temp_dir
        )
