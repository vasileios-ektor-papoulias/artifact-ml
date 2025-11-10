from typing import Optional, Type, TypeVar

import neptune

from artifact_experiment.base.tracking.backend.client import TrackingClient
from artifact_experiment.base.tracking.backend.logger_worker import LoggerWorker
from artifact_experiment.base.tracking.background.tracking_queue import TrackingQueue
from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
from artifact_experiment.libs.tracking.neptune.logger_worker import NeptuneLoggerWorker

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
    def _get_logger_worker(
        run: NeptuneRunAdapter, tracking_queue: TrackingQueue
    ) -> LoggerWorker[NeptuneRunAdapter]:
        return NeptuneLoggerWorker.build(
            run=run, queue=tracking_queue.queue, managed_temp_dir=tracking_queue.managed_temp_dir
        )
