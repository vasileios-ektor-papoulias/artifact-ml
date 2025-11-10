from typing import Optional, Type, TypeVar

from clearml import Task

from artifact_experiment.base.tracking.backend.client import TrackingClient
from artifact_experiment.base.tracking.backend.logger_worker import LoggerWorker
from artifact_experiment.base.tracking.background.tracking_queue import TrackingQueue
from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.logger_worker import (
    ClearMLLoggerWorker,
)

ClearMLTrackingClientT = TypeVar("ClearMLTrackingClientT", bound="ClearMLTrackingClient")


class ClearMLTrackingClient(TrackingClient[ClearMLRunAdapter]):
    @classmethod
    def build(
        cls: Type[ClearMLTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> ClearMLTrackingClientT:
        run = ClearMLRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls._build(run=run)
        return client

    @classmethod
    def from_native_run(
        cls: Type[ClearMLTrackingClientT], native_run: Task
    ) -> ClearMLTrackingClientT:
        run = ClearMLRunAdapter.from_native_run(native_run=native_run)
        client = cls._build(run=run)
        return client

    @staticmethod
    def _get_logger_worker(
        run: ClearMLRunAdapter,
        tracking_queue: TrackingQueue,
    ) -> LoggerWorker[ClearMLRunAdapter]:
        return ClearMLLoggerWorker.build(
            run=run, queue=tracking_queue.queue, managed_temp_dir=tracking_queue.managed_temp_dir
        )
