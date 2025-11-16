from typing import Optional, Type, TypeVar

from clearml import Task

from artifact_experiment._base.tracking.backend.client import TrackingClient
from artifact_experiment._base.tracking.backend.worker import BackendLoggingWorker
from artifact_experiment._base.tracking.background.queue import TrackingQueue
from artifact_experiment._impl.backends.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment._impl.backends.clear_ml.worker import ClearMLLoggingWorker

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
    def _get_worker(
        run: ClearMLRunAdapter, tracking_queue: TrackingQueue
    ) -> BackendLoggingWorker[ClearMLRunAdapter]:
        return ClearMLLoggingWorker.build(
            run=run, queue=tracking_queue.queue, temp_dir=tracking_queue.temp_dir
        )
