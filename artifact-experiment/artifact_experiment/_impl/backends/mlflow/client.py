from typing import Optional, Type, TypeVar

from artifact_experiment._base.tracking.backend.client import TrackingClient
from artifact_experiment._base.tracking.backend.worker import BackendLoggingWorker
from artifact_experiment._base.tracking.background.queue import TrackingQueue
from artifact_experiment._impl.backends.mlflow.adapter import MlflowNativeRun, MlflowRunAdapter
from artifact_experiment._impl.backends.mlflow.worker import MlflowLoggingWorker

MlflowTrackingClientT = TypeVar("MlflowTrackingClientT", bound="MlflowTrackingClient")


class MlflowTrackingClient(TrackingClient[MlflowRunAdapter]):
    @classmethod
    def build(
        cls: Type[MlflowTrackingClientT], experiment_id: str, run_id: Optional[str] = None
    ) -> MlflowTrackingClientT:
        run = MlflowRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
        client = cls._build(run=run)
        return client

    @classmethod
    def from_native_run(
        cls: Type[MlflowTrackingClientT], native_run: MlflowNativeRun
    ) -> MlflowTrackingClientT:
        run = MlflowRunAdapter.from_native_run(native_run=native_run)
        client = cls._build(run=run)
        return client

    @staticmethod
    def _get_worker(
        run: MlflowRunAdapter, tracking_queue: TrackingQueue
    ) -> BackendLoggingWorker[MlflowRunAdapter]:
        return MlflowLoggingWorker.build(
            run=run, queue=tracking_queue.queue, temp_dir=tracking_queue.temp_dir
        )
