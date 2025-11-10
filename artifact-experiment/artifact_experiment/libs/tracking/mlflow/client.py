from typing import Optional, Type, TypeVar

from artifact_experiment.base.tracking.backend.client import TrackingClient
from artifact_experiment.base.tracking.backend.logger_worker import LoggerWorker
from artifact_experiment.base.tracking.background.tracking_queue import TrackingQueue
from artifact_experiment.libs.tracking.mlflow.adapter import MlflowNativeRun, MlflowRunAdapter
from artifact_experiment.libs.tracking.mlflow.logger_worker import MlflowLoggerWorker

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
    def _get_logger_worker(
        run: MlflowRunAdapter, tracking_queue: TrackingQueue
    ) -> LoggerWorker[MlflowRunAdapter]:
        return MlflowLoggerWorker.build(
            run=run, queue=tracking_queue.queue, managed_temp_dir=tracking_queue.managed_temp_dir
        )
