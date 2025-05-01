import os
from dataclasses import dataclass
from typing import Optional

from mlflow.entities import Run, RunStatus
from mlflow.tracking import MlflowClient

from artifact_experiment.base.tracking.backend import TrackingBackend


class InactiveMlflowRunError(Exception):
    pass


@dataclass
class MlflowNativeClient:
    client: MlflowClient
    run: Run


class MlflowBackend(TrackingBackend[MlflowNativeClient]):
    _tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    ROOT_DIR = "artifact_ml"

    @property
    def experiment_id(self) -> str:
        return self._native_client.run.info.experiment_id

    @property
    def experiment_name(self) -> str:
        experiment = self._native_client.client.get_experiment(experiment_id=self.experiment_id)
        return experiment.name

    @property
    def run_uuid(self) -> str:
        return self._native_client.run.info.run_id

    @property
    def run_id(self) -> str:
        name = self._native_client.run.info.run_name
        return name if name is not None else self.run_id

    @property
    def run_status(self) -> str:
        return self._native_client.run.info.status

    @property
    def run_is_active(self) -> bool:
        return self.run_status.upper() == RunStatus.to_string(RunStatus.RUNNING)

    @classmethod
    def _get_native_client(cls, experiment_id: str, run_id: str) -> MlflowNativeClient:
        mlflow_client = MlflowClient(tracking_uri=cls._tracking_uri)
        run = cls._start_mlflow_run(
            mlflow_client=mlflow_client, experiment_id=experiment_id, run_id=run_id
        )
        native_client = MlflowNativeClient(client=mlflow_client, run=run)
        return native_client

    def _start_run(self, run_id: str):
        self._native_client = self._get_native_client(
            experiment_id=self.experiment_id, run_id=run_id
        )

    def _stop_run(self):
        self._native_client.client.set_terminated(run_id=self.run_uuid)

    @classmethod
    def _start_mlflow_run(cls, mlflow_client: MlflowClient, experiment_id: str, run_id: str) -> Run:
        run = cls._get_run_from_id(
            mlflow_client=mlflow_client, experiment_id=experiment_id, run_id=run_id
        )
        if run is None:
            run = mlflow_client.create_run(
                experiment_id=experiment_id,
                run_name=run_id,
            )
        if run.info.status != RunStatus.to_string(RunStatus.RUNNING):
            raise InactiveMlflowRunError(
                f"Inactive run with id {run_id} already exists with status {run.info.status}."
            )
        return run

    @staticmethod
    def _get_run_from_id(
        mlflow_client: MlflowClient, experiment_id: str, run_id: str
    ) -> Optional[Run]:
        ls_runs = mlflow_client.search_runs(
            experiment_ids=[experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'"
        )
        if ls_runs:
            run_info = ls_runs[0].info
            run = mlflow_client.get_run(run_id=run_info.run_id)
            return run
