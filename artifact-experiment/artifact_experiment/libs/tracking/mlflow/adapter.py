import os
from dataclasses import dataclass
from typing import List, Optional

from mlflow.entities import FileInfo, Metric, Run, RunStatus
from mlflow.tracking import MlflowClient

from artifact_experiment.base.tracking.adapter import RunAdapter


class InactiveMlflowRunError(Exception):
    pass


@dataclass
class MlflowNativeClient:
    client: MlflowClient
    run: Run


class MlflowRunAdapter(RunAdapter[MlflowNativeClient]):
    _tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

    @property
    def experiment_id(self) -> str:
        return self._native_run.run.info.experiment_id

    @property
    def experiment_name(self) -> str:
        experiment = self._native_run.client.get_experiment(experiment_id=self.experiment_id)
        return experiment.name

    @property
    def run_uuid(self) -> str:
        return self._native_run.run.info.run_id

    @property
    def id(self) -> str:
        name = self._native_run.run.info.run_name
        return name if name is not None else self.id

    @property
    def run_status(self) -> str:
        return self._native_run.run.info.status

    @property
    def is_active(self) -> bool:
        return self.run_status.upper() == RunStatus.to_string(RunStatus.RUNNING)

    def log_metric(self, backend_path: str, value: float, step: int = 0):
        if self.is_active:
            self._native_run.client.log_metric(
                run_id=self.id, key=backend_path, value=value, step=step
            )
        else:
            raise InactiveMlflowRunError("Run is inactive")

    def upload(self, backend_path: str, local_path: str):
        if self.is_active:
            self._native_run.client.log_artifact(
                run_id=self.id,
                local_path=local_path,
                artifact_path=backend_path,
            )
        else:
            raise InactiveMlflowRunError("Run is inactive")

    def get_ls_artifact_info(self, backend_path: str) -> List[FileInfo]:
        ls_artifact_infos = self._native_run.client.list_artifacts(
            run_id=self.id, path=backend_path
        )
        return ls_artifact_infos

    def get_ls_metric_history(self, backend_path: str) -> List[Metric]:
        ls_metric_history = self._native_run.client.get_metric_history(
            run_id=self.id, key=backend_path
        )
        return ls_metric_history

    def start(self):
        self._native_run = self._build_native_run(experiment_id=self.experiment_id, run_id=self.id)

    def stop(self):
        self._native_run.client.set_terminated(run_id=self.run_uuid)

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> MlflowNativeClient:
        mlflow_client = MlflowClient(tracking_uri=cls._tracking_uri)
        run = cls._start_mlflow_run(
            mlflow_client=mlflow_client, experiment_id=experiment_id, run_id=run_id
        )
        native_run = MlflowNativeClient(client=mlflow_client, run=run)
        return native_run

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
