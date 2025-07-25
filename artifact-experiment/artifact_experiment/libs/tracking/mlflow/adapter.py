import os
from dataclasses import dataclass
from typing import List, Optional

from mlflow.entities import Experiment, FileInfo, Metric, Run, RunStatus
from mlflow.tracking import MlflowClient

from artifact_experiment.base.tracking.adapter import InactiveRunError, RunAdapter


class InactiveMlflowRunError(InactiveRunError):
    pass


@dataclass
class MlflowNativeRun:
    client: MlflowClient
    run: Run


class MlflowRunAdapter(RunAdapter[MlflowNativeRun]):
    _root_dir = "artifact_ml"
    _default_tracking_uri = "http://localhost:5000"
    TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", _default_tracking_uri)

    @property
    def experiment_id(self) -> str:
        """User-chosen experiment ID (stored as name in MLflow)."""
        experiment = self._native_run.client.get_experiment(experiment_id=self.experiment_id)
        return experiment.name

    @property
    def experiment_uuid(self) -> str:
        """Internal MLflow experiment UUID."""
        return self._native_run.run.info.experiment_id

    @property
    def run_id(self) -> str:
        """User-chosen run ID (stored as run_name in MLflow)."""
        return str(self._native_run.run.info.run_name)

    @property
    def run_uuid(self) -> str:
        """Internal MLflow run UUID."""
        return self._native_run.run.info.run_id

    @property
    def run_status(self) -> str:
        return self._native_run.run.info.status

    @property
    def is_active(self) -> bool:
        return self.run_status.upper() == RunStatus.to_string(RunStatus.RUNNING)

    def stop(self):
        self._native_run.client.set_terminated(run_id=self.run_uuid)

    def upload(self, path_source: str, dir_target: str):
        if not self.is_active:
            raise InactiveMlflowRunError("Run is inactive")
        path_source = self._prepend_root_dir(path=path_source)
        self._native_run.client.log_artifact(
            run_id=self.run_uuid,
            local_path=path_source,
            artifact_path=dir_target,
        )

    def log_score(self, backend_path: str, value: float, step: int = 0):
        if not self.is_active:
            raise InactiveMlflowRunError("Run is inactive")
        backend_path = self._prepend_root_dir(path=backend_path)
        key = backend_path.replace("\\", "/")
        self._native_run.client.log_metric(run_id=self.run_uuid, key=key, value=value, step=step)

    def get_ls_artifact_info(self, backend_path: str) -> List[FileInfo]:
        backend_path = self._prepend_root_dir(path=backend_path)
        ls_artifact_infos = self._native_run.client.list_artifacts(
            run_id=self.run_uuid, path=backend_path
        )
        return ls_artifact_infos

    def get_ls_score_history(self, backend_path: str) -> List[Metric]:
        backend_path = self._prepend_root_dir(path=backend_path)
        key = backend_path.replace("\\", "/")
        ls_metric_history = self._native_run.client.get_metric_history(
            run_id=self.run_uuid, key=key
        )
        return ls_metric_history

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> MlflowNativeRun:
        native_client = MlflowClient(tracking_uri=cls.TRACKING_URI)
        experiment = cls._create_experiment(
            native_client=native_client, experiment_id=experiment_id
        )
        run = cls._create_run(native_client=native_client, experiment=experiment, run_id=run_id)
        native_run = MlflowNativeRun(client=native_client, run=run)
        return native_run

    @classmethod
    def _create_run(cls, native_client: MlflowClient, experiment: Experiment, run_id: str) -> Run:
        run = cls._get_run_from_id(
            native_client=native_client, experiment=experiment, run_id=run_id
        )
        if run is None:
            run = native_client.create_run(
                experiment_id=experiment.experiment_id,
                run_name=run_id,
            )
        if run.info.status != RunStatus.to_string(RunStatus.RUNNING):
            raise InactiveMlflowRunError(
                f"Inactive run with id {run_id} already exists with status {run.info.status}."
            )
        return run

    @classmethod
    def _create_experiment(cls, native_client: MlflowClient, experiment_id: str) -> Experiment:
        experiment = cls._get_experiment_from_id(
            native_client=native_client, experiment_id=experiment_id
        )
        if experiment is None:
            experiment_uuid = native_client.create_experiment(name=experiment_id)
            experiment = cls._get_experiment_from_uuid(
                native_client=native_client, experiment_uuid=experiment_uuid
            )
            assert experiment is not None, "Experiment creation failed"
        return experiment

    @staticmethod
    def _get_run_from_id(
        native_client: MlflowClient, experiment: Experiment, run_id: str
    ) -> Optional[Run]:
        ls_runs = native_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_id}'",
        )
        if ls_runs:
            run_info = ls_runs[0].info
            run = native_client.get_run(run_id=run_info.run_id)
            return run

    @staticmethod
    def _get_experiment_from_id(
        native_client: MlflowClient, experiment_id: str
    ) -> Optional[Experiment]:
        experiment = native_client.get_experiment_by_name(name=experiment_id)
        if experiment is not None:
            return experiment

    @staticmethod
    def _get_experiment_from_uuid(
        native_client: MlflowClient, experiment_uuid: str
    ) -> Optional[Experiment]:
        experiment = native_client.get_experiment(experiment_id=experiment_uuid)
        if experiment is not None:
            return experiment

    @classmethod
    def _prepend_root_dir(cls, path: str) -> str:
        return os.path.join(cls._root_dir, path)
