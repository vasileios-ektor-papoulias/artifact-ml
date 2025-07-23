import os
import time
from enum import Enum
from getpass import getpass
from typing import Any, Dict, Optional

import neptune
from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.adapter import InactiveRunError, RunAdapter


class NeptuneRunStatus(Enum):
    RUNNING = "running"
    INACTIVE = "inactive"


class InactiveNeptuneRunError(InactiveRunError):
    pass


class NeptuneRunAdapter(RunAdapter[neptune.Run]):
    _root_dir = "artifact_ml"
    _time_to_wait_before_stopping_seconds: int = 1
    _active_run_status = NeptuneRunStatus.RUNNING
    _api_token: Optional[str] = None

    def __init__(self, native_run: neptune.Run):
        super().__init__(native_run=native_run)
        self._experiment_id: Optional[str] = None
        self._run_id: Optional[str] = None

    @property
    def experiment_id(self) -> str:
        if self._experiment_id is None:
            self._experiment_id = self._native_run["sys/experiment/name"].fetch()
        assert self._experiment_id is not None
        return self._experiment_id

    @property
    def run_id(self) -> str:
        if self._run_id is None:
            self._run_id = self._native_run["sys/id"].fetch()
        assert self._run_id is not None
        return self._run_id

    @property
    def run_status(self) -> NeptuneRunStatus:
        try:
            return NeptuneRunStatus(self.run_metadata["sys"]["state"])
        except ValueError:
            raise RuntimeError(f"Unknown Neptune run state: {self.run_metadata['sys']['state']}")

    @property
    def is_active(self) -> bool:
        return self.run_status == self._active_run_status

    @property
    def run_metadata(self) -> Dict[str, Any]:
        run_metadata = self._native_run.fetch()
        return run_metadata

    def stop(self):
        if self.is_active:
            time.sleep(self._time_to_wait_before_stopping_seconds)
            self._native_run.stop()

    def upload(self, path_source: str, dir_target: str):
        if not self.is_active:
            raise InactiveNeptuneRunError("Run is inactive")
        dir_target = self._prepend_root_dir(path=dir_target).replace("\\", "/")
        self._native_run[dir_target].upload(path_source)

    def log(self, artifact_path: str, artifact: ArtifactResult):
        if not self.is_active:
            raise InactiveNeptuneRunError("Run is inactive")
        artifact_path = self._prepend_root_dir(path=artifact_path).replace("\\", "/")
        self._native_run[artifact_path].append(artifact)

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> neptune.Run:
        api_token = cls._get_api_token()
        native_run = neptune.init_run(
            api_token=api_token,
            project=experiment_id,
            custom_run_id=run_id,
        )
        return native_run

    @classmethod
    def _get_api_token(cls) -> str:
        if cls._api_token is None:
            cls._api_token = os.getenv(
                "NEPTUNE_API_TOKEN", default=getpass("Enter your Neptune API token: ")
            )
        return cls._api_token

    @classmethod
    def _prepend_root_dir(cls, path: str) -> str:
        return os.path.join(cls._root_dir, path)
