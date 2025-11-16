import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import neptune
from artifact_core.typing import ArtifactResult

from artifact_experiment._base.primitives.file import File
from artifact_experiment._base.tracking.backend.adapter import InactiveRunError, RunAdapter
from artifact_experiment._utils.collections.map_navigator import MapNavigator
from artifact_experiment._utils.system.env_var_reader import EnvVarReader

NeptuneLogType = Union[ArtifactResult, str, Dict[str, str]]


class NeptuneRunStatus(Enum):
    RUNNING = "running"
    INACTIVE = "inactive"


class InactiveNeptuneRunError(InactiveRunError):
    pass


class NeptuneRunAdapter(RunAdapter[neptune.Run]):
    _root_dir = "artifact_ml"
    _time_to_wait_before_stopping_seconds: int = 1
    _active_run_status = NeptuneRunStatus.RUNNING
    _api_token_env_var_name = "NEPTUNE_API_TOKEN"
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

    def log(self, artifact_path: str, artifact: NeptuneLogType):
        if not self.is_active:
            raise InactiveNeptuneRunError()
        artifact_path = self._prepend_root_dir(path=artifact_path)
        key = self._get_store_key(path=artifact_path)
        self._native_run[key].append(artifact)

    def log_file(self, backend_dir: str, file: File):
        if not self.is_active:
            raise InactiveNeptuneRunError()
        backend_dir = self._prepend_root_dir(path=backend_dir)
        key = self._get_store_key(path=backend_dir)
        self._native_run[key].upload(file.path_source, wait=True)

    def get_namespace_data(self, backend_path: str) -> Dict[str, Any]:
        backend_path = self._prepend_root_dir(path=backend_path)
        key = self._get_store_key(path=backend_path)
        run_metadata = self._native_run.fetch()
        namespace_data = MapNavigator.get(data=run_metadata, path=key, default={})
        return namespace_data

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
            cls._api_token = EnvVarReader.get(env_var_name=cls._api_token_env_var_name)
        return cls._api_token

    @classmethod
    def _prepend_root_dir(cls, path: str) -> str:
        return os.path.join(cls._root_dir, path.lstrip("/"))

    @staticmethod
    def _get_store_key(path: str) -> str:
        key = Path(path).as_posix()
        return key
