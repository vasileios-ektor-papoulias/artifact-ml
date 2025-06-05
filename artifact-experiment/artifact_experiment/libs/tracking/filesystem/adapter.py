import os
import shutil
from pathlib import Path

from artifact_experiment.base.tracking.adapter import InactiveRunError, RunAdapter


class InactiveFilesystemRunError(InactiveRunError):
    pass


class FilesystemRun:
    _default_root_dir = Path.home() / "artifact_ml"

    def __init__(self, experiment_id: str, run_id: str):
        self._experiment_id = experiment_id
        self._start(id=run_id)

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def experiment_dir(self) -> str:
        return os.path.join(str(self._default_root_dir), self._experiment_id)

    @property
    def run_dir(self) -> str:
        return os.path.join(self.experiment_dir, self._id)

    def stop(self):
        self._is_active = False

    def _start(self, id: str):
        self._is_active = True
        self._id = id
        self._create_run_dir()

    def _create_run_dir(self):
        os.makedirs(name=self.run_dir, exist_ok=True)


class FilesystemRunAdapter(RunAdapter[FilesystemRun]):
    @property
    def experiment_id(self) -> str:
        return self._native_run.experiment_id

    @property
    def run_id(self) -> str:
        return self._native_run.id

    @property
    def is_active(self) -> bool:
        return self._native_run.is_active

    @property
    def experiment_dir(self) -> str:
        return self._native_run.experiment_dir

    @property
    def run_dir(self) -> str:
        return self._native_run.run_dir

    def stop(self):
        self._native_run.stop()

    def upload(self, path_source: str, dir_target: str):
        if not self.is_active:
            raise InactiveFilesystemRunError("Cannot upload: run is not active.")
        dir_target_abs = os.path.join(self.run_dir, dir_target)
        os.makedirs(dir_target_abs, exist_ok=True)
        shutil.copy2(path_source, dir_target_abs)

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> FilesystemRun:
        return FilesystemRun(experiment_id=experiment_id, run_id=run_id)
