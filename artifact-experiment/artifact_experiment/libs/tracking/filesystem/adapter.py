import os
from pathlib import Path

from artifact_experiment.base.tracking.adapter import (
    RunAdapter,
)


class NoActiveFilesystemRunError(Exception):
    pass


class FilesystemRun:
    _default_root_dir = Path.home() / "artifact_ml"

    def __init__(self, experiment_id: str, run_id: str):
        self._experiment_id = experiment_id
        self.start(run_id=run_id)

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_is_active(self) -> bool:
        return self._run_is_active

    @property
    def experiment_dir(self) -> str:
        return os.path.join(str(self._default_root_dir), self._experiment_id)

    @property
    def run_dir(self) -> str:
        return os.path.join(self.experiment_dir, self._run_id)

    def start(self, run_id: str):
        self._run_is_active = True
        self._run_id = run_id
        self._create_run_dir()

    def stop(self):
        self._experiment_is_active = False

    def _create_run_dir(self):
        os.makedirs(name=self.run_dir, exist_ok=True)


class FilesystemRunAdapter(RunAdapter[FilesystemRun]):
    @property
    def experiment_id(self) -> str:
        return self._native_run.experiment_id

    @property
    def run_id(self) -> str:
        return self._native_run.run_id

    @property
    def run_is_active(self) -> bool:
        return self._native_run.run_is_active

    @property
    def experiment_dir(self) -> str:
        return self._native_run.experiment_dir

    @property
    def run_dir(self) -> str:
        return self._native_run.run_dir

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> FilesystemRun:
        return FilesystemRun(experiment_id=experiment_id, run_id=run_id)

    def _start(self, run_id: str):
        self._native_run = self._build_native_run(experiment_id=self.experiment_id, run_id=run_id)

    def stop(self):
        self._native_run.stop()
