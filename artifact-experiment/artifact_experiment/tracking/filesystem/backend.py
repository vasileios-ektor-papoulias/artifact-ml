import os
from pathlib import Path
from typing import TypeVar

from artifact_experiment.base.tracking.backend import (
    TrackingBackend,
)

filesystemBackendT = TypeVar("filesystemBackendT", bound="FilesystemBackend")


class FilesystemExperimentNotSetError(Exception):
    pass


class FilesystemExperiment:
    _default_root_dir = Path.home() / "artifact-ml"

    def __init__(self, experiment_id: str):
        self.start(experiment_id=experiment_id)

    @property
    def experiment_is_active(self) -> bool:
        return self._experiment_is_active

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def experiment_dir(self) -> str:
        return os.path.join(str(self._default_root_dir), self._experiment_id)

    def start(self, experiment_id: str):
        self._experiment_is_active = True
        self._experiment_id = experiment_id
        self._create_experiment_dir()

    def stop(self):
        self._experiment_is_active = False

    def _create_experiment_dir(self):
        os.makedirs(name=self.experiment_dir, exist_ok=True)


class FilesystemBackend(TrackingBackend[FilesystemExperiment]):
    @property
    def experiment_is_active(self) -> bool:
        return self._native_client.experiment_is_active

    @property
    def experiment_id(self) -> str:
        return self._native_client.experiment_id

    def _start_experiment(self, experiment_id: str):
        self._native_client = self._get_native_client(experiment_id=experiment_id)

    @classmethod
    def _stop_experiment(cls, native_client: FilesystemExperiment):
        native_client.stop()
