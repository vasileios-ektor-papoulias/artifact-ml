import time
from typing import Optional, Type, TypeVar

import neptune

from artifact_experiment.base.tracking.backend import TrackingBackend

neptuneBackendT = TypeVar("neptuneBackendT", bound="NeptuneBackend")


class NeptuneExperimentNotSetError(Exception):
    pass


class NeptuneBackend(TrackingBackend[neptune.Run]):
    ROOT_DIR = "artifact-ml"
    _time_to_wait_before_stopping_seconds = 1
    _neptune_api_token = ""
    _neptune_project_name = ""
    _neptune_project_key = ""

    @classmethod
    def build(cls: Type[neptuneBackendT], experiment_id: Optional[str] = None) -> neptuneBackendT:
        if experiment_id is None:
            experiment_id = cls._generate_random_id()
        native_client = cls._get_native_client(experiment_id=experiment_id)
        backend = cls(native_client=native_client)
        return backend

    @classmethod
    def from_native_client(
        cls: Type[neptuneBackendT], native_client: neptune.Run
    ) -> neptuneBackendT:
        backend = cls(native_client=native_client)
        return backend

    @property
    def experiment_is_active(self) -> bool:
        state = self._native_client["sys/state"].fetch()
        return state == "active"

    @property
    def experiment_id(self) -> str:
        return self._native_client["sys/id"].fetch()

    @property
    def native_client(self) -> neptune.Run:
        return self._native_client

    def _start_experiment(self, experiment_id: str):
        self._native_client = self._get_native_client(experiment_id=experiment_id)

    @classmethod
    def _stop_experiment(cls, native_client: neptune.Run):
        time.sleep(cls._time_to_wait_before_stopping_seconds)
        native_client.stop()

    @classmethod
    def _get_native_client(cls, experiment_id: str) -> neptune.Run:
        native_client = neptune.init_run(
            project=cls._neptune_project_name,
            api_token=cls._neptune_api_token,
            custom_run_id=experiment_id,
        )
        return native_client
