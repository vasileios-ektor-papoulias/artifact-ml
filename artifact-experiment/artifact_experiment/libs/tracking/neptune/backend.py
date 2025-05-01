import time

import neptune

from artifact_experiment.base.tracking.backend import TrackingBackend


class NoActiveNeptuneRunError(Exception):
    pass


class NeptuneBackend(TrackingBackend[neptune.Run]):
    ROOT_DIR = "artifact_ml"
    _neptune_api_token = ""
    _time_to_wait_before_stopping_seconds = 1

    @property
    def experiment_id(self) -> str:
        return self._native_client["sys/experiment/name"].fetch()

    @property
    def run_id(self) -> str:
        return self._native_client["sys/id"].fetch()

    @property
    def run_is_active(self) -> bool:
        state = self._native_client["sys/state"].fetch()
        return state == "active"

    def _start_experiment(self, run_id: str):
        self._native_client = self._get_native_client(
            experiment_id=self.experiment_id, run_id=run_id
        )

    def _stop_experiment(self):
        time.sleep(self._time_to_wait_before_stopping_seconds)
        self._native_client.stop()

    @classmethod
    def _get_native_client(cls, experiment_id: str, run_id: str) -> neptune.Run:
        native_client = neptune.init_run(
            api_token=cls._neptune_api_token,
            project=experiment_id,
            custom_run_id=run_id,
        )
        return native_client
