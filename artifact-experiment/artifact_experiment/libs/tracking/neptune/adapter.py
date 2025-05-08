import time

import neptune
from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.adapter import RunAdapter


class InactiveNeptuneRunError(Exception):
    pass


class NeptuneRunAdapter(RunAdapter[neptune.Run]):
    _neptune_api_token = ""
    _time_to_wait_before_stopping_seconds = 1

    @property
    def experiment_id(self) -> str:
        return self._native_run["sys/experiment/name"].fetch()

    @property
    def id(self) -> str:
        return self._native_run["sys/id"].fetch()

    @property
    def is_active(self) -> bool:
        state = self._native_run["sys/state"].fetch()
        return state == "active"

    def log(self, path: str, artifact: ArtifactResult):
        if self.is_active:
            self._native_run[path] = artifact
        else:
            raise InactiveNeptuneRunError("Run is inactive")

    def start(self):
        self._native_run = self._build_native_run(experiment_id=self.experiment_id, run_id=self.id)

    def stop(self):
        time.sleep(self._time_to_wait_before_stopping_seconds)
        self._native_run.stop()

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> neptune.Run:
        native_run = neptune.init_run(
            api_token=cls._neptune_api_token,
            project=experiment_id,
            custom_run_id=run_id,
        )
        return native_run
