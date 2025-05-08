import time

import neptune
from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.adapter import InactiveRunError, RunAdapter


class NeptuneRunStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class InactiveNeptuneRunError(InactiveRunError):
    pass


class NeptuneRunAdapter(RunAdapter[neptune.Run]):
    _time_to_wait_before_stopping_seconds: int = 1
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
    def run_status(self) -> str:
        return self._native_run["sys/state"].fetch()

    @property
    def is_active(self) -> bool:
        return self.run_status == NeptuneRunStatus.ACTIVE

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
