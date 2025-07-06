from artifact_core.base.artifact_dependencies import ArtifactResult
from artifact_experiment.base.tracking.adapter import RunAdapter


class DummyNativeRun:
    def __init__(self, experiment_id: str, run_id: str):
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._is_active = True

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def is_active(self) -> bool:
        return self._is_active

    @is_active.setter
    def is_active(self, is_active: bool):
        self._is_active = is_active

    def log(self, artifact_path: str, artifact: ArtifactResult):
        _ = artifact_path
        _ = artifact
        pass


class DummyTrackingBackend(RunAdapter[DummyNativeRun]):
    @property
    def experiment_id(self) -> str:
        return self._native_run.experiment_id

    @property
    def run_id(self) -> str:
        return self._native_run.run_id

    @property
    def is_active(self) -> bool:
        return self._native_run.is_active

    def stop(self):
        self._native_run.is_active = False

    def upload(self, path_source: str, dir_target: str):
        _ = path_source
        _ = dir_target
        pass

    def log(self, artifact_path: str, artifact: ArtifactResult):
        self._native_run.log(artifact_path=artifact_path, artifact=artifact)

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> DummyNativeRun:
        return DummyNativeRun(experiment_id=experiment_id, run_id=run_id)

    def _start(self, run_id: str):
        if self.run_id != run_id:
            self._native_run = self._build_native_run(
                experiment_id=self.experiment_id, run_id=run_id
            )
        else:
            self._native_run.is_active = True
