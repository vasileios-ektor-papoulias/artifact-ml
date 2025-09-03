from artifact_core.base.artifact_dependencies import ArtifactResult
from artifact_experiment.base.tracking.logger import ArtifactLogger

from tests.base.tracking.dummy.adapter import DummyRunAdapter


class DummyArtifactLogger(ArtifactLogger[ArtifactResult, DummyRunAdapter]):
    def _append(self, artifact_path: str, artifact: ArtifactResult):
        self._run.log(artifact_path=artifact_path, artifact=artifact)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return artifact_name

    def _get_root_dir(self) -> str:
        return ""
