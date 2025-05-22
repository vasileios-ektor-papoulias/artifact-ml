import os

from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneArtifactLogger


class NeptuneScoreLogger(NeptuneArtifactLogger[float]):
    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("scores", artifact_name)
