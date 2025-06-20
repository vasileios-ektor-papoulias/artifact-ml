import os

from numpy import ndarray

from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneArtifactLogger


class NeptuneArrayLogger(NeptuneArtifactLogger[ndarray]):
    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("arrays", artifact_name)
