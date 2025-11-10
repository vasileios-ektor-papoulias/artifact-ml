import os

from numpy import ndarray

from artifact_experiment.libs.tracking.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptuneArrayLogger(NeptuneArtifactLogger[ndarray]):
    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("arrays", item_name)
