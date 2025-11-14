import os

from artifact_core.typing import Array

from artifact_experiment._impl.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptuneArrayLogger(NeptuneArtifactLogger[Array]):
    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("arrays", item_name)
