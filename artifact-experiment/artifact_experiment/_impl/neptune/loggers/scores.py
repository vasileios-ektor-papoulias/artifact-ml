import os

from artifact_core.typing import Score

from artifact_experiment._impl.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptuneScoreLogger(NeptuneArtifactLogger[Score]):
    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("scores", item_name)
