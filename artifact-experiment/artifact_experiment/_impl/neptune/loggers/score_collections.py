import os

from artifact_core.typing import ScoreCollection

from artifact_experiment._impl.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptuneScoreCollectionLogger(NeptuneArtifactLogger[ScoreCollection]):
    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("score_collections", item_name)
