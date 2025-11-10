import os
from typing import Dict

from artifact_experiment.libs.tracking.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptuneScoreCollectionLogger(NeptuneArtifactLogger[Dict[str, float]]):
    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("score_collections", item_name)
