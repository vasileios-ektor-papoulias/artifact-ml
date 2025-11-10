import os
from typing import Dict

from numpy import ndarray

from artifact_experiment.libs.tracking.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptuneArrayCollectionLogger(NeptuneArtifactLogger[Dict[str, ndarray]]):
    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("array_collections", item_name)
