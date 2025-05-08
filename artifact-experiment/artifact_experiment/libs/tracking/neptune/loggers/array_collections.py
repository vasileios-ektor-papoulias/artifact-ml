from typing import Dict

from numpy import ndarray

from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneArtifactLogger


class NeptuneArrayCollectionLogger(NeptuneArtifactLogger[Dict[str, ndarray]]):
    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"array_collections/{artifact_name}"
