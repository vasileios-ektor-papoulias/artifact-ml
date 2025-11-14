import os

from artifact_core.typing import ArrayCollection

from artifact_experiment._impl.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptuneArrayCollectionLogger(NeptuneArtifactLogger[ArrayCollection]):
    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("array_collections", item_name)
