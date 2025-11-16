import os
from typing import Dict

from artifact_core.typing import ArrayCollection

from artifact_experiment._impl.backends.neptune.loggers.artifacts import NeptuneArtifactLogger
from artifact_experiment._utils.collections.array_stringifier import ArrayStringifer


class NeptuneArrayCollectionLogger(NeptuneArtifactLogger[ArrayCollection]):
    def _append(self, item_path: str, item: ArrayCollection):
        array_collection_stringified = self._stringify(array_collection=item)
        self._run.log(artifact_path=item_path, artifact=array_collection_stringified)

    @staticmethod
    def _stringify(array_collection: ArrayCollection) -> Dict[str, str]:
        return {
            name: ArrayStringifer.stringify(array=array) for name, array in array_collection.items()
        }

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("array_collections", item_name)
