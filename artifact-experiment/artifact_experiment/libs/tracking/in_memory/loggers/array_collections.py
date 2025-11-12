from typing import Dict

from artifact_experiment.libs.tracking.in_memory.loggers.artifacts import InMemoryArtifactLogger


class InMemoryArrayCollectionLogger(InMemoryArtifactLogger[Dict[str, Array]]):
    def _append(self, item_path: str, item: Dict[str, Array]):
        step = 1 + len(self._run.search_array_collection_store(artifact_path=item_path))
        key = self._get_store_key(item_path=item_path, step=step)
        self._run.log_array_collection(path=key, array_collection=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return f"array_collections/{item_name}"
