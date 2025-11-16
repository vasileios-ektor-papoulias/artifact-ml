from artifact_core.typing import ArrayCollection

from artifact_experiment._impl.backends.in_memory.loggers.artifacts import InMemoryArtifactLogger


class InMemoryArrayCollectionLogger(InMemoryArtifactLogger[ArrayCollection]):
    def _append(self, item_path: str, item: ArrayCollection):
        step = 1 + len(self._run.search_array_collection_store(store_path=item_path))
        key = self._get_store_key(item_path=item_path, step=step)
        self._run.log_array_collection(path=key, array_collection=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return f"array_collections/{item_name}"
