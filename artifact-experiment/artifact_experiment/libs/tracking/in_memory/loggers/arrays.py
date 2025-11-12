from artifact_experiment.libs.tracking.in_memory.loggers.artifacts import InMemoryArtifactLogger


class InMemoryArrayLogger(InMemoryArtifactLogger[Array]):
    def _append(self, item_path: str, item: Array):
        step = 1 + len(self._run.search_array_store(artifact_path=item_path))
        key = self._get_store_key(item_path=item_path, step=step)
        self._run.log_array(path=key, array=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return f"arrays/{item_name}"
