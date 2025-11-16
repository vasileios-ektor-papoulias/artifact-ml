from artifact_core.typing import PlotCollection

from artifact_experiment._impl.backends.in_memory.loggers.artifacts import InMemoryArtifactLogger


class InMemoryPlotCollectionLogger(InMemoryArtifactLogger[PlotCollection]):
    def _append(self, item_path: str, item: PlotCollection):
        step = 1 + len(self._run.search_plot_collection_store(store_path=item_path))
        key = self._get_store_key(item_path=item_path, step=step)
        self._run.log_plot_collection(path=key, plot_collection=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return f"plot_collections/{item_name}"
