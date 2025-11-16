from artifact_core.typing import Plot

from artifact_experiment._impl.backends.in_memory.loggers.artifacts import InMemoryArtifactLogger


class InMemoryPlotLogger(InMemoryArtifactLogger[Plot]):
    def _append(self, item_path: str, item: Plot):
        step = 1 + len(self._run.search_plot_store(store_path=item_path))
        key = self._get_store_key(item_path=item_path, step=step)
        self._run.log_plot(path=key, plot=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return f"plots/{item_name}"
