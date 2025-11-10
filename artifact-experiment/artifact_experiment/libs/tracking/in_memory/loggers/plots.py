from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.in_memory.loggers.artifacts import InMemoryArtifactLogger


class InMemoryPlotLogger(InMemoryArtifactLogger[Figure]):
    def _append(self, item_path: str, item: Figure):
        step = 1 + len(self._run.search_plot_store(artifact_path=item_path))
        key = self._get_store_key(item_path=item_path, step=step)
        self._run.log_plot(path=key, plot=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return f"plots/{item_name}"
