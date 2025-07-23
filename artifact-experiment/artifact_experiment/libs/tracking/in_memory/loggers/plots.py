from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.in_memory.loggers.base import (
    InMemoryArtifactLogger,
)


class InMemoryPlotLogger(InMemoryArtifactLogger[Figure]):
    def _append(self, artifact_path: str, artifact: Figure):
        step = 1 + len(self._run.search_plot_store(artifact_path=artifact_path))
        key = self._get_store_key(artifact_path=artifact_path, step=step)
        self._run.log_plot(path=key, plot=artifact)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plots/{artifact_name}"
