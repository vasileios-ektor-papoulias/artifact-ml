from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.in_memory.loggers.base import (
    InMemoryArtifactLogger,
)


class InMemoryPlotLogger(InMemoryArtifactLogger[Figure]):
    def _append(self, artifact_path: str, artifact: Figure):
        step = self._run.n_plots + 1
        key = self._get_store_key(artifact_path, step)
        self._run._native_run.dict_plots[key] = artifact

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plots/{artifact_name}"
