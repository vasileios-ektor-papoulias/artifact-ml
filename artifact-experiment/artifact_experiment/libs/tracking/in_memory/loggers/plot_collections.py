from typing import Dict

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.in_memory.loggers.base import (
    InMemoryArtifactLogger,
)


class InMemoryPlotCollectionLogger(InMemoryArtifactLogger[Dict[str, Figure]]):
    def _append(self, artifact_path: str, artifact: Dict[str, Figure]):
        step = self._run.n_plot_collections + 1
        key = self._get_store_key(artifact_path, step)
        self._run._native_run.dict_plot_collections[key] = artifact

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plot_collections/{artifact_name}"
