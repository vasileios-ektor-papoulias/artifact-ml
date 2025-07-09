from typing import Dict

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.in_memory.loggers.base import (
    InMemoryArtifactLogger,
)


class InMemoryPlotCollectionLogger(InMemoryArtifactLogger[Dict[str, Figure]]):
    def _append(self, artifact_path: str, artifact: Dict[str, Figure]):
        step = 1 + len(self._run.search_plot_collection_store(artifact_path=artifact_path))
        key = self._get_store_key(artifact_path, step)
        self._run.log_plot_collection(path=key, plot_collection=artifact)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plot_collections/{artifact_name}"
