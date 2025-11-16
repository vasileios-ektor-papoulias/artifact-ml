import os

from artifact_core.typing import PlotCollection

from artifact_experiment._impl.backends.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptunePlotCollectionLogger(NeptuneArtifactLogger[PlotCollection]):
    def _append(self, item_path: str, item: PlotCollection):
        self._run.log(artifact_path=item_path, artifact=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("plot_collections", item_name)
