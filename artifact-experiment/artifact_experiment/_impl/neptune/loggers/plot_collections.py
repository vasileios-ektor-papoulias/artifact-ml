import os

from artifact_core.typing import PlotCollection

from artifact_experiment._impl.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptunePlotCollectionLogger(NeptuneArtifactLogger[PlotCollection]):
    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("plot_collections", item_name)
