import os
from typing import Dict

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptunePlotCollectionLogger(NeptuneArtifactLogger[Dict[str, Figure]]):
    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("plot_collections", item_name)
