import os
from typing import Dict

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneArtifactLogger


class NeptunePlotCollectionLogger(NeptuneArtifactLogger[Dict[str, Figure]]):
    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("plot_collections", artifact_name)
