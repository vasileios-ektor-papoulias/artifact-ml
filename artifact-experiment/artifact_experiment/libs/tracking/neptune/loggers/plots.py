import os

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneArtifactLogger


class NeptunePlotLogger(NeptuneArtifactLogger[Figure]):
    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("plots", artifact_name)
