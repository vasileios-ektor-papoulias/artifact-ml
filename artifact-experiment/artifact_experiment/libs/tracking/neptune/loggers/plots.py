from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneArtifactLogger


class NeptunePlotLogger(NeptuneArtifactLogger[Figure]):
    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plots/{artifact_name}"
