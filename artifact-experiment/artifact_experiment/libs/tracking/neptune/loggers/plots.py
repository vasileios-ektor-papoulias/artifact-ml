import os

from artifact_experiment.libs.tracking.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptunePlotLogger(NeptuneArtifactLogger[Figure]):
    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("plots", item_name)
