import os

from artifact_core.typing import Plot

from artifact_experiment._impl.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptunePlotLogger(NeptuneArtifactLogger[Plot]):
    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("plots", item_name)
