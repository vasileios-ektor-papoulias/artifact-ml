import os

from artifact_core.typing import Plot

from artifact_experiment._impl.backends.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptunePlotLogger(NeptuneArtifactLogger[Plot]):
    def _append(self, item_path: str, item: Plot):
        self._run.log(artifact_path=item_path, artifact=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("plots", item_name)
