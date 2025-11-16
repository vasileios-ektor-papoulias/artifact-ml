import os

from artifact_core.typing import Score

from artifact_experiment._impl.backends.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptuneScoreLogger(NeptuneArtifactLogger[Score]):
    def _append(self, item_path: str, item: Score):
        self._run.log(artifact_path=item_path, artifact=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("scores", item_name)
