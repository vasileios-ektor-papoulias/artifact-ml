import os

from artifact_core.typing import ScoreCollection

from artifact_experiment._impl.backends.neptune.loggers.artifacts import NeptuneArtifactLogger


class NeptuneScoreCollectionLogger(NeptuneArtifactLogger[ScoreCollection]):
    def _append(self, item_path: str, item: ScoreCollection):
        self._run.log(artifact_path=item_path, artifact=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("score_collections", item_name)
