import os

from artifact_core.typing import Array

from artifact_experiment._impl.backends.neptune.loggers.artifacts import NeptuneArtifactLogger
from artifact_experiment._utils.collections.array_stringifier import ArrayStringifer


class NeptuneArrayLogger(NeptuneArtifactLogger[Array]):
    def _append(self, item_path: str, item: Array):
        array_stringified = ArrayStringifer.stringify(array=item)
        self._run.log(artifact_path=item_path, artifact=array_stringified)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("arrays", item_name)
