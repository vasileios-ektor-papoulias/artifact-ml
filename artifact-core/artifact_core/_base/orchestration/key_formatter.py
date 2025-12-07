from typing import Union

from artifact_core._base.primitives.artifact_type import ArtifactType


class ArtifactKeyFormatter:
    @staticmethod
    def get_artifact_key(artifact_type: Union[ArtifactType, str]) -> str:
        if isinstance(artifact_type, str):
            key = artifact_type
        else:
            key = artifact_type.name
        return key
