from typing import Any, Dict

from artifact_core._base.typing.artifact_result import Array

from tests._base.dummy.registries.base import DummyArtifactRegistry
from tests._base.dummy.types.arrays import DummyArrayType


class DummyArrayRegistry(DummyArtifactRegistry[DummyArrayType, Array]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
