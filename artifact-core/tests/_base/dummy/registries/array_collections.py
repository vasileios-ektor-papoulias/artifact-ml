from typing import Any, Dict

from artifact_core._base.typing.artifact_result import ArrayCollection

from tests._base.dummy.registries.base import DummyArtifactRegistry
from tests._base.dummy.types.array_collections import DummyArrayCollectionType


class DummyArrayCollectionRegistry(
    DummyArtifactRegistry[DummyArrayCollectionType, ArrayCollection]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
