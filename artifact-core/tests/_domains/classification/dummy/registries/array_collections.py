from typing import Any, Dict

from artifact_core._base.typing.artifact_result import ArrayCollection

from tests._domains.classification.dummy.registries.base import DummyClassificationRegistry
from tests._domains.classification.dummy.types.array_collections import (
    DummyClassificationArrayCollectionType,
)


class DummyClassificationArrayCollectionRegistry(
    DummyClassificationRegistry[DummyClassificationArrayCollectionType, ArrayCollection]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
