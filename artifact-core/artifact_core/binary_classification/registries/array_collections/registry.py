from typing import Any, Dict

from artifact_core.binary_classification.config.parsed import (
    DICT_ARRAY_COLLECTIONS_CONFIG,
)
from artifact_core.binary_classification.registries.array_collections.types import (
    BinaryClassificationArrayCollectionType,
)
from artifact_core.binary_classification.registries.base import (
    BinaryClassificationArrayCollectionRegistryBase,
)


class BinaryClassificationArrayCollectionRegistry(
    BinaryClassificationArrayCollectionRegistryBase[BinaryClassificationArrayCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_ARRAY_COLLECTIONS_CONFIG
