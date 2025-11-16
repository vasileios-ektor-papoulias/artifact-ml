from typing import Any, Mapping

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationArrayCollectionRegistryBase,
)
from artifact_core.binary_classification._types.array_collections import (
    BinaryClassificationArrayCollectionType,
)


class BinaryClassificationArrayCollectionRegistry(
    BinaryClassificationArrayCollectionRegistryBase[BinaryClassificationArrayCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.array_collections_config
