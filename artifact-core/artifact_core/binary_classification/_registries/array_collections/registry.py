from typing import Any, Dict

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.array_collections.types import (
    BinaryClassificationArrayCollectionType,
)
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationArrayCollectionRegistryBase,
)


class BinaryClassificationArrayCollectionRegistry(
    BinaryClassificationArrayCollectionRegistryBase[BinaryClassificationArrayCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_array_collections_config
