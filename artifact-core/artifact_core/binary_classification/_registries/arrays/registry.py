from typing import Any, Dict

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.arrays.types import (
    BinaryClassificationArrayType,
)
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationArrayRegistryBase,
)


class BinaryClassificationArrayRegistry(
    BinaryClassificationArrayRegistryBase[BinaryClassificationArrayType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_arrays_config
