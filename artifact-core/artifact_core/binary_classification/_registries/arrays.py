from typing import Any, Mapping

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationArrayRegistryBase,
)
from artifact_core.binary_classification._types.arrays import BinaryClassificationArrayType


class BinaryClassificationArrayRegistry(
    BinaryClassificationArrayRegistryBase[BinaryClassificationArrayType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.arrays_config
