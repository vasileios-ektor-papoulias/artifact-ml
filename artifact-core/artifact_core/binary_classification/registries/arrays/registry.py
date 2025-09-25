from typing import Any, Dict

from artifact_core.binary_classification.config.parsed import DICT_ARRAYS_CONFIG
from artifact_core.binary_classification.registries.arrays.types import (
    BinaryClassificationArrayType,
)
from artifact_core.binary_classification.registries.base import (
    BinaryClassificationArrayRegistryBase,
)


class BinaryClassificationArrayRegistry(
    BinaryClassificationArrayRegistryBase[BinaryClassificationArrayType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_ARRAYS_CONFIG
