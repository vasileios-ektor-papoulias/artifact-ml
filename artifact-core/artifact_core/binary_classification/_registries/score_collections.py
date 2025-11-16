from typing import Any, Mapping

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationScoreCollectionRegistryBase,
)
from artifact_core.binary_classification._types.score_collections import (
    BinaryClassificationScoreCollectionType,
)


class BinaryClassificationScoreCollectionRegistry(
    BinaryClassificationScoreCollectionRegistryBase[BinaryClassificationScoreCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.score_collections_config
