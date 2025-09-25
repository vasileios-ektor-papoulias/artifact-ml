from typing import Any, Dict

from artifact_core.binary_classification.config.parsed import (
    DICT_SCORE_COLLECTIONS_CONFIG,
)
from artifact_core.binary_classification.registries.base import (
    BinaryClassificationScoreCollectionRegistryBase,
)
from artifact_core.binary_classification.registries.score_collections.types import (
    BinaryClassificationScoreCollectionType,
)


class BinaryClassificationScoreCollectionRegistry(
    BinaryClassificationScoreCollectionRegistryBase[BinaryClassificationScoreCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_SCORE_COLLECTIONS_CONFIG
