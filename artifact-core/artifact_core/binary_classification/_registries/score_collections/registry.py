from typing import Any, Dict

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationScoreCollectionRegistryBase,
)
from artifact_core.binary_classification._registries.score_collections.types import (
    BinaryClassificationScoreCollectionType,
)


class BinaryClassificationScoreCollectionRegistry(
    BinaryClassificationScoreCollectionRegistryBase[BinaryClassificationScoreCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_scores_config
