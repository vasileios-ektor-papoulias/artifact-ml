from typing import Any, Dict

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationScoreRegistryBase,
)
from artifact_core.binary_classification._registries.scores.types import (
    BinaryClassificationScoreType,
)


class BinaryClassificationScoreRegistry(
    BinaryClassificationScoreRegistryBase[BinaryClassificationScoreType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_score_collections_config
