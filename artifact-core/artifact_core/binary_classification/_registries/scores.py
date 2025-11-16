from typing import Any, Mapping

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationScoreRegistryBase,
)
from artifact_core.binary_classification._types.scores import BinaryClassificationScoreType


class BinaryClassificationScoreRegistry(
    BinaryClassificationScoreRegistryBase[BinaryClassificationScoreType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.score_collections_config
