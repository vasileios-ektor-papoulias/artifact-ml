from typing import Any, Dict

from artifact_core.binary_classification.config.parsed import DICT_SCORES_CONFIG
from artifact_core.binary_classification.registries.base import (
    BinaryClassificationScoreRegistryBase,
)
from artifact_core.binary_classification.registries.scores.types import (
    BinaryClassificationScoreType,
)


class BinaryClassificationScoreRegistry(
    BinaryClassificationScoreRegistryBase[BinaryClassificationScoreType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_SCORES_CONFIG
