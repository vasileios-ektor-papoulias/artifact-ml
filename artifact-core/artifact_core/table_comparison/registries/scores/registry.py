from typing import Any, Dict

from artifact_core.table_comparison.config.parsed import DICT_SCORES_CONFIG
from artifact_core.table_comparison.registries.base import (
    TableComparisonScoreRegistryBase,
)
from artifact_core.table_comparison.registries.scores.types import (
    TableComparisonScoreType,
)


class TableComparisonScoreRegistry(TableComparisonScoreRegistryBase[TableComparisonScoreType]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_SCORES_CONFIG
