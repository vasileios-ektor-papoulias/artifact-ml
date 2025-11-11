from typing import Any, Dict

from artifact_core.table_comparison._config.parsed import CONFIG
from artifact_core.table_comparison._registries.base import (
    TableComparisonScoreRegistryBase,
)
from artifact_core.table_comparison._registries.scores.types import (
    TableComparisonScoreType,
)


class TableComparisonScoreRegistry(TableComparisonScoreRegistryBase[TableComparisonScoreType]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_scores_config
