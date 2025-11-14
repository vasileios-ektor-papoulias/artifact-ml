from typing import Any, Mapping

from artifact_core.table_comparison._config.parsed import CONFIG
from artifact_core.table_comparison._registries.base import TableComparisonScoreRegistryBase
from artifact_core.table_comparison._types.scores import TableComparisonScoreType


class TableComparisonScoreRegistry(TableComparisonScoreRegistryBase[TableComparisonScoreType]):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.scores_config
