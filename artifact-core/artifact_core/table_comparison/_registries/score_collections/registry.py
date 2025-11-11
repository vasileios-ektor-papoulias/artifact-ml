from typing import Any, Dict

from artifact_core.table_comparison._config.parsed import CONFIG
from artifact_core.table_comparison._registries.base import (
    TableComparisonScoreCollectionRegistryBase,
)
from artifact_core.table_comparison._registries.score_collections.types import (
    TableComparisonScoreCollectionType,
)


class TableComparisonScoreCollectionRegistry(
    TableComparisonScoreCollectionRegistryBase[TableComparisonScoreCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_score_collections_config
