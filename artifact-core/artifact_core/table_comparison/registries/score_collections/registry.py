from typing import Any, Dict

from artifact_core.table_comparison.config.parsed import (
    DICT_SCORE_COLLECTIONS_CONFIG,
)
from artifact_core.table_comparison.registries.base import (
    TableComparisonScoreCollectionRegistryBase,
)
from artifact_core.table_comparison.registries.score_collections.types import (
    TableComparisonScoreCollectionType,
)


class TableComparisonScoreCollectionRegistry(
    TableComparisonScoreCollectionRegistryBase[TableComparisonScoreCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_SCORE_COLLECTIONS_CONFIG
