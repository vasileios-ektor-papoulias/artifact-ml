from typing import Any, Mapping

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
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.score_collections_config
