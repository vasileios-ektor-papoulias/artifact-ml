from typing import Any, Dict

from artifact_core.table_comparison.config.parsed import (
    DICT_ARRAY_COLLECTIONS_CONFIG,
)
from artifact_core.table_comparison.registries.array_collections.types import (
    TableComparisonArrayCollectionType,
)
from artifact_core.table_comparison.registries.base import (
    TableComparisonArrayCollectionRegistryBase,
)


class TableComparisonArrayCollectionRegistry(
    TableComparisonArrayCollectionRegistryBase[TableComparisonArrayCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_ARRAY_COLLECTIONS_CONFIG
