from typing import Any, Mapping

from artifact_core.table_comparison._config.parsed import CONFIG
from artifact_core.table_comparison._registries.base import (
    TableComparisonArrayCollectionRegistryBase,
)
from artifact_core.table_comparison._types.array_collections import (
    TableComparisonArrayCollectionType,
)


class TableComparisonArrayCollectionRegistry(
    TableComparisonArrayCollectionRegistryBase[TableComparisonArrayCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.array_collections_config
