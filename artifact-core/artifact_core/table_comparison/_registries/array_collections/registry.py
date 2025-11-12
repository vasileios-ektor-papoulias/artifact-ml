from typing import Any, Mapping

from artifact_core.table_comparison._config.parsed import CONFIG
from artifact_core.table_comparison._registries.array_collections.types import (
    TableComparisonArrayCollectionType,
)
from artifact_core.table_comparison._registries.base import (
    TableComparisonArrayCollectionRegistryBase,
)


class TableComparisonArrayCollectionRegistry(
    TableComparisonArrayCollectionRegistryBase[TableComparisonArrayCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.array_collections_config
