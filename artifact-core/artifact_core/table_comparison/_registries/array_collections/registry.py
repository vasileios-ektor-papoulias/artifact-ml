from typing import Any, Dict

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
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_array_collections_config
