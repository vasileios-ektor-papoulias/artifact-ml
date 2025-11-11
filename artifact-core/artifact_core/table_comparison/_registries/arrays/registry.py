from typing import Any, Dict

from artifact_core.table_comparison._config.parsed import CONFIG
from artifact_core.table_comparison._registries.arrays.types import (
    TableComparisonArrayType,
)
from artifact_core.table_comparison._registries.base import (
    TableComparisonArrayRegistryBase,
)


class TableComparisonArrayRegistry(TableComparisonArrayRegistryBase[TableComparisonArrayType]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_arrays_config
