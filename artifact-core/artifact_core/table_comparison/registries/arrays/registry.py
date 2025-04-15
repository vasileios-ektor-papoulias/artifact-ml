from typing import Any, Dict

from artifact_core.table_comparison.config.parsed import DICT_ARRAYS_CONFIG
from artifact_core.table_comparison.registries.arrays.types import (
    TableComparisonArrayType,
)
from artifact_core.table_comparison.registries.base import (
    TableComparisonArrayRegistryBase,
)


class TableComparisonArrayRegistry(TableComparisonArrayRegistryBase[TableComparisonArrayType]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_ARRAYS_CONFIG
