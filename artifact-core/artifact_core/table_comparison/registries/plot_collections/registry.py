from typing import Any, Dict

from artifact_core.table_comparison.config.parsed import (
    DICT_PLOT_COLLECTIONS_CONFIG,
)
from artifact_core.table_comparison.registries.base import (
    TableComparisonPlotCollectionRegistryBase,
)
from artifact_core.table_comparison.registries.plot_collections.types import (
    TableComparisonPlotCollectionType,
)


class TableComparisonPlotCollectionRegistry(
    TableComparisonPlotCollectionRegistryBase[TableComparisonPlotCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_PLOT_COLLECTIONS_CONFIG
