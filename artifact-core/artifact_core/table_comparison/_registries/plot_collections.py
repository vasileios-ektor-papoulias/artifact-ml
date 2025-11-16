from typing import Any, Mapping

from artifact_core.table_comparison._config.parsed import CONFIG
from artifact_core.table_comparison._registries.base import (
    TableComparisonPlotCollectionRegistryBase,
)
from artifact_core.table_comparison._types.plot_collections import TableComparisonPlotCollectionType


class TableComparisonPlotCollectionRegistry(
    TableComparisonPlotCollectionRegistryBase[TableComparisonPlotCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.plot_collections_config
