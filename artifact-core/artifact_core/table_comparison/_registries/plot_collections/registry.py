from typing import Any, Dict

from artifact_core.table_comparison._config.parsed import CONFIG
from artifact_core.table_comparison._registries.base import (
    TableComparisonPlotCollectionRegistryBase,
)
from artifact_core.table_comparison._registries.plot_collections.types import (
    TableComparisonPlotCollectionType,
)


class TableComparisonPlotCollectionRegistry(
    TableComparisonPlotCollectionRegistryBase[TableComparisonPlotCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_plot_collections_config
