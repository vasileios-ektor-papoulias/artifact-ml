from typing import Any, Dict

from artifact_core.table_comparison.config.parsed import DICT_PLOTS_CONFIG
from artifact_core.table_comparison.registries.base import (
    TableComparisonPlotRegistryBase,
)
from artifact_core.table_comparison.registries.plots.types import (
    TableComparisonPlotType,
)


class TableComparisonPlotRegistry(TableComparisonPlotRegistryBase[TableComparisonPlotType]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_PLOTS_CONFIG
