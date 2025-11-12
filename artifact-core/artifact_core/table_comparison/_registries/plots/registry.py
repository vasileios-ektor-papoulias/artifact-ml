from typing import Any, Mapping

from artifact_core.table_comparison._config.parsed import CONFIG
from artifact_core.table_comparison._registries.base import TableComparisonPlotRegistryBase
from artifact_core.table_comparison._registries.plots.types import TableComparisonPlotType


class TableComparisonPlotRegistry(TableComparisonPlotRegistryBase[TableComparisonPlotType]):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.plots_config
