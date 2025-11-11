from typing import Any, Dict

from artifact_core.table_comparison._config.parsed import CONFIG
from artifact_core.table_comparison._registries.base import (
    TableComparisonPlotRegistryBase,
)
from artifact_core.table_comparison._registries.plots.types import (
    TableComparisonPlotType,
)


class TableComparisonPlotRegistry(TableComparisonPlotRegistryBase[TableComparisonPlotType]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_plots_config
