from typing import Any, Dict

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationPlotRegistryBase,
)
from artifact_core.binary_classification._registries.plots.types import (
    BinaryClassificationPlotType,
)


class BinaryClassificationPlotRegistry(
    BinaryClassificationPlotRegistryBase[BinaryClassificationPlotType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_plots_config
