from typing import Any, Mapping

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationPlotRegistryBase,
)
from artifact_core.binary_classification._types.plots import BinaryClassificationPlotType


class BinaryClassificationPlotRegistry(
    BinaryClassificationPlotRegistryBase[BinaryClassificationPlotType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.plots_config
