from typing import Any, Dict

from artifact_core.binary_classification.config.parsed import DICT_PLOTS_CONFIG
from artifact_core.binary_classification.registries.base import (
    BinaryClassificationPlotRegistryBase,
)
from artifact_core.binary_classification.registries.plots.types import (
    BinaryClassificationPlotType,
)


class BinaryClassificationPlotRegistry(
    BinaryClassificationPlotRegistryBase[BinaryClassificationPlotType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_PLOTS_CONFIG
