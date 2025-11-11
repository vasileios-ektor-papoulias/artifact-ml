from typing import Any, Dict

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationPlotCollectionRegistryBase,
)
from artifact_core.binary_classification._registries.plot_collections.types import (
    BinaryClassificationPlotCollectionType,
)


class BinaryClassificationPlotCollectionRegistry(
    BinaryClassificationPlotCollectionRegistryBase[BinaryClassificationPlotCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return CONFIG.dict_plot_collections_config
