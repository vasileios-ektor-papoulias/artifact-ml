from typing import Any, Mapping

from artifact_core.binary_classification._config.parsed import CONFIG
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationPlotCollectionRegistryBase,
)
from artifact_core.binary_classification._types.plot_collections import (
    BinaryClassificationPlotCollectionType,
)


class BinaryClassificationPlotCollectionRegistry(
    BinaryClassificationPlotCollectionRegistryBase[BinaryClassificationPlotCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return CONFIG.plot_collections_config
