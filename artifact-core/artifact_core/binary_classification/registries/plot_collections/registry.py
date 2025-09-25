from typing import Any, Dict

from artifact_core.binary_classification.config.parsed import (
    DICT_PLOT_COLLECTIONS_CONFIG,
)
from artifact_core.binary_classification.registries.base import (
    BinaryClassificationPlotCollectionRegistryBase,
)
from artifact_core.binary_classification.registries.plot_collections.types import (
    BinaryClassificationPlotCollectionType,
)


class BinaryClassificationPlotCollectionRegistry(
    BinaryClassificationPlotCollectionRegistryBase[BinaryClassificationPlotCollectionType]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return DICT_PLOT_COLLECTIONS_CONFIG
