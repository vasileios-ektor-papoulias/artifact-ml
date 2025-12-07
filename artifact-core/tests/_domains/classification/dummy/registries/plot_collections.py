from typing import Any, Dict

from artifact_core._base.typing.artifact_result import PlotCollection

from tests._domains.classification.dummy.registries.base import DummyClassificationRegistry
from tests._domains.classification.dummy.types.plot_collections import (
    DummyClassificationPlotCollectionType,
)


class DummyClassificationPlotCollectionRegistry(
    DummyClassificationRegistry[DummyClassificationPlotCollectionType, PlotCollection]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}

