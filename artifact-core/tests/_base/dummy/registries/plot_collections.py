from typing import Any, Dict

from artifact_core._base.typing.artifact_result import PlotCollection

from tests._base.dummy.registries.base import DummyArtifactRegistry
from tests._base.dummy.types.plot_collections import DummyPlotCollectionType


class DummyPlotCollectionRegistry(DummyArtifactRegistry[DummyPlotCollectionType, PlotCollection]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
