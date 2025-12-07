from typing import Any, Dict

from artifact_core._base.typing.artifact_result import Plot

from tests._base.dummy.registries.base import DummyArtifactRegistry
from tests._base.dummy.types.plots import DummyPlotType


class DummyPlotRegistry(DummyArtifactRegistry[DummyPlotType, Plot]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
