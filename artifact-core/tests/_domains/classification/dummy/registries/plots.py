from typing import Any, Dict

from artifact_core._base.typing.artifact_result import Plot

from tests._domains.classification.dummy.registries.base import DummyClassificationRegistry
from tests._domains.classification.dummy.types.plots import DummyClassificationPlotType


class DummyClassificationPlotRegistry(
    DummyClassificationRegistry[DummyClassificationPlotType, Plot]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}

