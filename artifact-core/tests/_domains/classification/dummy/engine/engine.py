from typing import Type

from tests._domains.classification.dummy.engine.base import DummyClassificationEngineBase
from tests._domains.classification.dummy.registries.array_collections import (
    DummyClassificationArrayCollectionRegistry,
)
from tests._domains.classification.dummy.registries.arrays import (
    DummyClassificationArrayRegistry,
)
from tests._domains.classification.dummy.registries.plot_collections import (
    DummyClassificationPlotCollectionRegistry,
)
from tests._domains.classification.dummy.registries.plots import (
    DummyClassificationPlotRegistry,
)
from tests._domains.classification.dummy.registries.score_collections import (
    DummyClassificationScoreCollectionRegistry,
)
from tests._domains.classification.dummy.registries.scores import (
    DummyClassificationScoreRegistry,
)
from tests._domains.classification.dummy.types.array_collections import (
    DummyClassificationArrayCollectionType,
)
from tests._domains.classification.dummy.types.arrays import DummyClassificationArrayType
from tests._domains.classification.dummy.types.plot_collections import (
    DummyClassificationPlotCollectionType,
)
from tests._domains.classification.dummy.types.plots import DummyClassificationPlotType
from tests._domains.classification.dummy.types.score_collections import (
    DummyClassificationScoreCollectionType,
)
from tests._domains.classification.dummy.types.scores import DummyClassificationScoreType


class DummyClassificationEngine(
    DummyClassificationEngineBase[
        DummyClassificationScoreType,
        DummyClassificationArrayType,
        DummyClassificationPlotType,
        DummyClassificationScoreCollectionType,
        DummyClassificationArrayCollectionType,
        DummyClassificationPlotCollectionType,
    ]
):
    @classmethod
    def _get_score_registry(cls) -> Type[DummyClassificationScoreRegistry]:
        return DummyClassificationScoreRegistry

    @classmethod
    def _get_array_registry(cls) -> Type[DummyClassificationArrayRegistry]:
        return DummyClassificationArrayRegistry

    @classmethod
    def _get_plot_registry(cls) -> Type[DummyClassificationPlotRegistry]:
        return DummyClassificationPlotRegistry

    @classmethod
    def _get_score_collection_registry(cls) -> Type[DummyClassificationScoreCollectionRegistry]:
        return DummyClassificationScoreCollectionRegistry

    @classmethod
    def _get_array_collection_registry(cls) -> Type[DummyClassificationArrayCollectionRegistry]:
        return DummyClassificationArrayCollectionRegistry

    @classmethod
    def _get_plot_collection_registry(cls) -> Type[DummyClassificationPlotCollectionRegistry]:
        return DummyClassificationPlotCollectionRegistry
