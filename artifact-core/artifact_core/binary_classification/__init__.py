from artifact_core._libs.resource_specs.binary_classification.spec import BinaryClassSpec
from artifact_core.binary_classification._engine.engine import BinaryClassificationEngine
from artifact_core.binary_classification._types.array_collections import (
    BinaryClassificationArrayCollectionType,
)
from artifact_core.binary_classification._types.arrays import BinaryClassificationArrayType
from artifact_core.binary_classification._types.plot_collections import (
    BinaryClassificationPlotCollectionType,
)
from artifact_core.binary_classification._types.plots import BinaryClassificationPlotType
from artifact_core.binary_classification._types.score_collections import (
    BinaryClassificationScoreCollectionType,
)
from artifact_core.binary_classification._types.scores import BinaryClassificationScoreType

__all__ = [
    "BinaryClassSpec",
    "BinaryClassificationEngine",
    "BinaryClassificationArrayCollectionType",
    "BinaryClassificationArrayType",
    "BinaryClassificationPlotCollectionType",
    "BinaryClassificationPlotType",
    "BinaryClassificationScoreCollectionType",
    "BinaryClassificationScoreType",
]


def _init_toolkit():
    from artifact_core._bootstrap.primitives.domain_toolkit import DomainToolkit
    from artifact_core._bootstrap.toolkit_initializer import ToolkitInitializer

    ToolkitInitializer.init_toolkit(domain_toolkit=DomainToolkit.BINARY_CLASSIFICATION)


_init_toolkit()
del _init_toolkit
