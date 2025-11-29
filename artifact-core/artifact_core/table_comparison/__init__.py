from artifact_core._libs.resource_specs.table_comparison.spec import TabularDataSpec
from artifact_core.table_comparison._engine.engine import TableComparisonEngine
from artifact_core.table_comparison._types.array_collections import (
    TableComparisonArrayCollectionType,
)
from artifact_core.table_comparison._types.arrays import TableComparisonArrayType
from artifact_core.table_comparison._types.plot_collections import (
    TableComparisonPlotCollectionType,
)
from artifact_core.table_comparison._types.plots import TableComparisonPlotType
from artifact_core.table_comparison._types.score_collections import (
    TableComparisonScoreCollectionType,
)
from artifact_core.table_comparison._types.scores import TableComparisonScoreType

__all__ = [
    "TabularDataSpec",
    "TableComparisonEngine",
    "TableComparisonArrayCollectionType",
    "TableComparisonArrayType",
    "TableComparisonPlotCollectionType",
    "TableComparisonPlotType",
    "TableComparisonScoreCollectionType",
    "TableComparisonScoreType",
]


def _init_toolkit():
    from artifact_core._bootstrap.primitives.domain_toolkit import DomainToolkit
    from artifact_core._bootstrap.toolkit_initializer import ToolkitInitializer

    ToolkitInitializer.init_toolkit(domain_toolkit=DomainToolkit.TABLE_COMPARISON)


_init_toolkit()
del _init_toolkit
