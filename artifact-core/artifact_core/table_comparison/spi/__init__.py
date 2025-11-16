from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonArray,
    TableComparisonArrayCollection,
    TableComparisonArtifact,
    TableComparisonPlot,
    TableComparisonPlotCollection,
    TableComparisonScore,
    TableComparisonScoreCollection,
)
from artifact_core.table_comparison._registries.array_collections import (
    TableComparisonArrayCollectionRegistry,
)
from artifact_core.table_comparison._registries.arrays import TableComparisonArrayRegistry
from artifact_core.table_comparison._registries.plot_collections import (
    TableComparisonPlotCollectionRegistry,
)
from artifact_core.table_comparison._registries.plots import TableComparisonPlotRegistry
from artifact_core.table_comparison._registries.score_collections import (
    TableComparisonScoreCollectionRegistry,
)
from artifact_core.table_comparison._registries.scores import TableComparisonScoreRegistry
from artifact_core.table_comparison._resources import TableComparisonArtifactResources

__all__ = [
    "TabularDataSpecProtocol",
    "TableComparisonArray",
    "TableComparisonArrayCollection",
    "TableComparisonArtifact",
    "TableComparisonPlot",
    "TableComparisonPlotCollection",
    "TableComparisonScore",
    "TableComparisonScoreCollection",
    "TableComparisonArtifactResources",
    "TableComparisonArrayCollectionRegistry",
    "TableComparisonArrayRegistry",
    "TableComparisonPlotCollectionRegistry",
    "TableComparisonPlotRegistry",
    "TableComparisonScoreCollectionRegistry",
    "TableComparisonScoreRegistry",
]
