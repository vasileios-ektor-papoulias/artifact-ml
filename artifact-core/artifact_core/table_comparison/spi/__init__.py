from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonArtifact,
    TableComparisonArtifactResources,
)
from artifact_core.table_comparison._registries.array_collections.registry import (
    TableComparisonArrayCollectionRegistry,
)
from artifact_core.table_comparison._registries.arrays.registry import (
    TableComparisonArrayRegistry,
)
from artifact_core.table_comparison._registries.plot_collections.registry import (
    TableComparisonPlotCollectionRegistry,
)
from artifact_core.table_comparison._registries.plots.registry import (
    TableComparisonPlotRegistry,
)
from artifact_core.table_comparison._registries.score_collections.registry import (
    TableComparisonScoreCollectionRegistry,
)
from artifact_core.table_comparison._registries.scores.registry import TableComparisonScoreRegistry

__all__ = [
    "TabularDataSpecProtocol",
    "TableComparisonArtifact",
    "TableComparisonArtifactResources",
    "TableComparisonArrayCollectionRegistry",
    "TableComparisonArrayRegistry",
    "TableComparisonPlotCollectionRegistry",
    "TableComparisonPlotRegistry",
    "TableComparisonScoreCollectionRegistry",
    "TableComparisonScoreRegistry",
]
