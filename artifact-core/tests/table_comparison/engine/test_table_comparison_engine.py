from artifact_core._libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._engine.engine import (
    TableComparisonArrayCollectionRegistry,
    TableComparisonArrayRegistry,
    TableComparisonEngine,
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotRegistry,
    TableComparisonScoreCollectionRegistry,
    TableComparisonScoreRegistry,
)


def test_state(
    resource_spec: TabularDataSpecProtocol,
):
    engine = TableComparisonEngine(resource_spec=resource_spec)
    assert engine.resource_spec == resource_spec
    assert engine.score_registry == TableComparisonScoreRegistry
    assert engine.array_registry == TableComparisonArrayRegistry
    assert engine.plot_registry == TableComparisonPlotRegistry
    assert engine.score_collection_registry == TableComparisonScoreCollectionRegistry
    assert engine.array_collection_registry == TableComparisonArrayCollectionRegistry
    assert engine.plot_collection_registry == TableComparisonPlotCollectionRegistry
