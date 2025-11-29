from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
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
    assert engine._score_registry == TableComparisonScoreRegistry
    assert engine._array_registry == TableComparisonArrayRegistry
    assert engine._plot_registry == TableComparisonPlotRegistry
    assert engine._score_collection_registry == TableComparisonScoreCollectionRegistry
    assert engine._array_collection_registry == TableComparisonArrayCollectionRegistry
    assert engine._plot_collection_registry == TableComparisonPlotCollectionRegistry
