from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.engine.engine import (
    TableComparisonArrayCollectionRegistry,
    TableComparisonArrayRegistry,
    TableComparisonEngine,
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotRegistry,
    TableComparisonScoreCollectionRegistry,
    TableComparisonScoreRegistry,
)


def test_state(
    data_spec: TabularDataSpecProtocol,
):
    engine = TableComparisonEngine(data_spec=data_spec)
    assert engine.data_spec == data_spec
    assert engine.score_registry == TableComparisonScoreRegistry
    assert engine.array_registry == TableComparisonArrayRegistry
    assert engine.plot_registry == TableComparisonPlotRegistry
    assert engine.score_collection_registry == TableComparisonScoreCollectionRegistry
    assert engine.array_collection_registry == TableComparisonArrayCollectionRegistry
    assert engine.plot_collection_registry == TableComparisonPlotCollectionRegistry
