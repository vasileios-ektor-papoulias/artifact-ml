from typing import Type

import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonScore,
)
from artifact_core.table_comparison.artifacts.scores.mean_js import MeanJSDistance
from artifact_core.table_comparison.artifacts.scores.pairwise_correlation_distance import (
    PairwiseCorrelationDistance,
)
from artifact_core.table_comparison.registries.scores.registry import (
    TableComparisonScoreRegistry,
    TableComparisonScoreType,
)


@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonScoreType.MEAN_JS_DISTANCE, MeanJSDistance),
        (TableComparisonScoreType.PAIRWISE_CORRELATION_DISTANCE, PairwiseCorrelationDistance),
    ],
)
def test_get(
    data_spec: TabularDataSpecProtocol,
    artifact_type: TableComparisonScoreType,
    artifact_class: Type[TableComparisonScore],
):
    artifact = TableComparisonScoreRegistry.get(artifact_type=artifact_type, data_spec=data_spec)
    assert isinstance(artifact, artifact_class)
    assert artifact.data_spec == data_spec
