from typing import Type

import pytest
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonScore,
)
from artifact_core.table_comparison.artifacts.scores.correlation import (
    CorrelationDistanceScore,
)
from artifact_core.table_comparison.artifacts.scores.mean_js import MeanJSDistanceScore
from artifact_core.table_comparison.registries.scores.registry import (
    TableComparisonScoreRegistry,
    TableComparisonScoreType,
)


@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonScoreType.MEAN_JS_DISTANCE, MeanJSDistanceScore),
        (TableComparisonScoreType.CORRELATION_DISTANCE, CorrelationDistanceScore),
    ],
)
def test_get(
    resource_spec: TabularDataSpecProtocol,
    artifact_type: TableComparisonScoreType,
    artifact_class: Type[TableComparisonScore],
):
    artifact = TableComparisonScoreRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
