from typing import Type

import pytest
from artifact_core._libs.resources_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonScore,
)
from artifact_core.table_comparison._artifacts.scores.correlation import (
    CorrelationDistanceScore,
)
from artifact_core.table_comparison._artifacts.scores.mean_js import MeanJSDistanceScore
from artifact_core.table_comparison._registries.scores.scores import (
    TableComparisonScoreRegistry,
    TableComparisonScoreType,
)


@pytest.mark.unit
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
