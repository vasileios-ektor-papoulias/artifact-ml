from typing import Type

import pytest
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonScoreCollection,
)
from artifact_core.table_comparison._artifacts.score_collections.js import JSDistanceScores
from artifact_core.table_comparison._registries.score_collections import (
    TableComparisonScoreCollectionRegistry,
    TableComparisonScoreCollectionType,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonScoreCollectionType.JS_DISTANCE, JSDistanceScores),
    ],
)
def test_get(
    resource_spec: TabularDataSpecProtocol,
    artifact_type: TableComparisonScoreCollectionType,
    artifact_class: Type[TableComparisonScoreCollection],
):
    artifact = TableComparisonScoreCollectionRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
