from typing import Type

import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonScoreCollection,
)
from artifact_core.table_comparison.artifacts.score_collections.js import JSDistance
from artifact_core.table_comparison.registries.score_collections.registry import (
    TableComparisonScoreCollectionRegistry,
    TableComparisonScoreCollectionType,
)


@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonScoreCollectionType.JS_DISTANCE, JSDistance),
    ],
)
def test_get(
    data_spec: TabularDataSpecProtocol,
    artifact_type: TableComparisonScoreCollectionType,
    artifact_class: Type[TableComparisonScoreCollection],
):
    artifact = TableComparisonScoreCollectionRegistry.get(
        artifact_type=artifact_type, data_spec=data_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.data_spec == data_spec
