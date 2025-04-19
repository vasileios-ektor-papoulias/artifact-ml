from typing import Type

import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonArray,
)
from artifact_core.table_comparison.registries.arrays.registry import (
    TableComparisonArrayRegistry,
    TableComparisonArrayType,
)


@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [],
)
def test_get(
    data_spec: TabularDataSpecProtocol,
    artifact_type: TableComparisonArrayType,
    artifact_class: Type[TableComparisonArray],
):
    artifact = TableComparisonArrayRegistry.get(artifact_type=artifact_type, data_spec=data_spec)
    assert isinstance(artifact, artifact_class)
    assert artifact.data_spec == data_spec
