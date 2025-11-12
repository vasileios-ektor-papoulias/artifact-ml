from typing import Type

import pytest
from artifact_core._libs.resources_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonArray,
)
from artifact_core.table_comparison._registries.arrays.registry import (
    TableComparisonArray,
    TableComparisonArrayRegistry,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [],
)
def test_get(
    resource_spec: TabularDataSpecProtocol,
    artifact_type: TableComparisonArray,
    artifact_class: Type[TableComparisonArray],
):
    artifact = TableComparisonArrayRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
