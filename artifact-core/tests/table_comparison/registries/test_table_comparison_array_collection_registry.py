from typing import Type

import pytest
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.array_collections.descriptive_stats import (
    FirstQuartileJuxtaposition,
    MaxJuxtaposition,
    MeanJuxtaposition,
    MedianJuxtaposition,
    MinJuxtaposition,
    STDJuxtaposition,
    ThirdQuartileJuxtaposition,
    VarianceJuxtaposition,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonArrayCollection,
)
from artifact_core.table_comparison.registries.array_collections.registry import (
    TableComparisonArrayCollectionRegistry,
    TableComparisonArrayCollectionType,
)


@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonArrayCollectionType.MEAN_JUXTAPOSITION, MeanJuxtaposition),
        (TableComparisonArrayCollectionType.STD_JUXTAPOSITION, STDJuxtaposition),
        (TableComparisonArrayCollectionType.VARIANCE_JUXTAPOSITION, VarianceJuxtaposition),
        (TableComparisonArrayCollectionType.MEDIAN_JUXTAPOSITION, MedianJuxtaposition),
        (
            TableComparisonArrayCollectionType.FIRST_QUARTILE_JUXTAPOSITION,
            FirstQuartileJuxtaposition,
        ),
        (
            TableComparisonArrayCollectionType.THIRD_QUARTILE_JUXTAPOSITION,
            ThirdQuartileJuxtaposition,
        ),
        (TableComparisonArrayCollectionType.MIN_JUXTAPOSITION, MinJuxtaposition),
        (TableComparisonArrayCollectionType.MAX_JUXTAPOSITION, MaxJuxtaposition),
    ],
)
def test_get(
    resource_spec: TabularDataSpecProtocol,
    artifact_type: TableComparisonArrayCollectionType,
    artifact_class: Type[TableComparisonArrayCollection],
):
    artifact = TableComparisonArrayCollectionRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
