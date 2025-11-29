from typing import Type

import pytest
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.array_collections.descriptive_stats import (
    FirstQuartileJuxtapositionArrays,
    MaxJuxtapositionArrays,
    MeanJuxtapositionArrays,
    MedianJuxtapositionArrays,
    MinJuxtapositionArrays,
    STDJuxtapositionArrays,
    ThirdQuartileJuxtapositionArrays,
    VarianceJuxtapositionArrays,
)
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonArrayCollection,
)
from artifact_core.table_comparison._registries.array_collections import (
    TableComparisonArrayCollectionRegistry,
    TableComparisonArrayCollectionType,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonArrayCollectionType.MEAN_JUXTAPOSITION, MeanJuxtapositionArrays),
        (TableComparisonArrayCollectionType.STD_JUXTAPOSITION, STDJuxtapositionArrays),
        (TableComparisonArrayCollectionType.VARIANCE_JUXTAPOSITION, VarianceJuxtapositionArrays),
        (TableComparisonArrayCollectionType.MEDIAN_JUXTAPOSITION, MedianJuxtapositionArrays),
        (
            TableComparisonArrayCollectionType.FIRST_QUARTILE_JUXTAPOSITION,
            FirstQuartileJuxtapositionArrays,
        ),
        (
            TableComparisonArrayCollectionType.THIRD_QUARTILE_JUXTAPOSITION,
            ThirdQuartileJuxtapositionArrays,
        ),
        (TableComparisonArrayCollectionType.MIN_JUXTAPOSITION, MinJuxtapositionArrays),
        (TableComparisonArrayCollectionType.MAX_JUXTAPOSITION, MaxJuxtapositionArrays),
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
