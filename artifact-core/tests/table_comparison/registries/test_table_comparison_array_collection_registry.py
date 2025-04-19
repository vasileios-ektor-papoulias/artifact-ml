from typing import Type

import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.array_collections.descriptive_statistics import (
    ContinuousFeatureFirstQuartilesJuxtaposition,
    ContinuousFeatureMaximaJuxtaposition,
    ContinuousFeatureMeansJuxtaposition,
    ContinuousFeatureMediansJuxtaposition,
    ContinuousFeatureMinimaJuxtaposition,
    ContinuousFeatureSTDsJuxtaposition,
    ContinuousFeatureThirdQuartilesJuxtaposition,
    ContinuousFeatureVariancesJuxtaposition,
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
        (TableComparisonArrayCollectionType.MEANS, ContinuousFeatureMeansJuxtaposition),
        (TableComparisonArrayCollectionType.STDS, ContinuousFeatureSTDsJuxtaposition),
        (TableComparisonArrayCollectionType.VARIANCES, ContinuousFeatureVariancesJuxtaposition),
        (TableComparisonArrayCollectionType.MEDIANS, ContinuousFeatureMediansJuxtaposition),
        (
            TableComparisonArrayCollectionType.FIRST_QUARTILES,
            ContinuousFeatureFirstQuartilesJuxtaposition,
        ),
        (
            TableComparisonArrayCollectionType.THIRD_QUARTILES,
            ContinuousFeatureThirdQuartilesJuxtaposition,
        ),
        (TableComparisonArrayCollectionType.MINIMA, ContinuousFeatureMinimaJuxtaposition),
        (TableComparisonArrayCollectionType.MAXIMA, ContinuousFeatureMaximaJuxtaposition),
    ],
)
def test_get(
    data_spec: TabularDataSpecProtocol,
    artifact_type: TableComparisonArrayCollectionType,
    artifact_class: Type[TableComparisonArrayCollection],
):
    artifact = TableComparisonArrayCollectionRegistry.get(
        artifact_type=artifact_type, data_spec=data_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.data_spec == data_spec
