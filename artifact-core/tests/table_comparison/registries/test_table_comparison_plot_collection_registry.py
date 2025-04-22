from typing import Type

import pytest
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlotCollection,
)
from artifact_core.table_comparison.artifacts.plot_collections.cdf import (
    CDFComparisonPlotCollection,
)
from artifact_core.table_comparison.artifacts.plot_collections.pdf import (
    PDFComparisonPlotCollection,
)
from artifact_core.table_comparison.registries.plot_collections.registry import (
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotCollectionType,
)


@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonPlotCollectionType.CDF_PLOTS, CDFComparisonPlotCollection),
        (TableComparisonPlotCollectionType.PDF_PLOTS, PDFComparisonPlotCollection),
    ],
)
def test_get(
    resource_spec: TabularDataSpecProtocol,
    artifact_type: TableComparisonPlotCollectionType,
    artifact_class: Type[TableComparisonPlotCollection],
):
    artifact = TableComparisonPlotCollectionRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
