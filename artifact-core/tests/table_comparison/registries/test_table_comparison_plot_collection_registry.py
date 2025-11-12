from typing import Type

import pytest
from artifact_core._libs.resources_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonPlotCollection,
)
from artifact_core.table_comparison._artifacts.plot_collections.cdf import (
    CDFPlots,
)
from artifact_core.table_comparison._artifacts.plot_collections.correlations import (
    CorrelationHeatmaps,
)
from artifact_core.table_comparison._artifacts.plot_collections.pdf import (
    PDFPlots,
)
from artifact_core.table_comparison._registries.plot_collections.registry import (
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotCollectionType,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonPlotCollectionType.CDF, CDFPlots),
        (TableComparisonPlotCollectionType.PDF, PDFPlots),
        (
            TableComparisonPlotCollectionType.CORRELATION_HEATMAPS,
            CorrelationHeatmaps,
        ),
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
