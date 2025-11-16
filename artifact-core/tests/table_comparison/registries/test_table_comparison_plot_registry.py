from typing import Type

import pytest
from artifact_core._libs.resources_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison._artifacts.plots.cdf import CDFPlot
from artifact_core.table_comparison._artifacts.plots.correlations import (
    CorrelationHeatmapJuxtapositionPlot,
)
from artifact_core.table_comparison._artifacts.plots.descriptive_stats import (
    DescriptiveStatsAlignmentPlot,
    FirstQuartileAlignmentPlot,
    MaxAlignmentPlot,
    MeanAlignmentPlot,
    MedianAlignmentPlot,
    MinAlignmentPlot,
    STDAlignmentPlot,
    ThirdQuartileAlignmentPlot,
    VarianceAlignmentPlot,
)
from artifact_core.table_comparison._artifacts.plots.pca import (
    PCAJuxtapositionPlot,
)
from artifact_core.table_comparison._artifacts.plots.pdf import PDFPlot
from artifact_core.table_comparison._artifacts.plots.truncated_svd import (
    TruncatedSVDJuxtapositionPlot,
)
from artifact_core.table_comparison._artifacts.plots.tsne import (
    TSNEJuxtapositionPlot,
)
from artifact_core.table_comparison._registries.plots.plots import (
    TableComparisonPlot,
    TableComparisonPlotRegistry,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonPlot.PDF, PDFPlot),
        (TableComparisonPlot.CDF, CDFPlot),
        (
            TableComparisonPlot.DESCRIPTIVE_STATS_ALIGNMENT,
            DescriptiveStatsAlignmentPlot,
        ),
        (TableComparisonPlot.MEAN_ALIGNMENT, MeanAlignmentPlot),
        (TableComparisonPlot.STD_ALIGNMENT, STDAlignmentPlot),
        (TableComparisonPlot.VARIANCE_ALIGNMENT, VarianceAlignmentPlot),
        (TableComparisonPlot.MEDIAN_ALIGNMENT, MedianAlignmentPlot),
        (
            TableComparisonPlot.FIRST_QUARTILE_ALIGNMENT,
            FirstQuartileAlignmentPlot,
        ),
        (
            TableComparisonPlot.THIRD_QUARTILE_ALIGNMENT,
            ThirdQuartileAlignmentPlot,
        ),
        (TableComparisonPlot.MIN_ALIGNMENT, MinAlignmentPlot),
        (TableComparisonPlot.MAX_ALIGNMENT, MaxAlignmentPlot),
        (
            TableComparisonPlot.CORRELATION_HEATMAP_JUXTAPOSITION,
            CorrelationHeatmapJuxtapositionPlot,
        ),
        (TableComparisonPlot.PCA_JUXTAPOSITION, PCAJuxtapositionPlot),
        (
            TableComparisonPlot.TRUNCATED_SVD_JUXTAPOSITION,
            TruncatedSVDJuxtapositionPlot,
        ),
        (TableComparisonPlot.TSNE_JUXTAPOSITION, TSNEJuxtapositionPlot),
    ],
)
def test_get(
    resource_spec: TabularDataSpecProtocol,
    artifact_type: TableComparisonPlot,
    artifact_class: Type[TableComparisonPlot],
):
    artifact = TableComparisonPlotRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
