from typing import Type

import pytest
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison.artifacts.plots.cdf import CDFPlot
from artifact_core.table_comparison.artifacts.plots.correlations import (
    CorrelationHeatmaps,
)
from artifact_core.table_comparison.artifacts.plots.descriptive_stats import (
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
from artifact_core.table_comparison.artifacts.plots.pca import (
    PCAPlot,
)
from artifact_core.table_comparison.artifacts.plots.pdf import PDFPlot
from artifact_core.table_comparison.artifacts.plots.truncated_svd import (
    TruncatedSVDPlot,
)
from artifact_core.table_comparison.artifacts.plots.tsne import (
    TSNEPlot,
)
from artifact_core.table_comparison.registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)


@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonPlotType.PDF_PLOT, PDFPlot),
        (TableComparisonPlotType.CDF_PLOT, CDFPlot),
        (
            TableComparisonPlotType.DESCRIPTIVE_STATS_ALIGNMENT_PLOT,
            DescriptiveStatsAlignmentPlot,
        ),
        (TableComparisonPlotType.MEAN_ALIGNMENT_PLOT, MeanAlignmentPlot),
        (TableComparisonPlotType.STD_ALIGNMENT_PLOT, STDAlignmentPlot),
        (TableComparisonPlotType.VARIANCE_ALIGNMENT_PLOT, VarianceAlignmentPlot),
        (TableComparisonPlotType.MEDIAN_ALIGNMENT_PLOT, MedianAlignmentPlot),
        (
            TableComparisonPlotType.FIRST_QUARTILE_ALIGNMENT_PLOT,
            FirstQuartileAlignmentPlot,
        ),
        (
            TableComparisonPlotType.THIRD_QUARTILE_ALIGNMENT_PLOT,
            ThirdQuartileAlignmentPlot,
        ),
        (TableComparisonPlotType.MIN_ALIGNMENT_PLOT, MinAlignmentPlot),
        (TableComparisonPlotType.MAX_ALIGNMENT_PLOT, MaxAlignmentPlot),
        (
            TableComparisonPlotType.CORRELATION_HEATMAPS,
            CorrelationHeatmaps,
        ),
        (TableComparisonPlotType.PCA_PROJECTION_PLOT, PCAPlot),
        (
            TableComparisonPlotType.TRUNCATED_SVD_PROJECTION_PLOT,
            TruncatedSVDPlot,
        ),
        (TableComparisonPlotType.TSNE_PROJECTION_PLOT, TSNEPlot),
    ],
)
def test_get(
    resource_spec: TabularDataSpecProtocol,
    artifact_type: TableComparisonPlotType,
    artifact_class: Type[TableComparisonPlot],
):
    artifact = TableComparisonPlotRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
