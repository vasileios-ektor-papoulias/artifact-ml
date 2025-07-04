from typing import Type

import pytest
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison.artifacts.plots.cdf import CDFPlot
from artifact_core.table_comparison.artifacts.plots.correlations import (
    CorrelationHeatmapJuxtapositionPlot,
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
    PCAJuxtapositionPlot,
)
from artifact_core.table_comparison.artifacts.plots.pdf import PDFPlot
from artifact_core.table_comparison.artifacts.plots.truncated_svd import (
    TruncatedSVDJuxtapositionPlot,
)
from artifact_core.table_comparison.artifacts.plots.tsne import (
    TSNEJuxtapositionPlot,
)
from artifact_core.table_comparison.registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)


@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonPlotType.PDF, PDFPlot),
        (TableComparisonPlotType.CDF, CDFPlot),
        (
            TableComparisonPlotType.DESCRIPTIVE_STATS_ALIGNMENT,
            DescriptiveStatsAlignmentPlot,
        ),
        (TableComparisonPlotType.MEAN_ALIGNMENT, MeanAlignmentPlot),
        (TableComparisonPlotType.STD_ALIGNMENT, STDAlignmentPlot),
        (TableComparisonPlotType.VARIANCE_ALIGNMENT, VarianceAlignmentPlot),
        (TableComparisonPlotType.MEDIAN_ALIGNMENT, MedianAlignmentPlot),
        (
            TableComparisonPlotType.FIRST_QUARTILE_ALIGNMENT,
            FirstQuartileAlignmentPlot,
        ),
        (
            TableComparisonPlotType.THIRD_QUARTILE_ALIGNMENT,
            ThirdQuartileAlignmentPlot,
        ),
        (TableComparisonPlotType.MIN_ALIGNMENT, MinAlignmentPlot),
        (TableComparisonPlotType.MAX_ALIGNMENT, MaxAlignmentPlot),
        (
            TableComparisonPlotType.CORRELATION_HEATMAP_JUXTAPOSITION,
            CorrelationHeatmapJuxtapositionPlot,
        ),
        (TableComparisonPlotType.PCA_JUXTAPOSITION, PCAJuxtapositionPlot),
        (
            TableComparisonPlotType.TRUNCATED_SVD_JUXTAPOSITION,
            TruncatedSVDJuxtapositionPlot,
        ),
        (TableComparisonPlotType.TSNE_JUXTAPOSITION, TSNEJuxtapositionPlot),
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
