from typing import Type

import pytest
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison.artifacts.plots.cdf import CDFComparisonCombinedPlot
from artifact_core.table_comparison.artifacts.plots.descriptive_stat_comparison import (
    ContinuousFeatureDescriptiveStatsComparisonPlot,
    ContinuousFeatureFirstQuartileComparisonPlot,
    ContinuousFeatureMaximaComparisonPlot,
    ContinuousFeatureMeanComparisonPlot,
    ContinuousFeatureMedianComparisonPlot,
    ContinuousFeatureMinimaComparisonPlot,
    ContinuousFeatureSTDComparisonPlot,
    ContinuousFeatureThirdQuartileComparisonPlot,
    ContinuousFeatureVarianceComparisonPlot,
)
from artifact_core.table_comparison.artifacts.plots.pairwise_correlations import (
    CorrelationComparisonCombinedPlot,
)
from artifact_core.table_comparison.artifacts.plots.pca_projection import (
    PCAProjectionComparisonPlot,
)
from artifact_core.table_comparison.artifacts.plots.pdf import PDFComparisonCombinedPlot
from artifact_core.table_comparison.artifacts.plots.truncated_svd_projection import (
    TruncatedSVDProjectionComparisonPlot,
)
from artifact_core.table_comparison.artifacts.plots.tsne_projection import (
    TSNEProjectionComparisonPlot,
)
from artifact_core.table_comparison.registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)


@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (TableComparisonPlotType.PDF_PLOT, PDFComparisonCombinedPlot),
        (TableComparisonPlotType.CDF_PLOT, CDFComparisonCombinedPlot),
        (
            TableComparisonPlotType.DESCRIPTIVE_STATS_COMPARISON_PLOT,
            ContinuousFeatureDescriptiveStatsComparisonPlot,
        ),
        (TableComparisonPlotType.MEAN_COMPARISON_PLOT, ContinuousFeatureMeanComparisonPlot),
        (TableComparisonPlotType.STD_COMPARISON_PLOT, ContinuousFeatureSTDComparisonPlot),
        (TableComparisonPlotType.VARIANCE_COMPARISON_PLOT, ContinuousFeatureVarianceComparisonPlot),
        (TableComparisonPlotType.MEDIAN_COMPARISON_PLOT, ContinuousFeatureMedianComparisonPlot),
        (
            TableComparisonPlotType.FIRST_QUARTILE_COMPARISON_PLOT,
            ContinuousFeatureFirstQuartileComparisonPlot,
        ),
        (
            TableComparisonPlotType.THIRD_QUARTILE_COMPARISON_PLOT,
            ContinuousFeatureThirdQuartileComparisonPlot,
        ),
        (TableComparisonPlotType.MIN_COMPARISON_PLOT, ContinuousFeatureMinimaComparisonPlot),
        (TableComparisonPlotType.MAX_COMPARISON_PLOT, ContinuousFeatureMaximaComparisonPlot),
        (
            TableComparisonPlotType.PAIRWISE_CORRELATION_COMPARISON_HEATMAP,
            CorrelationComparisonCombinedPlot,
        ),
        (TableComparisonPlotType.PCA_PROJECTION_PLOT, PCAProjectionComparisonPlot),
        (
            TableComparisonPlotType.TRUNCATED_SVD_PROJECTION_PLOT,
            TruncatedSVDProjectionComparisonPlot,
        ),
        (TableComparisonPlotType.TSNE_PROJECTION_PLOT, TSNEProjectionComparisonPlot),
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
