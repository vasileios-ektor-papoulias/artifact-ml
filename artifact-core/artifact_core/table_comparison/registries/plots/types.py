from artifact_core.base.registry import ArtifactType


class TableComparisonPlotType(ArtifactType):
    PDF_PLOT = "pdf_plot"
    CDF_PLOT = "cdf_plot"
    DESCRIPTIVE_STATS_COMPARISON_PLOT = "descriptive_stats_comparison_plot"
    MEAN_COMPARISON_PLOT = "means_comparison_plot"
    STD_COMPARISON_PLOT = "stds"
    VARIANCE_COMPARISON_PLOT = "variances"
    MEDIAN_COMPARISON_PLOT = "medians"
    FIRST_QUARTILE_COMPARISON_PLOT = "first_quartiles"
    THIRD_QUARTILE_COMPARISON_PLOT = "third_quartiles"
    MIN_COMPARISON_PLOT = "minima"
    MAX_COMPARISON_PLOT = "maxima"
    PAIRWISE_CORRELATION_COMPARISON_HEATMAP = "pairwise_correlation_comparison_heatmap"
    PCA_PROJECTION_PLOT = "pca_projection_plot"
    TRUNCATED_SVD_PROJECTION_PLOT = "truncated_svd_projection_plot"
    TSNE_PROJECTION_PLOT = "tsne_projection_plot"
