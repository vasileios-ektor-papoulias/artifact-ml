from artifact_core.base.registry import ArtifactType


class TableComparisonPlotType(ArtifactType):
    PDF_PLOT = "pdf_plot"
    CDF_PLOT = "cdf_plot"
    DESCRIPTIVE_STATS_ALIGNMENT_PLOT = "descriptive_stats_alignment_plot"
    MEAN_ALIGNMENT_PLOT = "mean_alignment_plot"
    STD_ALIGNMENT_PLOT = "std_alignment_plot"
    VARIANCE_ALIGNMENT_PLOT = "variance_alignment_plot"
    MEDIAN_ALIGNMENT_PLOT = "median_alignment_plot"
    FIRST_QUARTILE_ALIGNMENT_PLOT = "first_quartile_alignment_plot"
    THIRD_QUARTILE_ALIGNMENT_PLOT = "third_quartile_alignment_plot"
    MIN_ALIGNMENT_PLOT = "min_alignment_plot"
    MAX_ALIGNMENT_PLOT = "max_alignment_plot"
    CORRELATION_HEATMAP_PLOT = "correlation_heatmap_plot"
    PCA_PROJECTION_PLOT = "pca_projection_plot"
    TRUNCATED_SVD_PROJECTION_PLOT = "truncated_svd_projection_plot"
    TSNE_PROJECTION_PLOT = "tsne_projection_plot"
