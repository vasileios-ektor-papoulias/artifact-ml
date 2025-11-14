from artifact_core._base.primitives.artifact_type import ArtifactType


class TableComparisonPlotType(ArtifactType):
    PDF = "pdf"
    CDF = "cdf"
    DESCRIPTIVE_STATS_ALIGNMENT = "descriptive_stats_alignment_plot"
    MEAN_ALIGNMENT = "mean_alignment_plot"
    STD_ALIGNMENT = "std_alignment_plot"
    VARIANCE_ALIGNMENT = "variance_alignment_plot"
    MEDIAN_ALIGNMENT = "median_alignment_plot"
    FIRST_QUARTILE_ALIGNMENT = "first_quartile_alignment_plot"
    THIRD_QUARTILE_ALIGNMENT = "third_quartile_alignment_plot"
    MIN_ALIGNMENT = "min_alignment_plot"
    MAX_ALIGNMENT = "max_alignment_plot"
    CORRELATION_HEATMAP_JUXTAPOSITION = "correlation_heatmap_juxtaposition"
    PCA_JUXTAPOSITION = "pca_juxtaposition"
    TRUNCATED_SVD_JUXTAPOSITION = "truncated_svd_juxtaposition"
    TSNE_JUXTAPOSITION = "tsne_juxtaposition"
