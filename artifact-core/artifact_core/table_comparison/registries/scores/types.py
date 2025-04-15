from artifact_core.base.registry import ArtifactType


class TableComparisonScoreType(ArtifactType):
    MEAN_JS_DISTANCE = "mean_js_distance"
    PAIRWISE_CORRELATION_DISTANCE = "pairwise_correlation_distance"
