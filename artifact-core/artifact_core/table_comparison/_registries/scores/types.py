from artifact_core._base.registry import ArtifactType


class TableComparisonScoreType(ArtifactType):
    MEAN_JS_DISTANCE = "mean_js_distance"
    CORRELATION_DISTANCE = "correlation_distance"
