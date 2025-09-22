from artifact_core.base.registry import ArtifactType


class BinaryClassificationScoreCollectionType(ArtifactType):
    CONFUSION_COUNTS = "confusion_counts"
    BINARY_PREDICTION_SCORES = "binary_prediction_scores"
    THRESHOLD_VARIATION_SCORES = "threshold_variation_scores"
