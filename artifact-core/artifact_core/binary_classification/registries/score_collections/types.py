from artifact_core.base.registry import ArtifactType


class BinaryClassificationScoreCollectionType(ArtifactType):
    BINARY_PREDICTION_SCORES = "binary_prediction_scores"
    CONFUSION_COUNTS = "confusion_counts"
