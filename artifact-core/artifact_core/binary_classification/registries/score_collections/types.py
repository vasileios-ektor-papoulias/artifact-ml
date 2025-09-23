from artifact_core.base.registry import ArtifactType


class BinaryClassificationScoreCollectionType(ArtifactType):
    CONFUSION_COUNTS = "confusion_counts"
    BINARY_PREDICTION_SCORES = "binary_prediction_scores"
    THRESHOLD_VARIATION_SCORES = "threshold_variation_scores"
    SCORE_STATS = "score_stats"
    POSITIVE_CLASS_SCORE_STATS = "positive_class_score_stats"
    NEGATIVE_CLASS_SCORE_STATS = "negative_class_score_stats"
    SCORE_MEANS = "score_means"
    SCORE_STDS = "score_stds"
    SCORE_VARIANCES = "score_variances"
    SCORE_MEDIANS = "score_medians"
    SCORE_FIRST_QUARTILES = "score_first_quartiles"
    SCORE_THIRD_QUARTILES = "score_third_quartiles"
    SCORE_MINIMA = "score_minima"
    SCORE_MAXIMA = "score_maxima"
