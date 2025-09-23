from artifact_core.base.registry import ArtifactType


class BinaryClassificationPlotType(ArtifactType):
    CONFUSION_MATRIX_PLOT = "confusion_matrix_plot"
    ROC_CURVE = "roc_curve"
    PR_CURVE = "pr_curve"
    DET_CURVE = "det_curve"
    TPR_THRESHOLD_CURVE = "tpr_threshold_curve"
    PRECISION_THRESHOLD_CURVE = "precision_threshold_curve"
    SCORE_DISTRIBUTION_PLOT = "score_distribution_plot"
