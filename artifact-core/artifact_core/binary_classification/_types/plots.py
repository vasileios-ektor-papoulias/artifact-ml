from artifact_core._base.primitives.artifact_type import ArtifactType


class BinaryClassificationPlotType(ArtifactType):
    CONFUSION_MATRIX_PLOT = "confusion_matrix_plot"
    ROC_CURVE = "roc_curve"
    PR_CURVE = "pr_curve"
    DET_CURVE = "det_curve"
    RECALL_THRESHOLD_CURVE = "recall_threshold_curve"
    PRECISION_THRESHOLD_CURVE = "precision_threshold_curve"
    SCORE_PDF = "score_pdf"
    GROUND_TRUTH_PROB_PDF = "ground_truth_prob_pdf"
