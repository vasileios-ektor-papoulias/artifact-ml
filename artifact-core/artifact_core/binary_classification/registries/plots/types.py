from artifact_core.base.registry import ArtifactType


class BinaryClassificationPlotType(ArtifactType):
    CONFUSION_MATRIX = "confusion_matrix"
    ROC = "roc"
    PR = "pr"
    DET = "det"
    TPR_THRESHOLD = "tpr_threshold"
    PRECISION_THRESHOLD = "precision_threshold"
