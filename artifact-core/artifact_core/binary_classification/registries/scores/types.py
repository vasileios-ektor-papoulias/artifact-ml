from artifact_core.base.registry import ArtifactType


class BinaryClassificationScoreType(ArtifactType):
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    PRECISION = "precision"
    NPV = "npv"
    RECALL = "recall"
    TNR = "tnr"
    FPR = "fp"
    FNR = "fnr"
    F1 = "f1"
    MCC = "mcc"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
