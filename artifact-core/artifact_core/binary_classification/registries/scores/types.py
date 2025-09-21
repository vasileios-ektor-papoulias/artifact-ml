from artifact_core.base.registry import ArtifactType


class BinaryClassificationScoreType(ArtifactType):
    ACCURACY = "ACCURACY"
    BALANCED_ACCURACY = "BALANCED_ACCURACY"
    PRECISION = "PRECISION"
    NPV = "NPV"
    RECALL = "RECALL"
    TNR = "TNR"
    FPR = "FPR"
    FNR = "FNR"
    F1 = "F1"
    MCC = "MCC"
