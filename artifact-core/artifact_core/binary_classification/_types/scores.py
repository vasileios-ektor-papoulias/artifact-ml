from artifact_core._base.primitives.artifact_type import ArtifactType


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
    GROUND_TRUTH_PROB_MEAN = "ground_truth_prob_mean"
    GROUND_TRUTH_PROB_STD = "ground_truth_prob_std"
    GROUND_TRUTH_PROB_VARIANCE = "ground_truth_prob_variance"
    GROUND_TRUTH_PROB_MEDIAN = "ground_truth_prob_median"
    GROUND_TRUTH_PROB_FIRST_QUARTILE = "ground_truth_prob_first_quartile"
    GROUND_TRUTH_PROB_THIRD_QUARTILE = "ground_truth_prob_third_quartile"
    GROUND_TRUTH_PROB_MIN = "ground_truth_prob_min"
    GROUND_TRUTH_PROB_MAX = "ground_truth_prob_max"
