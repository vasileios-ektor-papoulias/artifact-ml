from typing import Type

import pytest
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationScore
from artifact_core.binary_classification._artifacts.scores.ground_truth_prob import (
    GroundTruthProbFirstQuartile,
    GroundTruthProbMax,
    GroundTruthProbMean,
    GroundTruthProbMedian,
    GroundTruthProbMin,
    GroundTruthProbSTD,
    GroundTruthProbThirdQuartile,
    GroundTruthProbVariance,
)
from artifact_core.binary_classification._artifacts.scores.prediction_metrics import (
    AccuracyScore,
    BalancedAccuracyScore,
    F1ScoreScore,
    FNRScore,
    FPRScore,
    MCCScore,
    NPVScore,
    PrecisionScore,
    RecallScore,
    TNRScore,
)
from artifact_core.binary_classification._artifacts.scores.threshold_variation import (
    PRAUCScore,
    ROCAUCScore,
)
from artifact_core.binary_classification._registries.scores import (
    BinaryClassificationScoreRegistry,
)
from artifact_core.binary_classification._types.scores import BinaryClassificationScoreType


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (BinaryClassificationScoreType.ACCURACY, AccuracyScore),
        (BinaryClassificationScoreType.BALANCED_ACCURACY, BalancedAccuracyScore),
        (BinaryClassificationScoreType.PRECISION, PrecisionScore),
        (BinaryClassificationScoreType.NPV, NPVScore),
        (BinaryClassificationScoreType.RECALL, RecallScore),
        (BinaryClassificationScoreType.TNR, TNRScore),
        (BinaryClassificationScoreType.FPR, FPRScore),
        (BinaryClassificationScoreType.FNR, FNRScore),
        (BinaryClassificationScoreType.F1, F1ScoreScore),
        (BinaryClassificationScoreType.MCC, MCCScore),
        (BinaryClassificationScoreType.ROC_AUC, ROCAUCScore),
        (BinaryClassificationScoreType.PR_AUC, PRAUCScore),
        (BinaryClassificationScoreType.GROUND_TRUTH_PROB_MEAN, GroundTruthProbMean),
        (BinaryClassificationScoreType.GROUND_TRUTH_PROB_STD, GroundTruthProbSTD),
        (BinaryClassificationScoreType.GROUND_TRUTH_PROB_VARIANCE, GroundTruthProbVariance),
        (BinaryClassificationScoreType.GROUND_TRUTH_PROB_MEDIAN, GroundTruthProbMedian),
        (
            BinaryClassificationScoreType.GROUND_TRUTH_PROB_FIRST_QUARTILE,
            GroundTruthProbFirstQuartile,
        ),
        (
            BinaryClassificationScoreType.GROUND_TRUTH_PROB_THIRD_QUARTILE,
            GroundTruthProbThirdQuartile,
        ),
        (BinaryClassificationScoreType.GROUND_TRUTH_PROB_MIN, GroundTruthProbMin),
        (BinaryClassificationScoreType.GROUND_TRUTH_PROB_MAX, GroundTruthProbMax),
    ],
)
def test_get(
    resource_spec: BinaryClassSpecProtocol,
    artifact_type: BinaryClassificationScoreType,
    artifact_class: Type[BinaryClassificationScore],
):
    artifact = BinaryClassificationScoreRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
