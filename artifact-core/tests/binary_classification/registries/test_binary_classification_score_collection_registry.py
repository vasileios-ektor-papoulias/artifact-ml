from typing import Type

import pytest
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationScoreCollection
from artifact_core.binary_classification._artifacts.score_collections.confusion import (
    NormalizedConfusionCounts,
)
from artifact_core.binary_classification._artifacts.score_collections.ground_truth_prob import (
    GroundTruthProbStats,
)
from artifact_core.binary_classification._artifacts.score_collections.prediction_metrics import (
    BinaryPredictionScores,
)
from artifact_core.binary_classification._artifacts.score_collections.score_stats_by_split import (
    ScoreFirstQuartiles,
    ScoreMaxima,
    ScoreMeans,
    ScoreMedians,
    ScoreMinima,
    ScoreStds,
    ScoreThirdQuartiles,
    ScoreVariances,
)
from artifact_core.binary_classification._artifacts.score_collections.score_stats_by_stat import (
    NegativeClassScoreStats,
    PositiveClassScoreStats,
    ScoreStats,
)
from artifact_core.binary_classification._artifacts.score_collections.threshold_variation import (
    ThresholdVariationScores,
)
from artifact_core.binary_classification._registries.score_collections import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification._types.score_collections import (
    BinaryClassificationScoreCollectionType,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (
            BinaryClassificationScoreCollectionType.NORMALIZED_CONFUSION_COUNTS,
            NormalizedConfusionCounts,
        ),
        (
            BinaryClassificationScoreCollectionType.BINARY_PREDICTION_SCORES,
            BinaryPredictionScores,
        ),
        (
            BinaryClassificationScoreCollectionType.THRESHOLD_VARIATION_SCORES,
            ThresholdVariationScores,
        ),
        (BinaryClassificationScoreCollectionType.SCORE_STATS, ScoreStats),
        (
            BinaryClassificationScoreCollectionType.POSITIVE_CLASS_SCORE_STATS,
            PositiveClassScoreStats,
        ),
        (
            BinaryClassificationScoreCollectionType.NEGATIVE_CLASS_SCORE_STATS,
            NegativeClassScoreStats,
        ),
        (BinaryClassificationScoreCollectionType.SCORE_MEANS, ScoreMeans),
        (BinaryClassificationScoreCollectionType.SCORE_STDS, ScoreStds),
        (BinaryClassificationScoreCollectionType.SCORE_VARIANCES, ScoreVariances),
        (BinaryClassificationScoreCollectionType.SCORE_MEDIANS, ScoreMedians),
        (BinaryClassificationScoreCollectionType.SCORE_FIRST_QUARTILES, ScoreFirstQuartiles),
        (BinaryClassificationScoreCollectionType.SCORE_THIRD_QUARTILES, ScoreThirdQuartiles),
        (BinaryClassificationScoreCollectionType.SCORE_MINIMA, ScoreMinima),
        (BinaryClassificationScoreCollectionType.SCORE_MAXIMA, ScoreMaxima),
        (BinaryClassificationScoreCollectionType.GROUND_TRUTH_PROB_STATS, GroundTruthProbStats),
    ],
)
def test_get(
    resource_spec: BinaryClassSpecProtocol,
    artifact_type: BinaryClassificationScoreCollectionType,
    artifact_class: Type[BinaryClassificationScoreCollection],
):
    artifact = BinaryClassificationScoreCollectionRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
