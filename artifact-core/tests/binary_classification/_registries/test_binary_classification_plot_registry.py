from typing import Type

import pytest
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationPlot
from artifact_core.binary_classification._artifacts.plots.confusion import ConfusionMatrixPlot
from artifact_core.binary_classification._artifacts.plots.ground_truth_prob import (
    GroundTruthProbPDFPlot,
)
from artifact_core.binary_classification._artifacts.plots.score_distribution import (
    ScoreDistributionPlot,
)
from artifact_core.binary_classification._artifacts.plots.threshold_variation import (
    DETCurve,
    PRCurve,
    PrecisionThresholdCurve,
    RecallThresholdCurve,
    ROCCurve,
)
from artifact_core.binary_classification._registries.plots import (
    BinaryClassificationPlotRegistry,
)
from artifact_core.binary_classification._types.plots import BinaryClassificationPlotType


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (BinaryClassificationPlotType.CONFUSION_MATRIX_PLOT, ConfusionMatrixPlot),
        (BinaryClassificationPlotType.ROC_CURVE, ROCCurve),
        (BinaryClassificationPlotType.PR_CURVE, PRCurve),
        (BinaryClassificationPlotType.DET_CURVE, DETCurve),
        (BinaryClassificationPlotType.RECALL_THRESHOLD_CURVE, RecallThresholdCurve),
        (BinaryClassificationPlotType.PRECISION_THRESHOLD_CURVE, PrecisionThresholdCurve),
        (BinaryClassificationPlotType.SCORE_PDF, ScoreDistributionPlot),
        (BinaryClassificationPlotType.GROUND_TRUTH_PROB_PDF, GroundTruthProbPDFPlot),
    ],
)
def test_get(
    resource_spec: BinaryClassSpecProtocol,
    artifact_type: BinaryClassificationPlotType,
    artifact_class: Type[BinaryClassificationPlot],
):
    artifact = BinaryClassificationPlotRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
