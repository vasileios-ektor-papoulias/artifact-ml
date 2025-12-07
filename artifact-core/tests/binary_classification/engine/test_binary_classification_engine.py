from typing import Any, Mapping, Type
from unittest.mock import ANY

import numpy as np
import pytest
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core._utils.collections.entity_store import IdentifierType
from artifact_core.binary_classification._artifacts.array_collections.confusion import (
    ConfusionMatrixCollection,
)
from artifact_core.binary_classification._artifacts.arrays.confusion import ConfusionMatrix
from artifact_core.binary_classification._artifacts.base import BinaryClassificationArtifact
from artifact_core.binary_classification._artifacts.plot_collections.confusion import (
    ConfusionMatrixPlotCollection,
)
from artifact_core.binary_classification._artifacts.plot_collections.score_distribution import (
    ScoreDistributionPlots,
)
from artifact_core.binary_classification._artifacts.plot_collections.threshold_variation import (
    ThresholdVariationCurves,
)
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
from artifact_core.binary_classification._engine.engine import BinaryClassificationEngine
from artifact_core.binary_classification._registries.array_collections import (
    BinaryClassificationArrayCollectionRegistry,
)
from artifact_core.binary_classification._registries.arrays import BinaryClassificationArrayRegistry
from artifact_core.binary_classification._registries.plot_collections import (
    BinaryClassificationPlotCollectionRegistry,
)
from artifact_core.binary_classification._registries.plots import BinaryClassificationPlotRegistry
from artifact_core.binary_classification._registries.score_collections import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification._registries.scores import BinaryClassificationScoreRegistry
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources
from artifact_core.binary_classification._types.array_collections import (
    BinaryClassificationArrayCollectionType,
)
from artifact_core.binary_classification._types.arrays import BinaryClassificationArrayType
from artifact_core.binary_classification._types.plot_collections import (
    BinaryClassificationPlotCollectionType,
)
from artifact_core.binary_classification._types.plots import BinaryClassificationPlotType
from artifact_core.binary_classification._types.score_collections import (
    BinaryClassificationScoreCollectionType,
)
from artifact_core.binary_classification._types.scores import BinaryClassificationScoreType
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def id_to_true() -> Mapping[IdentifierType, str]:
    return {0: "positive", 1: "negative", 2: "positive"}


@pytest.fixture
def id_to_predicted() -> Mapping[IdentifierType, str]:
    return {0: "positive", 1: "negative", 2: "negative"}


@pytest.fixture
def id_to_probs_pos() -> Mapping[IdentifierType, float]:
    return {0: 0.9, 1: 0.2, 2: 0.4}


SCORE_TEST_CASES = [
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
]


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", SCORE_TEST_CASES)
def test_produce_score(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationScoreType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result = 0.85
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get = mocker.spy(obj=BinaryClassificationScoreRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_score(score_type=artifact_type, resources=resources)

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    assert result == fake_result


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", SCORE_TEST_CASES)
def test_produce_classification_score(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationScoreType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result = 0.85
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationScoreRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_classification_score(
        score_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result == fake_result


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", SCORE_TEST_CASES)
def test_produce_binary_classification_score(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    id_to_true: Mapping[IdentifierType, str],
    id_to_predicted: Mapping[IdentifierType, str],
    id_to_probs_pos: Mapping[IdentifierType, float],
    artifact_type: BinaryClassificationScoreType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result = 0.85
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationScoreRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_binary_classification_score(
        score_type=artifact_type,
        true=id_to_true,
        predicted=id_to_predicted,
        probs_pos=id_to_probs_pos,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result == fake_result


ARRAY_TEST_CASES = [
    (BinaryClassificationArrayType.CONFUSION_MATRIX, ConfusionMatrix),
]


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", ARRAY_TEST_CASES)
def test_produce_array(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationArrayType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result = np.array([[1, 0], [0, 1]])
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get = mocker.spy(obj=BinaryClassificationArrayRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_array(array_type=artifact_type, resources=resources)

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    np.testing.assert_array_equal(result, fake_result)


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", ARRAY_TEST_CASES)
def test_produce_classification_array(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationArrayType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result = np.array([[1, 0], [0, 1]])
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationArrayRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_classification_array(
        array_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    np.testing.assert_array_equal(result, fake_result)


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", ARRAY_TEST_CASES)
def test_produce_binary_classification_array(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    id_to_true: Mapping[IdentifierType, str],
    id_to_predicted: Mapping[IdentifierType, str],
    id_to_probs_pos: Mapping[IdentifierType, float],
    artifact_type: BinaryClassificationArrayType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result = np.array([[1, 0], [0, 1]])
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationArrayRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_binary_classification_array(
        array_type=artifact_type,
        true=id_to_true,
        predicted=id_to_predicted,
        probs_pos=id_to_probs_pos,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    np.testing.assert_array_equal(result, fake_result)


PLOT_TEST_CASES = [
    (BinaryClassificationPlotType.CONFUSION_MATRIX_PLOT, ConfusionMatrixPlot),
    (BinaryClassificationPlotType.ROC_CURVE, ROCCurve),
    (BinaryClassificationPlotType.PR_CURVE, PRCurve),
    (BinaryClassificationPlotType.DET_CURVE, DETCurve),
    (BinaryClassificationPlotType.RECALL_THRESHOLD_CURVE, RecallThresholdCurve),
    (BinaryClassificationPlotType.PRECISION_THRESHOLD_CURVE, PrecisionThresholdCurve),
    (BinaryClassificationPlotType.SCORE_PDF, ScoreDistributionPlot),
    (BinaryClassificationPlotType.GROUND_TRUTH_PROB_PDF, GroundTruthProbPDFPlot),
]


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", PLOT_TEST_CASES)
def test_produce_plot(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationPlotType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result = Figure()
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get = mocker.spy(obj=BinaryClassificationPlotRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_plot(plot_type=artifact_type, resources=resources)

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    assert result is fake_result


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", PLOT_TEST_CASES)
def test_produce_classification_plot(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationPlotType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result = Figure()
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationPlotRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_classification_plot(
        plot_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result is fake_result


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", PLOT_TEST_CASES)
def test_produce_binary_classification_plot(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    id_to_true: Mapping[IdentifierType, str],
    id_to_predicted: Mapping[IdentifierType, str],
    id_to_probs_pos: Mapping[IdentifierType, float],
    artifact_type: BinaryClassificationPlotType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result = Figure()
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationPlotRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_binary_classification_plot(
        plot_type=artifact_type,
        true=id_to_true,
        predicted=id_to_predicted,
        probs_pos=id_to_probs_pos,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result is fake_result


SCORE_COLLECTION_TEST_CASES = [
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
    (
        BinaryClassificationScoreCollectionType.GROUND_TRUTH_PROB_STATS,
        GroundTruthProbStats,
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", SCORE_COLLECTION_TEST_CASES)
def test_produce_score_collection(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationScoreCollectionType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result: Any = {"tp": 0.5, "tn": 0.3}
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get = mocker.spy(obj=BinaryClassificationScoreCollectionRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_score_collection(
        score_collection_type=artifact_type, resources=resources
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    assert result == fake_result


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", SCORE_COLLECTION_TEST_CASES)
def test_produce_classification_score_collection(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationScoreCollectionType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result: Any = {"tp": 0.5, "tn": 0.3}
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationScoreCollectionRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_classification_score_collection(
        score_collection_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result == fake_result


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", SCORE_COLLECTION_TEST_CASES)
def test_produce_binary_classification_score_collection(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    id_to_true: Mapping[IdentifierType, str],
    id_to_predicted: Mapping[IdentifierType, str],
    id_to_probs_pos: Mapping[IdentifierType, float],
    artifact_type: BinaryClassificationScoreCollectionType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result: Any = {"tp": 0.5, "tn": 0.3}
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationScoreCollectionRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_binary_classification_score_collection(
        score_collection_type=artifact_type,
        true=id_to_true,
        predicted=id_to_predicted,
        probs_pos=id_to_probs_pos,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result == fake_result


ARRAY_COLLECTION_TEST_CASES = [
    (
        BinaryClassificationArrayCollectionType.CONFUSION_MATRICES,
        ConfusionMatrixCollection,
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", ARRAY_COLLECTION_TEST_CASES)
def test_produce_array_collection(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationArrayCollectionType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result: Any = {"all": np.array([[1, 0], [0, 1]])}
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get = mocker.spy(obj=BinaryClassificationArrayCollectionRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_array_collection(
        array_collection_type=artifact_type, resources=resources
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    for key in fake_result:
        np.testing.assert_array_equal(result[key], fake_result[key])


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", ARRAY_COLLECTION_TEST_CASES)
def test_produce_classification_array_collection(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationArrayCollectionType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result: Any = {"all": np.array([[1, 0], [0, 1]])}
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationArrayCollectionRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_classification_array_collection(
        array_collection_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    for key in fake_result:
        np.testing.assert_array_equal(result[key], fake_result[key])


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", ARRAY_COLLECTION_TEST_CASES)
def test_produce_binary_classification_array_collection(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    id_to_true: Mapping[IdentifierType, str],
    id_to_predicted: Mapping[IdentifierType, str],
    id_to_probs_pos: Mapping[IdentifierType, float],
    artifact_type: BinaryClassificationArrayCollectionType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result: Any = {"all": np.array([[1, 0], [0, 1]])}
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationArrayCollectionRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_binary_classification_array_collection(
        array_collection_type=artifact_type,
        true=id_to_true,
        predicted=id_to_predicted,
        probs_pos=id_to_probs_pos,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    for key in fake_result:
        np.testing.assert_array_equal(result[key], fake_result[key])


PLOT_COLLECTION_TEST_CASES = [
    (
        BinaryClassificationPlotCollectionType.CONFUSION_MATRIX_PLOTS,
        ConfusionMatrixPlotCollection,
    ),
    (
        BinaryClassificationPlotCollectionType.THRESHOLD_VARIATION_CURVES,
        ThresholdVariationCurves,
    ),
    (BinaryClassificationPlotCollectionType.SCORE_PDF_PLOTS, ScoreDistributionPlots),
]


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", PLOT_COLLECTION_TEST_CASES)
def test_produce_plot_collection(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationPlotCollectionType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result: Any = {"all": Figure()}
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get = mocker.spy(obj=BinaryClassificationPlotCollectionRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_plot_collection(plot_collection_type=artifact_type, resources=resources)

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    assert result == fake_result


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", PLOT_COLLECTION_TEST_CASES)
def test_produce_classification_plot_collection(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    artifact_type: BinaryClassificationPlotCollectionType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result: Any = {"all": Figure()}
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationPlotCollectionRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_classification_plot_collection(
        plot_collection_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result == fake_result


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", PLOT_COLLECTION_TEST_CASES)
def test_produce_binary_classification_plot_collection(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    id_to_true: Mapping[IdentifierType, str],
    id_to_predicted: Mapping[IdentifierType, str],
    id_to_probs_pos: Mapping[IdentifierType, float],
    artifact_type: BinaryClassificationPlotCollectionType,
    artifact_class: Type[BinaryClassificationArtifact],
):
    fake_result: Any = {"all": Figure()}
    engine = BinaryClassificationEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(obj=BinaryClassificationPlotCollectionRegistry, name="get")
    spy_compute = mocker.patch.object(
        target=artifact_class, attribute="compute", return_value=fake_result
    )

    result = engine.produce_binary_classification_plot_collection(
        plot_collection_type=artifact_type,
        true=id_to_true,
        predicted=id_to_predicted,
        probs_pos=id_to_probs_pos,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result == fake_result
