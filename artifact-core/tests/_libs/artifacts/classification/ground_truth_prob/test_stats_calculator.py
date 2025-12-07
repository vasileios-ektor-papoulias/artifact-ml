from typing import Dict, List

import numpy as np
import pytest
from artifact_core._libs.artifacts.classification.ground_truth_prob.calculator import (  # noqa: E501
    GroundTruthProbCalculator,
)
from artifact_core._libs.artifacts.classification.ground_truth_prob.stats_calculator import (  # noqa: E501
    GroundTruthProbStatsCalculator,
)
from artifact_core._libs.resources.classification.class_store import (  # noqa: E501
    ClassStore,
)
from artifact_core._libs.resources.classification.classification_results import (  # noqa: E501
    ClassificationResults,
)
from artifact_core._libs.tools.calculators.descriptive_stats_calculator import (  # noqa: E501
    DescriptiveStatistic,
    DescriptiveStatsCalculator,
)
from artifact_core._utils.collections.entity_store import IdentifierType
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_true_class, id_to_logits, stat",
    [
        (
            {0: "A", 1: "B", 2: "C"},
            {
                0: np.array([2.0, 0.5, 0.1]),
                1: np.array([0.1, 2.5, 0.2]),
                2: np.array([0.2, 0.1, 3.0]),
            },
            DescriptiveStatistic.MEAN,
        ),
        (
            {0: "A", 1: "B"},
            {
                0: np.array([1.0, 0.5, 0.5]),
                1: np.array([0.5, 1.0, 0.5]),
            },
            DescriptiveStatistic.STD,
        ),
        (
            {0: "A"},
            {0: np.array([2.0, 0.5, 0.1])},
            DescriptiveStatistic.MEDIAN,
        ),
        (
            {0: "A", 1: "B", 2: "C"},
            {
                0: np.array([2.0, 0.5, 0.1]),
                1: np.array([0.1, 2.5, 0.2]),
                2: np.array([0.2, 0.1, 3.0]),
            },
            DescriptiveStatistic.MIN,
        ),
        (
            {0: "A", 1: "B", 2: "C"},
            {
                0: np.array([2.0, 0.5, 0.1]),
                1: np.array([0.1, 2.5, 0.2]),
                2: np.array([0.2, 0.1, 3.0]),
            },
            DescriptiveStatistic.MAX,
        ),
    ],
)
def test_compute(
    mocker: MockerFixture,
    true_class_store: ClassStore,
    classification_results: ClassificationResults,
    id_to_true_class: Dict[IdentifierType, str],
    id_to_logits: Dict[IdentifierType, np.ndarray],
    stat: DescriptiveStatistic,
):
    for identifier, true_class in id_to_true_class.items():
        true_class_store.set_class(identifier=identifier, class_name=true_class)  # noqa: E501
        classification_results.set_single(
            identifier=identifier,
            predicted_class=true_class,
            logits=id_to_logits[identifier],
        )
    spy_calculator = mocker.spy(  # noqa: E501
        obj=GroundTruthProbCalculator, name="compute_id_to_prob_ground_truth"
    )
    spy_stats = mocker.spy(  # noqa: E501
        obj=DescriptiveStatsCalculator, name="compute_stat"
    )
    result = GroundTruthProbStatsCalculator.compute(
        classification_results=classification_results,
        true_class_store=true_class_store,
        stat=stat,
    )
    assert isinstance(result, float)
    spy_calculator.assert_called_once()
    spy_stats.assert_called_once()
    assert spy_stats.call_args.kwargs["stat"] == stat


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_true_class, id_to_logits, stats",
    [
        (
            {0: "A", 1: "B", 2: "C"},
            {
                0: np.array([2.0, 0.5, 0.1]),
                1: np.array([0.1, 2.5, 0.2]),
                2: np.array([0.2, 0.1, 3.0]),
            },
            [DescriptiveStatistic.MEAN],
        ),
        (
            {0: "A", 1: "B"},
            {
                0: np.array([1.0, 0.5, 0.5]),
                1: np.array([0.5, 1.0, 0.5]),
            },
            [DescriptiveStatistic.MEAN, DescriptiveStatistic.STD],
        ),
        (
            {0: "A"},
            {0: np.array([2.0, 0.5, 0.1])},
            [DescriptiveStatistic.MIN, DescriptiveStatistic.MAX, DescriptiveStatistic.MEDIAN],  # noqa: E501
        ),
    ],
)
def test_compute_multiple(
    mocker: MockerFixture,
    true_class_store: ClassStore,
    classification_results: ClassificationResults,
    id_to_true_class: Dict[IdentifierType, str],
    id_to_logits: Dict[IdentifierType, np.ndarray],
    stats: List[DescriptiveStatistic],
):
    for identifier, true_class in id_to_true_class.items():
        true_class_store.set_class(identifier=identifier, class_name=true_class)  # noqa: E501
        classification_results.set_single(
            identifier=identifier,
            predicted_class=true_class,
            logits=id_to_logits[identifier],
        )
    spy_calculator = mocker.spy(  # noqa: E501
        obj=GroundTruthProbCalculator, name="compute_id_to_prob_ground_truth"
    )
    spy_stats = mocker.spy(  # noqa: E501
        obj=DescriptiveStatsCalculator, name="compute_dict_stats"
    )
    result = GroundTruthProbStatsCalculator.compute_multiple(
        classification_results=classification_results,
        true_class_store=true_class_store,
        stats=stats,
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == set(stats)
    spy_calculator.assert_called_once()
    spy_stats.assert_called_once()
    assert list(spy_stats.call_args.kwargs["stats"]) == stats
    for stat, value in result.items():
        assert isinstance(value, float)


@pytest.mark.unit
def test_compute_multiple_empty_stats(
    true_class_store: ClassStore,
    classification_results: ClassificationResults,
):
    true_class_store.set_class(identifier=0, class_name="A")
    classification_results.set_single(
        identifier=0,
        predicted_class="A",
        logits=np.array([2.0, 0.5, 0.1]),
    )
    result = GroundTruthProbStatsCalculator.compute_multiple(
        classification_results=classification_results,
        true_class_store=true_class_store,
        stats=[],
    )
    assert result == {}
