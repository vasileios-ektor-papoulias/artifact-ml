from typing import List

import pytest
from artifact_core._libs.artifacts.binary_classification.threshold_variation.plotter import (
    ThresholdVariationCurvePlotter,
    ThresholdVariationCurvePlotterConfig,
    ThresholdVariationCurveType,
)
from artifact_core._utils.collections.map_aligner import MapAligner
from matplotlib.figure import Figure
from pytest_mock import MockerFixture

from tests._libs.artifacts.binary_classification.conftest import BinaryDataTuple


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, curve_type",
    [
        ("binary_data_balanced", ThresholdVariationCurveType.ROC),
        ("binary_data_balanced", ThresholdVariationCurveType.PR),
        ("binary_data_balanced", ThresholdVariationCurveType.DET),
        ("binary_data_balanced", ThresholdVariationCurveType.RECALL_THRESHOLD),
        ("binary_data_balanced", ThresholdVariationCurveType.PRECISION_THRESHOLD),
        ("binary_data_imbalanced", ThresholdVariationCurveType.ROC),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_plot(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    binary_data_dispatcher: BinaryDataTuple,
    curve_type: ThresholdVariationCurveType,
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    spy_aligner = mocker.spy(obj=MapAligner, name="align")
    result = ThresholdVariationCurvePlotter.plot(
        curve_type=curve_type, true=id_to_is_pos, probs=id_to_prob_pos
    )
    assert isinstance(result, Figure)
    spy_aligner.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, curve_types",
    [
        ("binary_data_balanced", [ThresholdVariationCurveType.ROC]),
        ("binary_data_balanced", [ThresholdVariationCurveType.ROC, ThresholdVariationCurveType.PR]),
        ("binary_data_balanced", list(ThresholdVariationCurveType)),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_plot_multiple(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    binary_data_dispatcher: BinaryDataTuple,
    curve_types: List[ThresholdVariationCurveType],
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    spy_plot = mocker.spy(obj=ThresholdVariationCurvePlotter, name="plot")
    result = ThresholdVariationCurvePlotter.plot_multiple(
        curve_types=curve_types, true=id_to_is_pos, probs=id_to_prob_pos
    )
    assert set(result.keys()) == set(curve_types)
    for fig in result.values():
        assert isinstance(fig, Figure)
    assert spy_plot.call_count == len(curve_types)


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, curve_type",
    [
        ("binary_data_all_positive", ThresholdVariationCurveType.ROC),
        ("binary_data_all_positive", ThresholdVariationCurveType.DET),
        ("binary_data_all_negative", ThresholdVariationCurveType.ROC),
        ("binary_data_all_negative", ThresholdVariationCurveType.RECALL_THRESHOLD),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_plot_single_class_returns_empty_fig(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    binary_data_dispatcher: BinaryDataTuple,
    curve_type: ThresholdVariationCurveType,
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    spy_empty_fig = mocker.spy(obj=ThresholdVariationCurvePlotter, name="_empty_fig")
    result = ThresholdVariationCurvePlotter.plot(
        curve_type=curve_type, true=id_to_is_pos, probs=id_to_prob_pos
    )
    assert isinstance(result, Figure)
    spy_empty_fig.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("binary_data_dispatcher", ["binary_data_balanced"], indirect=True)
def test_plot_with_custom_config(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    binary_data_dispatcher: BinaryDataTuple,
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    config = ThresholdVariationCurvePlotterConfig(
        dpi=100, linewidth=3.0, color="red", show_baseline=False
    )
    spy_plot_roc = mocker.spy(obj=ThresholdVariationCurvePlotter, name="_plot_roc")
    result = ThresholdVariationCurvePlotter.plot(
        curve_type=ThresholdVariationCurveType.ROC,
        true=id_to_is_pos,
        probs=id_to_prob_pos,
        config=config,
    )
    assert isinstance(result, Figure)
    spy_plot_roc.assert_called_once()
