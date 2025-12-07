from typing import List

import pytest
from artifact_core._libs.artifacts.binary_classification.confusion.calculator import (
    NormalizedConfusionCalculator,
)
from artifact_core._libs.artifacts.binary_classification.confusion.normalizer import (
    ConfusionMatrixNormalizationStrategy,
)
from artifact_core._libs.artifacts.binary_classification.confusion.plotter import (
    ConfusionMatrixPlotConfig,
    ConfusionMatrixPlotter,
)
from artifact_core._libs.artifacts.binary_classification.confusion.raw import RawConfusionCalculator
from matplotlib.figure import Figure
from pytest_mock import MockerFixture

from tests._libs.artifacts.binary_classification.conftest import BinaryDataTuple


@pytest.mark.unit
@pytest.mark.parametrize("normalization", list(ConfusionMatrixNormalizationStrategy))
@pytest.mark.parametrize(
    "binary_data_dispatcher",
    ["binary_data_balanced", "binary_data_imbalanced"],
    indirect=True,
)
def test_plot(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    binary_data_dispatcher: BinaryDataTuple,
    normalization: ConfusionMatrixNormalizationStrategy,
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    spy_raw = mocker.spy(obj=RawConfusionCalculator, name="compute_confusion_matrix")
    spy_normalized = mocker.spy(
        obj=NormalizedConfusionCalculator, name="compute_normalized_confusion_matrix"
    )
    result = ConfusionMatrixPlotter.plot(
        true=id_to_is_pos, predicted=id_to_pred_pos, normalization=normalization
    )
    assert isinstance(result, Figure)
    spy_raw.assert_called_once()
    spy_normalized.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    "normalization_types",
    [
        [ConfusionMatrixNormalizationStrategy.TRUE],
        [ConfusionMatrixNormalizationStrategy.TRUE, ConfusionMatrixNormalizationStrategy.ALL],
        list(ConfusionMatrixNormalizationStrategy),
    ],
)
@pytest.mark.parametrize(
    "binary_data_dispatcher",
    ["binary_data_balanced"],
    indirect=True,
)
def test_plot_multiple(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    binary_data_dispatcher: BinaryDataTuple,
    normalization_types: List[ConfusionMatrixNormalizationStrategy],
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    spy_normalized = mocker.spy(
        obj=NormalizedConfusionCalculator, name="compute_normalized_confusion_matrix"
    )
    result = ConfusionMatrixPlotter.plot_multiple(
        true=id_to_is_pos, predicted=id_to_pred_pos, normalization_types=normalization_types
    )
    assert set(result.keys()) == set(normalization_types)
    for _, fig in result.items():
        assert isinstance(fig, Figure)
    assert spy_normalized.call_count == len(normalization_types)


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher",
    ["binary_data_balanced"],
    indirect=True,
)
def test_plot_with_custom_config(
    set_agg_backend, close_all_figs_after_test, binary_data_dispatcher: BinaryDataTuple
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    config = ConfusionMatrixPlotConfig(
        title="Custom Title",
        dpi=100,
        show_values=False,
    )
    result = ConfusionMatrixPlotter.plot(
        true=id_to_is_pos,
        predicted=id_to_pred_pos,
        normalization=ConfusionMatrixNormalizationStrategy.NONE,
        config=config,
    )
    assert isinstance(result, Figure)
