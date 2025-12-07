from typing import List

import pytest
from artifact_core._libs.artifacts.binary_classification.score_distribution.partitioner import (
    BinarySampleSplit,
)
from artifact_core._libs.artifacts.binary_classification.score_distribution.plotter import (
    ScoreDistributionPlotterConfig,
    ScorePDFPlotter,
)
from artifact_core._libs.artifacts.binary_classification.score_distribution.sampler import (
    ScoreDistributionSampler,
)
from artifact_core._libs.tools.plotters.overlaid_pdf_plotter import OverlaidPDFPlotter
from artifact_core._libs.tools.plotters.pdf_plotter import PDFPlotter
from matplotlib.figure import Figure
from pytest_mock import MockerFixture

from tests._libs.artifacts.binary_classification.conftest import BinaryDataTuple


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, split",
    [
        ("binary_data_balanced", BinarySampleSplit.ALL),
        ("binary_data_balanced", BinarySampleSplit.POSITIVE),
        ("binary_data_balanced", BinarySampleSplit.NEGATIVE),
        ("binary_data_imbalanced", BinarySampleSplit.ALL),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_plot(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    binary_data_dispatcher: BinaryDataTuple,
    split: BinarySampleSplit,
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    spy_sampler = mocker.spy(obj=ScoreDistributionSampler, name="get_sample")
    spy_pdf = mocker.spy(obj=PDFPlotter, name="plot_pdf")
    result = ScorePDFPlotter.plot(
        id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos, split=split
    )
    assert isinstance(result, Figure)
    spy_sampler.assert_called_once()
    spy_pdf.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    "splits",
    [
        [BinarySampleSplit.ALL],
        [BinarySampleSplit.POSITIVE, BinarySampleSplit.NEGATIVE],
        list(BinarySampleSplit),
    ],
)
@pytest.mark.parametrize("binary_data_dispatcher", ["binary_data_balanced"], indirect=True)
def test_plot_multiple(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    binary_data_dispatcher: BinaryDataTuple,
    splits: List[BinarySampleSplit],
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    spy_pdf = mocker.spy(obj=PDFPlotter, name="plot_pdf")
    result = ScorePDFPlotter.plot_multiple(
        id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos, splits=splits
    )
    assert set(result.keys()) == set(splits)
    for split, fig in result.items():
        assert isinstance(fig, Figure)
    assert spy_pdf.call_count == len(splits)


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher", ["binary_data_balanced", "binary_data_imbalanced"], indirect=True
)
def test_plot_overlaid(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    binary_data_dispatcher: BinaryDataTuple,
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    spy_sampler = mocker.spy(obj=ScoreDistributionSampler, name="get_sample")
    spy_overlaid = mocker.spy(obj=OverlaidPDFPlotter, name="plot_overlaid_pdf")
    result = ScorePDFPlotter.plot_overlaid(id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos)
    assert isinstance(result, Figure)
    assert spy_sampler.call_count == 2
    spy_overlaid.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("binary_data_dispatcher", ["binary_data_balanced"], indirect=True)
def test_plot_with_custom_config(
    set_agg_backend,
    close_all_figs_after_test,
    binary_data_dispatcher: BinaryDataTuple,
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    config = ScoreDistributionPlotterConfig(
        prob_col_name="Custom Prob",
        label_positive="Custom Pos",
        label_negative="Custom Neg",
    )
    result = ScorePDFPlotter.plot(
        id_to_is_pos=id_to_is_pos,
        id_to_prob_pos=id_to_prob_pos,
        split=BinarySampleSplit.ALL,
        config=config,
    )
    assert isinstance(result, Figure)
