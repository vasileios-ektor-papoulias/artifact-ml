from typing import List, Tuple

import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.cdf.overlaid_plotter import (  # noqa: E501
    TabularOverlaidCDFPlotter,
)
from artifact_core._libs.tools.plotters.overlaid_cdf_plotter import (  # noqa: E501
    OverlaidCDFPlotter,
)
from artifact_core._libs.tools.plotters.plot_combiner import PlotCombiner
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, cts_features, expected_plot_count",
    [
        (("df_small_real", "df_small_synthetic"), ["cts_1", "cts_2"], 2),
        (("df_small_real", "df_small_synthetic"), ["cts_1"], 1),
        (("df_small_real", "df_small_synthetic"), [], 0),
    ],
    indirect=["df_pair_dispatcher"],
)
def test_get_overlaid_cdf_plot_collection(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    cts_features: List[str],
    expected_plot_count: int,
):
    spy = mocker.spy(obj=OverlaidCDFPlotter, name="plot_overlaid_cdf")
    df_real, df_synthetic = df_pair_dispatcher

    result = TabularOverlaidCDFPlotter.get_overlaid_cdf_plot_collection(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        cts_features=cts_features,
    )

    assert isinstance(result, dict)
    assert len(result) == expected_plot_count
    assert spy.call_count == expected_plot_count

    for i, (feature, fig) in enumerate(result.items()):
        assert isinstance(fig, Figure)
        assert fig.get_axes()
        title_text = fig.texts[0].get_text() if fig.texts else ""
        assert title_text == f"CDF Comparison: {feature}"
        assert spy.call_args_list[i].kwargs["feature_name"] == feature


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, cts_features, expected_axes_count",
    [
        (("df_small_real", "df_small_synthetic"), ["cts_1", "cts_2"], 3),
        (("df_small_real", "df_small_synthetic"), ["cts_1"], 3),
        (("df_small_real", "df_small_synthetic"), [], 0),
    ],
    indirect=["df_pair_dispatcher"],
)
def test_get_overlaid_cdf_plot(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    cts_features: List[str],
    expected_axes_count: int,
):
    spy_combiner = mocker.spy(obj=PlotCombiner, name="combine")
    df_real, df_synthetic = df_pair_dispatcher

    result = TabularOverlaidCDFPlotter.get_overlaid_cdf_plot(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        cts_features=cts_features,
    )

    assert isinstance(result, Figure)
    spy_combiner.assert_called_once()
    assert len(spy_combiner.call_args.kwargs["plots"]) == len(cts_features)

    if expected_axes_count > 0:
        assert len(result.axes) == expected_axes_count
        assert result.texts
        title = result.texts[0].get_text()
        assert "Cumulative Density Function Comparison" in title
    else:
        assert len(result.axes) == 0
