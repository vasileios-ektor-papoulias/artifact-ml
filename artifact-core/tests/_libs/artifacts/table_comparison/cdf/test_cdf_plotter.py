from typing import List

import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.cdf.plotter import (
    TabularCDFPlotter,
)
from artifact_core._libs.tools.plotters.cdf_plotter import CDFPlotter
from artifact_core._libs.tools.plotters.plot_combiner import PlotCombiner
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, cts_features, expected_plot_count",
    [
        ("df_simple", ["cts_1", "cts_2"], 2),
        ("df_simple", ["cts_1"], 1),
        ("df_simple", [], 0),
    ],
    indirect=["df_dispatcher"],
)
def test_get_cdf_plot_collection(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    df_dispatcher: pd.DataFrame,
    cts_features: List[str],
    expected_plot_count: int,
):
    spy = mocker.spy(obj=CDFPlotter, name="plot_cdf")
    result = TabularCDFPlotter.get_cdf_plot_collection(
        dataset=df_dispatcher, cts_features=cts_features
    )
    assert isinstance(result, dict)
    assert len(result) == expected_plot_count
    assert spy.call_count == expected_plot_count
    for i, (feature, fig) in enumerate(result.items()):
        assert isinstance(fig, Figure)
        assert fig.get_axes()
        title_text = fig.texts[0].get_text() if fig.texts else ""
        assert title_text == f"CDF: {feature}"
        assert spy.call_args_list[i].kwargs["feature_name"] == feature


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, cts_features, expected_axes_count",
    [
        ("df_simple", ["cts_1", "cts_2"], 3),
        ("df_simple", ["cts_1"], 3),
        ("df_simple", [], 0),
    ],
    indirect=["df_dispatcher"],
)
def test_get_cdf_plot(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    df_dispatcher: pd.DataFrame,
    cts_features: List[str],
    expected_axes_count: int,
):
    spy_combiner = mocker.spy(obj=PlotCombiner, name="combine")
    result = TabularCDFPlotter.get_cdf_plot(dataset=df_dispatcher, cts_features=cts_features)
    assert isinstance(result, Figure)
    spy_combiner.assert_called_once()
    assert len(spy_combiner.call_args.kwargs["plots"]) == len(cts_features)
    if expected_axes_count > 0:
        assert len(result.axes) == expected_axes_count
        assert result.texts
        assert "Cumulative Density Functions" in result.texts[0].get_text()
    else:
        assert len(result.axes) == 0
