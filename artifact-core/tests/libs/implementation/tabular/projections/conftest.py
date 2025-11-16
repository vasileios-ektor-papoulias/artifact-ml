from typing import List, Literal, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.projections.base.plotter import (
    ProjectionPlotter,
    ProjectionPlotterConfig,
)
from artifact_core._libs.artifacts.tools.plotters.plot_combiner import PlotCombinationConfig
from pytest_mock import MockerFixture


@pytest.fixture
def dataset_small() -> Tuple[pd.DataFrame, List[str], List[str]]:
    np.random.seed(0)
    return (
        pd.DataFrame(np.random.rand(10, 3), columns=["cts_1", "cts_2", "cts_3"]),
        [],
        ["cts_1", "cts_2", "cts_3"],
    )


@pytest.fixture
def dataset_large() -> Tuple[pd.DataFrame, List[str], List[str]]:
    np.random.seed(1)
    return (
        pd.DataFrame(np.random.rand(50, 5), columns=["cts_1", "cts_2", "cts_3", "cts_4", "cts_5"]),
        [],
        ["cts_1", "cts_2", "cts_3", "cts_4", "cts_5"],
    )


@pytest.fixture
def dataset_mixed() -> Tuple[pd.DataFrame, List[str], List[str]]:
    np.random.seed(2)
    return (
        pd.DataFrame(
            {
                "cts_1": np.random.rand(20),
                "cts_2": np.random.randn(20) * 10 + 5,
                "cat_1": np.random.choice(["A", "B", "C"], size=20),
                "cat_2": np.random.choice(["X", "Y"], size=20),
            },
        ),
        ["cat_1", "cat_2"],
        ["cts_1", "cts_2"],
    )


@pytest.fixture
def dataset_dispatcher(request: pytest.FixtureRequest) -> Tuple[pd.DataFrame, List[str], List[str]]:
    ls_dataset_fixtures = ["dataset_small", "dataset_large", "dataset_mixed"]
    if request.param not in ls_dataset_fixtures:
        raise ValueError(
            f"Data fixture not found: fixture param should be one of: {ls_dataset_fixtures}"
        )
    return request.getfixturevalue(request.param)


@pytest.fixture
def projection_2d_real_dispatcher(
    request: pytest.FixtureRequest,
) -> Optional[Array]:
    projection_type: Literal["projection_2d_real", "null"] = request.param
    if projection_type == "null":
        projection = None
    else:
        projection = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    return projection


@pytest.fixture
def projection_2d_synthetic_dispatcher(
    request: pytest.FixtureRequest,
) -> Optional[Array]:
    projection_type: Literal["projection_2d_real", "null"] = request.param
    if projection_type == "null":
        projection = None
    else:
        projection = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5], [9.5, 10.5]])
    return projection


@pytest.fixture
def mock_plotter(
    mocker: MockerFixture,
) -> MagicMock:
    mock_plotter = mocker.Mock(spec=ProjectionPlotter)
    mock_plotter.produce_projection_plot.return_value = mocker.Mock(spec=Figure)
    mock_plotter.produce_projection_comparison_plot.return_value = mocker.Mock(spec=Figure)
    return mock_plotter


@pytest.fixture
def default_projection_plotter() -> ProjectionPlotter:
    plotter = ProjectionPlotter()
    return plotter


@pytest.fixture
def custom_projection_plotter() -> ProjectionPlotter:
    config = ProjectionPlotterConfig(
        scatter_color="blue",
        failed_suffix="Custom failed message",
        figsize=(8, 8),
        title_prefix="Custom Projection",
        combined_config=PlotCombinationConfig(
            n_cols=1,
            dpi=100,
            combined_title="Custom Combined Title",
        ),
    )
    plotter = ProjectionPlotter(config=config)
    return plotter


@pytest.fixture
def projection_plotter_dispatcher(request: pytest.FixtureRequest) -> ProjectionPlotter:
    ls_projection_plotter_fixture_names = [
        "default_projection_plotter",
        "custom_projection_plotter",
    ]
    if request.param not in ls_projection_plotter_fixture_names:
        raise ValueError(
            f"Projection plotter fixture {request.param} not found: "
            + f"fixture param should be one of: {ls_projection_plotter_fixture_names}"
        )
    return request.getfixturevalue(request.param)
