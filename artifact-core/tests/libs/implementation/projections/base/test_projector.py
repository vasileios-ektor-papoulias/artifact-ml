from typing import Callable, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from artifact_core.libs.implementation.projections.base.plotter import ProjectionPlotter
from matplotlib.figure import Figure
from pandas import DataFrame
from pytest_mock import MockerFixture

from tests.libs.implementation.projections.base.dummy.projector import (
    DummyProjector,
    DummyProjectorHyperparams,
)


@pytest.fixture
def mock_plotter_factory(
    mocker: MockerFixture,
) -> Callable[[], MagicMock]:
    def factory() -> MagicMock:
        mock_plotter = mocker.Mock(spec=ProjectionPlotter)
        mock_plotter.produce_projection_plot.return_value = mocker.Mock(spec=Figure)
        mock_plotter.produce_projection_comparison_plot.return_value = mocker.Mock(spec=Figure)
        return mock_plotter

    return factory


@pytest.fixture
def dataset_small() -> Tuple[DataFrame, List[str], List[str]]:
    np.random.seed(0)
    return pd.DataFrame(np.random.rand(10, 3), columns=["c1", "c2", "c3"]), [], ["c1", "c2", "c3"]


@pytest.fixture
def dataset_large() -> Tuple[DataFrame, List[str], List[str]]:
    np.random.seed(1)
    return (
        pd.DataFrame(np.random.rand(50, 5), columns=["c1", "c2", "c3", "c4", "c5"]),
        [],
        ["c1", "c2", "c3", "c4", "c5"],
    )


@pytest.fixture
def dataset_mixed() -> Tuple[DataFrame, List[str], List[str]]:
    np.random.seed(2)
    return (
        pd.DataFrame(
            {
                "num1": np.random.rand(20),
                "num2": np.random.randn(20) * 10 + 5,
                "cat1": np.random.choice(["A", "B", "C"], size=20),
                "cat2": np.random.choice(["X", "Y"], size=20),
            },
        ),
        ["cat1", "cat2"],
        ["num1", "num2"],
    )


@pytest.fixture
def dataset_dispatcher(request: pytest.FixtureRequest) -> Tuple[DataFrame, List[str], List[str]]:
    if request.param not in ["dataset_small", "dataset_large", "dataset_mixed"]:
        raise ValueError(
            "Data fixture not found: fixture param should be one of: "
            + "'dataset_small', 'dataset_large', 'dataset_mixed'."
        )
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "ls_cat_features, ls_cts_features, hyperparams, "
    + "expected_projection_type, expected_exception",
    [
        (["c1"], [], DummyProjectorHyperparams(projection_type="random"), "random", None),
        ([], ["c1"], DummyProjectorHyperparams(projection_type="random"), "random", None),
        (["c1"], ["c2"], DummyProjectorHyperparams(projection_type="random"), "random", None),
        (
            ["c1", "c2"],
            ["c1"],
            DummyProjectorHyperparams(projection_type="random"),
            "random",
            ValueError("Categorical and continuous features overlap: {'c1'}"),
        ),
        (
            ["c1"],
            ["c1"],
            DummyProjectorHyperparams(projection_type="random"),
            "random",
            ValueError("Categorical and continuous features overlap: {'c1'}"),
        ),
        (["c1", "c2"], ["c3"], DummyProjectorHyperparams(projection_type="random"), "random", None),
        (
            [],
            [],
            DummyProjectorHyperparams(projection_type="random"),
            "random",
            ValueError("Both categorical and continuous feature lists are empty."),
        ),
        (["c1"], [], DummyProjectorHyperparams(projection_type="invalid"), "invalid", None),
        ([], ["c1"], DummyProjectorHyperparams(projection_type="invalid"), "invalid", None),
        (["c1"], ["c2"], DummyProjectorHyperparams(projection_type="invalid"), "invalid", None),
        (
            ["c1"],
            ["c1"],
            DummyProjectorHyperparams(projection_type="invalid"),
            "invalid",
            ValueError("Categorical and continuous features overlap: {'c1'}"),
        ),
        (
            [],
            [],
            DummyProjectorHyperparams(projection_type="invalid"),
            "invalid",
            ValueError("Both categorical and continuous feature lists are empty."),
        ),
    ],
)
def test_build(
    ls_cat_features: list[str],
    ls_cts_features: list[str],
    hyperparams: DummyProjectorHyperparams,
    expected_projection_type: str,
    expected_exception: Optional[BaseException],
):
    if expected_exception:
        expected_exception_type = type(expected_exception)
        expected_exception_message = expected_exception.args[0]
        with pytest.raises(
            expected_exception_type,
            match=expected_exception_message,
        ):
            _ = DummyProjector.build(
                ls_cat_features=ls_cat_features,
                ls_cts_features=ls_cts_features,
                projector_config=hyperparams,
            )
    else:
        projector = DummyProjector.build(
            ls_cat_features=ls_cat_features,
            ls_cts_features=ls_cts_features,
            projector_config=hyperparams,
        )
        assert projector.projection_name == "dummy_projection"
        assert isinstance(projector._hyperparams, DummyProjectorHyperparams)
        assert projector._hyperparams.projection_type == expected_projection_type
        assert isinstance(projector._plotter, ProjectionPlotter)


@pytest.mark.parametrize(
    "hyperparams, dataset_dispatcher, expected_shape, expected_exception",
    [
        (
            DummyProjectorHyperparams(projection_type="random"),
            "dataset_small",
            (10, 3),
            None,
        ),
        (
            DummyProjectorHyperparams(projection_type="random"),
            "dataset_large",
            (50, 5),
            None,
        ),
        (
            DummyProjectorHyperparams(projection_type="invalid"),
            "dataset_small",
            None,
            ValueError("Unknown projection type: invalid"),
        ),
        (
            DummyProjectorHyperparams(projection_type="special"),
            "dataset_small",
            None,
            ValueError("Unknown projection type: special"),
        ),
    ],
    indirect=["dataset_dispatcher"],
)
def test_project(
    hyperparams: DummyProjectorHyperparams,
    dataset_dispatcher: Tuple[DataFrame, List[str], List[str]],
    expected_shape: Tuple[int, int],
    expected_exception: Optional[BaseException],
):
    df, ls_cat_features, ls_cts_features = dataset_dispatcher
    projector = DummyProjector.build(
        ls_cat_features=ls_cat_features,
        ls_cts_features=ls_cts_features,
        projector_config=hyperparams,
    )
    if expected_exception:
        expected_exception_type = type(expected_exception)
        expected_exception_message = expected_exception.args[0]
        with pytest.raises(
            expected_exception_type,
            match=expected_exception_message,
        ):
            _ = projector.project(
                dataset=df,
            )
    else:
        result = projector.project(dataset=df)
        assert isinstance(result, np.ndarray)
        assert result.shape == expected_shape
        assert np.all(result >= 0) and np.all(result <= 1), (
            "Dummy projector should map in the interval [0, 1]"
        )


@pytest.mark.parametrize(
    "dataset_dispatcher",
    [
        ("dataset_small"),
        ("dataset_mixed"),
    ],
    indirect=["dataset_dispatcher"],
)
def test_produce_projection_plot(
    mock_plotter_factory: Callable[[], MagicMock],
    dataset_dispatcher: Tuple[DataFrame, List[str], List[str]],
):
    df, ls_cat_features, ls_cts_features = dataset_dispatcher
    mock_plotter = mock_plotter_factory()
    projector = DummyProjector(
        ls_cat_features=ls_cat_features,
        ls_cts_features=ls_cts_features,
        hyperparams=DummyProjectorHyperparams(projection_type="random"),
        plotter=mock_plotter,
    )
    result = projector.produce_projection_plot(dataset=df)
    assert result == mock_plotter.produce_projection_plot.return_value
    mock_plotter.produce_projection_plot.assert_called_once()
    args, kwargs = mock_plotter.produce_projection_plot.call_args
    assert "projection_name" in kwargs
    assert kwargs["projection_name"] == projector.projection_name
    assert "dataset_projection_2d" in kwargs


@pytest.mark.parametrize(
    "dataset_dispatcher",
    [
        ("dataset_small"),
        ("dataset_mixed"),
    ],
    indirect=["dataset_dispatcher"],
)
def test_produce_projection_comparison_plot(
    mock_plotter_factory: Callable[[], MagicMock],
    dataset_dispatcher: Tuple[DataFrame, List[str], List[str]],
):
    df, ls_cat_features, ls_cts_features = dataset_dispatcher
    mock_plotter = mock_plotter_factory()
    projector = DummyProjector(
        ls_cat_features=ls_cat_features,
        ls_cts_features=ls_cts_features,
        hyperparams=DummyProjectorHyperparams(projection_type="random"),
        plotter=mock_plotter,
    )
    result = projector.produce_projection_comparison_plot(dataset_real=df, dataset_synthetic=df)
    assert result == mock_plotter.produce_projection_comparison_plot.return_value
    mock_plotter.produce_projection_comparison_plot.assert_called_once()
    args, kwargs = mock_plotter.produce_projection_comparison_plot.call_args
    assert "projection_name" in kwargs
    assert kwargs["projection_name"] == projector.projection_name
    assert "dataset_projection_2d_real" in kwargs
    assert "dataset_projection_2d_synthetic" in kwargs
