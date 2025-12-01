from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.projections.base.plotter import (
    ProjectionPlotter,
)

from tests._libs.artifacts.table_comparison.projections.base.dummy.projector import (
    DummyProjector,
    DummyProjectorHyperparams,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "ls_cat_features, ls_cts_features, hyperparams, "
    + "expected_projection_type, expected_exception",
    [
        (["cts_1"], [], DummyProjectorHyperparams(projection_type="random"), "random", None),
        ([], ["cts_1"], DummyProjectorHyperparams(projection_type="random"), "random", None),
        (["cts_1"], ["cts_2"], DummyProjectorHyperparams(projection_type="random"), "random", None),
        (
            ["cts_1", "cts_2"],
            ["cts_1"],
            DummyProjectorHyperparams(projection_type="random"),
            "random",
            ValueError("Categorical and continuous features overlap: {'cts_1'}"),
        ),
        (
            ["cts_1"],
            ["cts_1"],
            DummyProjectorHyperparams(projection_type="random"),
            "random",
            ValueError("Categorical and continuous features overlap: {'cts_1'}"),
        ),
        (
            ["cts_1", "cts_2"],
            ["cts_3"],
            DummyProjectorHyperparams(projection_type="random"),
            "random",
            None,
        ),
        (
            [],
            [],
            DummyProjectorHyperparams(projection_type="random"),
            "random",
            ValueError("Both categorical and continuous feature lists are empty."),
        ),
        (["cts_1"], [], DummyProjectorHyperparams(projection_type="invalid"), "invalid", None),
        ([], ["cts_1"], DummyProjectorHyperparams(projection_type="invalid"), "invalid", None),
        (
            ["cts_1"],
            ["cts_2"],
            DummyProjectorHyperparams(projection_type="invalid"),
            "invalid",
            None,
        ),
        (
            ["cts_1"],
            ["cts_1"],
            DummyProjectorHyperparams(projection_type="invalid"),
            "invalid",
            ValueError("Categorical and continuous features overlap: {'cts_1'}"),
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


@pytest.mark.unit
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
    dataset_dispatcher: Tuple[pd.DataFrame, List[str], List[str]],
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset_dispatcher",
    [
        ("dataset_small"),
        ("dataset_mixed"),
    ],
    indirect=["dataset_dispatcher"],
)
def test_produce_projection_plot(
    mock_plotter: MagicMock,
    dataset_dispatcher: Tuple[pd.DataFrame, List[str], List[str]],
):
    df, ls_cat_features, ls_cts_features = dataset_dispatcher
    projector = DummyProjector(
        cat_features=ls_cat_features,
        cts_features=ls_cts_features,
        hyperparams=DummyProjectorHyperparams(projection_type="random"),
        plotter=mock_plotter,
    )
    result = projector.produce_projection_plot(dataset=df)
    assert result == mock_plotter.produce_projection_plot.return_value
    mock_plotter.produce_projection_plot.assert_called_once()
    _, kwargs = mock_plotter.produce_projection_plot.call_args
    assert "projection_name" in kwargs
    assert kwargs["projection_name"] == projector.projection_name
    assert "dataset_projection_2d" in kwargs


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset_dispatcher",
    [
        ("dataset_small"),
        ("dataset_mixed"),
    ],
    indirect=["dataset_dispatcher"],
)
def test_produce_projection_comparison_plot(
    mock_plotter: MagicMock,
    dataset_dispatcher: Tuple[pd.DataFrame, List[str], List[str]],
):
    df, ls_cat_features, ls_cts_features = dataset_dispatcher
    projector = DummyProjector(
        cat_features=ls_cat_features,
        cts_features=ls_cts_features,
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
