from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
from artifact_experiment.base.callbacks.artifact import (
    ArtifactCallbackResources,
)
from matplotlib.figure import Figure

from tests.base.validation_plan.dummy import (
    DummyResourceSpec,
    DummyValidationPlan,
)


def test_init(
    resource_spec: DummyResourceSpec,
    callback_resources: ArtifactCallbackResources,
):
    plan = DummyValidationPlan.build(resource_spec=resource_spec, tracking_client=None)
    plan.execute(resources=callback_resources)
    plan.clear_cache()
    _ = plan.scores
    _ = plan.arrays
    _ = plan.plots
    _ = plan.score_collections
    _ = plan.array_collections
    _ = plan.plot_collections
    assert isinstance(plan, DummyValidationPlan)


@pytest.mark.parametrize(
    "tracking_client_provided",
    [True, False],
)
def test_build(
    resource_spec: DummyResourceSpec,
    mock_tracking_client: MagicMock,
    callback_resources: ArtifactCallbackResources,
    tracking_client_provided: bool,
):
    tracking_client = mock_tracking_client if tracking_client_provided else None
    plan = DummyValidationPlan.build(resource_spec=resource_spec, tracking_client=tracking_client)
    assert isinstance(plan, DummyValidationPlan)
    plan.execute(resources=callback_resources)
    if tracking_client_provided:
        mock_tracking_client.log_score.assert_called()
        mock_tracking_client.log_array.assert_called()
        mock_tracking_client.log_plot.assert_called()
        mock_tracking_client.log_score_collection.assert_called()
        mock_tracking_client.log_array_collection.assert_called()
        mock_tracking_client.log_plot_collection.assert_called()
        score_calls = mock_tracking_client.log_score.call_args_list
        assert len(score_calls) >= 2  # SCORE1 and SCORE2
        for call in score_calls:
            args, kwargs = call
            assert "score" in kwargs
            assert "name" in kwargs
            assert isinstance(kwargs["score"], float)
            assert isinstance(kwargs["name"], str)
        array_calls = mock_tracking_client.log_array.call_args_list
        assert len(array_calls) >= 1  # ARRAY1
        for call in array_calls:
            args, kwargs = call
            assert "array" in kwargs
            assert "name" in kwargs
        score_collection_calls = mock_tracking_client.log_score_collection.call_args_list
        assert len(score_collection_calls) >= 1
        for call in score_collection_calls:
            args, kwargs = call
            assert "score_collection" in kwargs
            assert "name" in kwargs
    else:
        mock_tracking_client.log_score.assert_not_called()
        mock_tracking_client.log_array.assert_not_called()
        mock_tracking_client.log_plot.assert_not_called()
        mock_tracking_client.log_score_collection.assert_not_called()
        mock_tracking_client.log_array_collection.assert_not_called()
        mock_tracking_client.log_plot_collection.assert_not_called()

    plan.clear_cache()
    _ = plan.scores
    _ = plan.arrays
    _ = plan.plots
    _ = plan.score_collections
    _ = plan.array_collections
    _ = plan.plot_collections


@pytest.mark.parametrize(
    "cache_type, expected_values",
    [
        ("scores", {"score1": 1.0, "score2": 2.0}),
        ("arrays", {"array1": np.array([1, 2, 3])}),
        ("plots", {"plot1": Figure()}),
        ("score_collections", {"collection1": {"sub1": 1.0, "sub2": 2.0}}),
        ("array_collections", {"collection1": {"sub1": np.array([1, 2])}}),
        ("plot_collections", {"collection1": {"sub1": Figure()}}),
    ],
)
def test_properties(
    mock_handlers: Dict[str, MagicMock],
    cache_type: str,
    expected_values: Dict,
):
    plan = DummyValidationPlan(
        score_handler=mock_handlers["score"],
        array_handler=mock_handlers["array"],
        plot_handler=mock_handlers["plot"],
        score_collection_handler=mock_handlers["score_collection"],
        array_collection_handler=mock_handlers["array_collection"],
        plot_collection_handler=mock_handlers["plot_collection"],
    )
    handler_mapping = {
        "scores": "score",
        "arrays": "array",
        "plots": "plot",
        "score_collections": "score_collection",
        "array_collections": "array_collection",
        "plot_collections": "plot_collection",
    }
    mock_handlers[handler_mapping[cache_type]].active_cache = expected_values
    actual_values = getattr(plan, cache_type)
    assert actual_values == expected_values


def test_execute(
    mock_handlers: Dict[str, MagicMock],
    callback_resources: ArtifactCallbackResources,
):
    plan = DummyValidationPlan(
        score_handler=mock_handlers["score"],
        array_handler=mock_handlers["array"],
        plot_handler=mock_handlers["plot"],
        score_collection_handler=mock_handlers["score_collection"],
        array_collection_handler=mock_handlers["array_collection"],
        plot_collection_handler=mock_handlers["plot_collection"],
    )

    plan.execute(resources=callback_resources)
    for handler in mock_handlers.values():
        handler.execute.assert_called_once_with(resources=callback_resources)


@pytest.mark.parametrize(
    "handlers_to_test, expected_clear_calls",
    [
        (["score"], 1),
        (["score", "array"], 2),
        (["score", "array", "plot"], 3),
        (["score", "array", "plot", "score_collection"], 4),
        (
            ["score", "array", "plot", "score_collection", "array_collection"],
            5,
        ),
        (
            [
                "score",
                "array",
                "plot",
                "score_collection",
                "array_collection",
                "plot_collection",
            ],
            6,
        ),
    ],
)
def test_clear_cache(
    mock_handlers: Dict[str, MagicMock],
    handlers_to_test: list,
    expected_clear_calls: int,
):
    plan_handlers = {}
    for handler_type in [
        "score",
        "array",
        "plot",
        "score_collection",
        "array_collection",
        "plot_collection",
    ]:
        if handler_type in handlers_to_test:
            plan_handlers[f"{handler_type}_handler"] = mock_handlers[handler_type]
        else:
            plan_handlers[f"{handler_type}_handler"] = MagicMock()
    plan = DummyValidationPlan(**plan_handlers)
    plan.clear_cache()
    total_calls = 0
    for handler_type in handlers_to_test:
        mock_handlers[handler_type].clear.assert_called_once()
        total_calls += 1

    assert total_calls == expected_clear_calls


def test_tracking_client_integration(
    resource_spec: DummyResourceSpec,
    mock_tracking_client: MagicMock,
    callback_resources: ArtifactCallbackResources,
):
    plan = DummyValidationPlan.build(
        resource_spec=resource_spec, tracking_client=mock_tracking_client
    )
    plan.execute(resources=callback_resources)
    score_calls = mock_tracking_client.log_score.call_args_list
    assert len(score_calls) >= 2
    for call in score_calls:
        args, kwargs = call
        assert "score" in kwargs
        assert "name" in kwargs
        assert isinstance(kwargs["name"], str)
        assert isinstance(kwargs["score"], float)
        assert kwargs["score"] == 42.0
    array_calls = mock_tracking_client.log_array.call_args_list
    assert len(array_calls) >= 1
    for call in array_calls:
        args, kwargs = call
        assert "array" in kwargs
        assert "name" in kwargs
        assert isinstance(kwargs["name"], str)
    plot_calls = mock_tracking_client.log_plot.call_args_list
    assert len(plot_calls) >= 1
    score_collection_calls = mock_tracking_client.log_score_collection.call_args_list
    assert len(score_collection_calls) >= 1
    array_collection_calls = mock_tracking_client.log_array_collection.call_args_list
    assert len(array_collection_calls) >= 1
    plot_collection_calls = mock_tracking_client.log_plot_collection.call_args_list
    assert len(plot_collection_calls) >= 1
