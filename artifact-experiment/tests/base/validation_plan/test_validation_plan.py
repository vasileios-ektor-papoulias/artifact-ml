from typing import Dict, Optional
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


@pytest.mark.parametrize(
    "tracking_client_provided",
    [True, False],
)
def test_build(
    mock_tracking_client: MagicMock,
    resource_spec: DummyResourceSpec,
    tracking_client_provided: bool,
):
    tracking_client = mock_tracking_client if tracking_client_provided else None
    plan = DummyValidationPlan.build(resource_spec=resource_spec, tracking_client=tracking_client)
    assert isinstance(plan, DummyValidationPlan)
    assert plan.tracking_enabled == tracking_client_provided


@pytest.mark.parametrize(
    "initial_client, new_client",
    [
        (None, True),
        (True, None),
        (True, True),
        (None, None),
    ],
)
def test_tracking_client_setter(
    mock_tracking_client: MagicMock,
    resource_spec: DummyResourceSpec,
    callback_resources: ArtifactCallbackResources,
    initial_client: Optional[bool],
    new_client: Optional[bool],
):
    initial_tracking_client = mock_tracking_client if initial_client else None
    new_tracking_client = mock_tracking_client if new_client else None
    plan = DummyValidationPlan.build(
        resource_spec=resource_spec, tracking_client=initial_tracking_client
    )
    assert plan.tracking_enabled == (initial_client is not None)
    plan.tracking_client = new_tracking_client
    assert plan.tracking_client == new_tracking_client
    assert plan.tracking_enabled == (new_client is not None)
    
    plan.execute(resources=callback_resources)
    
    if new_client is not None:
        mock_tracking_client.log_score.assert_called()
    else:
        mock_tracking_client.log_score.assert_not_called()
        mock_tracking_client.log_array.assert_not_called()
        mock_tracking_client.log_plot.assert_not_called()
        mock_tracking_client.log_score_collection.assert_not_called()
        mock_tracking_client.log_array_collection.assert_not_called()
        mock_tracking_client.log_plot_collection.assert_not_called()


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
def test_artifact_result_accessors(
    mock_handlers: Dict[str, MagicMock],
    cache_type: str,
    expected_values: Dict,
):
    for handler in mock_handlers.values():
        handler.tracking_client = None
    plan = DummyValidationPlan(
        score_handler=mock_handlers["scores"],
        array_handler=mock_handlers["arrays"],
        plot_handler=mock_handlers["plots"],
        score_collection_handler=mock_handlers["score_collections"],
        array_collection_handler=mock_handlers["array_collections"],
        plot_collection_handler=mock_handlers["plot_collections"],
    )
    assert not plan.tracking_enabled
    mock_handlers[cache_type].active_cache = expected_values
    actual_values = getattr(plan, cache_type)
    assert actual_values == expected_values


def test_execute(
    mock_handlers: Dict[str, MagicMock],
    callback_resources: ArtifactCallbackResources,
):
    for handler in mock_handlers.values():
        handler.tracking_client = None
    plan = DummyValidationPlan(
        score_handler=mock_handlers["scores"],
        array_handler=mock_handlers["arrays"],
        plot_handler=mock_handlers["plots"],
        score_collection_handler=mock_handlers["score_collections"],
        array_collection_handler=mock_handlers["array_collections"],
        plot_collection_handler=mock_handlers["plot_collections"],
    )
    plan.execute(resources=callback_resources)
    for handler in mock_handlers.values():
        handler.execute.assert_called_once_with(resources=callback_resources)


def test_tracking_client_integration(
    mock_tracking_client: MagicMock,
    resource_spec: DummyResourceSpec,
    callback_resources: ArtifactCallbackResources,
):
    plan = DummyValidationPlan.build(
        resource_spec=resource_spec, tracking_client=mock_tracking_client
    )
    assert plan.tracking_enabled
    plan.execute(resources=callback_resources)
    score_calls = mock_tracking_client.log_score.call_args_list
    assert len(score_calls) == 1
    for call in score_calls:
        _, kwargs = call
        assert "name" in kwargs
        assert kwargs["name"] == "SCORE1"
        assert "score" in kwargs
        assert kwargs["score"] == 0
    array_calls = mock_tracking_client.log_array.call_args_list
    assert len(array_calls) == 0
    plot_calls = mock_tracking_client.log_plot.call_args_list
    assert len(plot_calls) == 0
    score_collection_calls = mock_tracking_client.log_score_collection.call_args_list
    assert len(score_collection_calls) == 0
    array_collection_calls = mock_tracking_client.log_array_collection.call_args_list
    assert len(array_collection_calls) == 0
    plot_collection_calls = mock_tracking_client.log_plot_collection.call_args_list
    assert len(plot_collection_calls) == 0


def test_clear_cache(
    mock_handlers: Dict[str, MagicMock],
):
    for handler in mock_handlers.values():
        handler.tracking_client = None
    plan = DummyValidationPlan(
        score_handler=mock_handlers["scores"],
        array_handler=mock_handlers["arrays"],
        plot_handler=mock_handlers["plots"],
        score_collection_handler=mock_handlers["score_collections"],
        array_collection_handler=mock_handlers["array_collections"],
        plot_collection_handler=mock_handlers["plot_collections"],
    )
    assert not plan.tracking_enabled

    plan.clear_cache()

    for handler in mock_handlers.values():
        handler.clear.assert_called_once()
