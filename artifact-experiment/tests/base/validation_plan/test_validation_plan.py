from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_core.base.artifact_dependencies import ArtifactResult
from artifact_experiment.base.callbacks.artifact import (
    ArtifactCallbackResources,
)

from tests.base.validation_plan.dummy.validation_plan import (
    DummyArtifactResources,
    DummyResourceSpec,
    DummyValidationPlan,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "tracking_client_provided",
    [True, False],
)
def test_build(
    resources_factory: Callable[
        [], Tuple[ArtifactCallbackResources, DummyArtifactResources, DummyResourceSpec]
    ],
    mock_tracking_client_factory: Callable[[], MagicMock],
    tracking_client_provided: bool,
):
    _, _, resource_spec = resources_factory()
    tracking_client = mock_tracking_client_factory() if tracking_client_provided else None
    plan = DummyValidationPlan.build(resource_spec=resource_spec, tracking_client=tracking_client)
    assert isinstance(plan, DummyValidationPlan)
    assert plan.tracking_enabled == tracking_client_provided


@pytest.mark.unit
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
    resources_factory: Callable[
        [], Tuple[ArtifactCallbackResources, DummyArtifactResources, DummyResourceSpec]
    ],
    mock_tracking_client_factory: Callable[[], MagicMock],
    initial_client: Optional[bool],
    new_client: Optional[bool],
):
    callback_resources, _, resource_spec = resources_factory()
    initial_tracking_client = mock_tracking_client_factory() if initial_client else None
    new_tracking_client = mock_tracking_client_factory() if new_client else None
    plan = DummyValidationPlan.build(
        resource_spec=resource_spec, tracking_client=initial_tracking_client
    )
    assert plan.tracking_enabled == (initial_client is not None)
    plan.tracking_client = new_tracking_client
    assert plan.tracking_client == new_tracking_client
    assert plan.tracking_enabled == (new_client is not None)
    plan.execute(resources=callback_resources)
    if new_tracking_client is not None:
        new_tracking_client.log_score.assert_called()
        new_tracking_client.log_array.assert_called()
        new_tracking_client.log_plot.assert_called()
        new_tracking_client.log_score_collection.assert_called()
        new_tracking_client.log_array_collection.assert_called()
        new_tracking_client.log_plot_collection.assert_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "cache_type, ls_artifact_results",
    [
        ("scores", ["score_1", "score_2"]),
        ("arrays", ["array_1", "array_2"]),
        ("plots", ["plot_1", "plot_2"]),
        ("score_collections", ["score_collection_1", "score_collection_2"]),
        ("array_collections", ["array_collection_1", "array_collection_2"]),
        ("plot_collections", ["plot_collection_1", "plot_collection_2"]),
    ],
    indirect=["ls_artifact_results"],
)
def test_artifact_result_accessors(
    mock_callback_handlers: Dict[str, MagicMock],
    cache_type: str,
    ls_artifact_results: List[ArtifactResult],
):
    dict_cache = {
        f"entry_{idx}": artifact_result for idx, artifact_result in enumerate(ls_artifact_results)
    }
    for handler in mock_callback_handlers.values():
        handler.tracking_client = None
    plan = DummyValidationPlan(
        resource_spec=DummyResourceSpec(),
        score_handler=mock_callback_handlers["scores"],
        array_handler=mock_callback_handlers["arrays"],
        plot_handler=mock_callback_handlers["plots"],
        score_collection_handler=mock_callback_handlers["score_collections"],
        array_collection_handler=mock_callback_handlers["array_collections"],
        plot_collection_handler=mock_callback_handlers["plot_collections"],
    )
    mock_callback_handlers[cache_type].active_cache = dict_cache
    actual_values = getattr(plan, cache_type)
    assert actual_values == dict_cache


def test_execute(
    resources_factory: Callable[
        [], Tuple[ArtifactCallbackResources, DummyArtifactResources, DummyResourceSpec]
    ],
    mock_callback_handlers: Dict[str, MagicMock],
):
    callback_resources, _, _ = resources_factory()
    plan = DummyValidationPlan(
        resource_spec=DummyResourceSpec(),
        score_handler=mock_callback_handlers["scores"],
        array_handler=mock_callback_handlers["arrays"],
        plot_handler=mock_callback_handlers["plots"],
        score_collection_handler=mock_callback_handlers["score_collections"],
        array_collection_handler=mock_callback_handlers["array_collections"],
        plot_collection_handler=mock_callback_handlers["plot_collections"],
    )
    plan.execute(resources=callback_resources)
    for handler in mock_callback_handlers.values():
        handler.execute.assert_called_once_with(resources=callback_resources)


def test_execute_integration(
    resources_factory: Callable[
        [], Tuple[ArtifactCallbackResources, DummyArtifactResources, DummyResourceSpec]
    ],
    mock_tracking_client_factory: Callable[[], MagicMock],
):
    callback_resources, _, resource_spec = resources_factory()
    tracking_client = mock_tracking_client_factory()
    plan = DummyValidationPlan.build(resource_spec=resource_spec, tracking_client=tracking_client)
    assert plan.tracking_enabled
    plan.execute(resources=callback_resources)
    score_calls = tracking_client.log_score.call_args_list
    assert len(score_calls) == 1
    score_names = [call[1]["name"] for call in score_calls]
    assert "DUMMY_SCORE_1" in score_names
    array_calls = tracking_client.log_array.call_args_list
    assert len(array_calls) == 1
    assert array_calls[0][1]["name"] == "DUMMY_ARRAY_1"
    plot_calls = tracking_client.log_plot.call_args_list
    assert len(plot_calls) == 1
    assert plot_calls[0][1]["name"] == "DUMMY_PLOT_1"
    score_collection_calls = tracking_client.log_score_collection.call_args_list
    assert len(score_collection_calls) == 1
    assert score_collection_calls[0][1]["name"] == "DUMMY_SCORE_COLLECTION_1"
    array_collection_calls = tracking_client.log_array_collection.call_args_list
    assert len(array_collection_calls) == 1
    assert array_collection_calls[0][1]["name"] == "DUMMY_ARRAY_COLLECTION_1"
    plot_collection_calls = tracking_client.log_plot_collection.call_args_list
    assert len(plot_collection_calls) == 1
    assert plot_collection_calls[0][1]["name"] == "DUMMY_PLOT_COLLECTION_1"


def test_clear_cache(
    mock_callback_handlers: Dict[str, MagicMock],
):
    for handler in mock_callback_handlers.values():
        handler.tracking_client = None
    plan = DummyValidationPlan(
        resource_spec=DummyResourceSpec(),
        score_handler=mock_callback_handlers["scores"],
        array_handler=mock_callback_handlers["arrays"],
        plot_handler=mock_callback_handlers["plots"],
        score_collection_handler=mock_callback_handlers["score_collections"],
        array_collection_handler=mock_callback_handlers["array_collections"],
        plot_collection_handler=mock_callback_handlers["plot_collections"],
    )
    plan.clear_cache()
    for handler in mock_callback_handlers.values():
        handler.clear.assert_called_once()
