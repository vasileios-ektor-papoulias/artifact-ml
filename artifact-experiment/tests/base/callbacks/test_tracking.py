from typing import Callable, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_experiment.base.callbacks.base import CallbackResources
from artifact_experiment.base.callbacks.tracking import (
    ArrayCallbackHandler,
    ArrayCollectionCallbackHandler,
    PlotCallbackHandler,
    PlotCollectionCallbackHandler,
    ScoreCallbackHandler,
    ScoreCollectionCallbackHandler,
    TrackingCallback,
    TrackingCallbackHandler,
)
from artifact_experiment.base.tracking.client import TrackingClient

from tests.base.callbacks.dummy.tracking import (
    DummyArrayCallback,
    DummyArrayCollectionCallback,
    DummyPlotCallback,
    DummyPlotCollectionCallback,
    DummyScoreCallback,
    DummyScoreCollectionCallback,
)


@pytest.fixture
def resources_factory() -> Callable[[], CallbackResources]:
    def _factory() -> CallbackResources:
        return CallbackResources()

    return _factory


@pytest.fixture
def callback_factory() -> Callable[[str, str, Optional[TrackingClient]], TrackingCallback]:
    def _factory(
        callback_type: str, callback_key: str, tracking_client: Optional[TrackingClient]
    ) -> TrackingCallback:
        if callback_type == "score":
            return DummyScoreCallback(key=callback_key, tracking_client=tracking_client)
        elif callback_type == "array":
            return DummyArrayCallback(key=callback_key, tracking_client=tracking_client)
        elif callback_type == "plot":
            return DummyPlotCallback(key=callback_key, tracking_client=tracking_client)
        elif callback_type == "score_collection":
            return DummyScoreCollectionCallback(key=callback_key, tracking_client=tracking_client)
        elif callback_type == "array_collection":
            return DummyArrayCollectionCallback(key=callback_key, tracking_client=tracking_client)
        elif callback_type == "plot_collection":
            return DummyPlotCollectionCallback(key=callback_key, tracking_client=tracking_client)
        else:
            raise ValueError(f"Unknown callback type: {callback_type}")

    return _factory


@pytest.fixture
def handler_factory(
    callback_factory: Callable[[str, str, Optional[TrackingClient]], TrackingCallback],
) -> Callable[
    [str, List[str], Optional[TrackingClient]],
    Tuple[TrackingCallbackHandler, List[TrackingCallback]],
]:
    def _factory(
        callback_type: str, ls_callback_keys: List[str], tracking_client: Optional[TrackingClient]
    ) -> Tuple[TrackingCallbackHandler, List[TrackingCallback]]:
        ls_callbacks = [
            callback_factory(callback_type, callback_key, tracking_client)
            for callback_key in ls_callback_keys
        ]
        if callback_type == "score":
            handler = ScoreCallbackHandler(
                ls_callbacks=ls_callbacks, tracking_client=tracking_client
            )
        elif callback_type == "array":
            handler = ArrayCallbackHandler(
                ls_callbacks=ls_callbacks, tracking_client=tracking_client
            )
        elif callback_type == "plot":
            handler = PlotCallbackHandler(
                ls_callbacks=ls_callbacks, tracking_client=tracking_client
            )
        elif callback_type == "score_collection":
            handler = ScoreCollectionCallbackHandler(
                ls_callbacks=ls_callbacks, tracking_client=tracking_client
            )
        elif callback_type == "array_collection":
            handler = ArrayCollectionCallbackHandler(
                ls_callbacks=ls_callbacks, tracking_client=tracking_client
            )
        elif callback_type == "plot_collection":
            handler = PlotCollectionCallbackHandler(
                ls_callbacks=ls_callbacks, tracking_client=tracking_client
            )
        else:
            raise ValueError(f"Unknown callback type: {callback_type}")
        return handler, ls_callbacks

    return _factory


@pytest.mark.parametrize(
    "callback_type, has_tracking_client",
    [
        ("score", True),
        ("score", False),
        ("array", True),
        ("array", False),
        ("plot", True),
        ("plot", False),
        ("score_collection", True),
        ("score_collection", False),
        ("array_collection", True),
        ("array_collection", False),
        ("plot_collection", True),
        ("plot_collection", False),
    ],
)
def test_callback_execute(
    resources_factory: Callable[[], CallbackResources],
    callback_factory: Callable[[str, str, Optional[TrackingClient]], TrackingCallback],
    mock_tracking_client_factory: Callable[[], MagicMock],
    callback_type: str,
    has_tracking_client: bool,
):
    callback_key = "test_key"
    resources = resources_factory()
    tracking_client = mock_tracking_client_factory() if has_tracking_client else None
    callback = callback_factory(callback_type, callback_key, tracking_client)
    assert callback.tracking_enabled == has_tracking_client
    callback.execute(resources=resources)
    assert callback.key == callback_key
    if tracking_client is not None:
        log_method = getattr(tracking_client, "log_" + callback_type)
        log_method.assert_called_once()
        log_method_args, log_method_kwargs = log_method.call_args
        assert log_method_args == ()
        assert list(log_method_kwargs.values()) == [callback.value, callback.key]


@pytest.mark.parametrize(
    "callback_type, tracking_is_enabled_start, tracking_is_enabled_end",
    [
        ("score", False, True),
        ("score", True, False),
        ("score", True, True),
        ("score", False, False),
        ("array", False, True),
        ("array", True, False),
        ("array", True, True),
        ("array", False, False),
        ("plot", False, True),
        ("plot", True, False),
        ("plot", True, True),
        ("plot", False, False),
        ("score_collection", False, True),
        ("score_collection", True, False),
        ("score_collection", True, True),
        ("score_collection", False, False),
        ("array_collection", False, True),
        ("array_collection", True, False),
        ("array_collection", True, True),
        ("array_collection", False, False),
        ("plot_collection", False, True),
        ("plot_collection", True, False),
        ("plot_collection", True, True),
        ("plot_collection", False, False),
    ],
)
def test_callback_tracking_client_setter(
    callback_factory: Callable[[str, str, Optional[TrackingClient]], TrackingCallback],
    mock_tracking_client_factory: Callable[[], MagicMock],
    callback_type: str,
    tracking_is_enabled_start: bool,
    tracking_is_enabled_end: bool,
):
    callback_key = "test_key"
    initial_tracking_client = mock_tracking_client_factory() if tracking_is_enabled_start else None
    new_tracking_client = mock_tracking_client_factory() if tracking_is_enabled_end else None
    callback = callback_factory(callback_type, callback_key, initial_tracking_client)
    assert callback.tracking_enabled == tracking_is_enabled_start
    callback.tracking_client = new_tracking_client
    assert callback._tracking_client == new_tracking_client
    assert callback.tracking_enabled == tracking_is_enabled_end


@pytest.mark.parametrize(
    "callback_type, ls_callback_keys, has_tracking_client",
    [
        ("score", ["key_1", "key_2"], True),
        ("score", ["key_1", "key_2"], False),
        ("array", ["key_1", "key_2"], True),
        ("array", ["key_1", "key_2"], False),
        ("plot", ["key_1", "key_2"], True),
        ("plot", ["key_1", "key_2"], False),
        ("score_collection", ["key_1", "key_2"], True),
        ("score_collection", ["key_1", "key_2"], False),
        ("array_collection", ["key_1", "key_2"], True),
        ("array_collection", ["key_1", "key_2"], False),
        ("plot_collection", ["key_1", "key_2"], True),
        ("plot_collection", ["key_1", "key_2"], False),
    ],
)
def test_handler_execute(
    resources_factory: Callable[[], CallbackResources],
    handler_factory: Callable[
        [str, List[str], Optional[TrackingClient]],
        Tuple[TrackingCallbackHandler, List[TrackingCallback]],
    ],
    mock_tracking_client_factory: Callable[[], MagicMock],
    callback_type: str,
    ls_callback_keys: List[str],
    has_tracking_client: bool,
):
    callback_resources = resources_factory()
    tracking_client = mock_tracking_client_factory() if has_tracking_client else None
    handler, ls_callbacks = handler_factory(callback_type, ls_callback_keys, tracking_client)
    handler.tracking_client = tracking_client
    assert handler.tracking_enabled == has_tracking_client
    handler.execute(resources=callback_resources)
    assert len(handler.active_cache) == len(ls_callback_keys)
    for callback_key in ls_callback_keys:
        assert callback_key in handler.active_cache
        assert handler.active_cache[callback_key] is not None
    if tracking_client is not None:
        log_method = getattr(tracking_client, "log_" + callback_type)
        assert log_method.call_count == len(ls_callback_keys)
        for call_args, callback in zip(log_method.call_args_list, ls_callbacks):
            args, kwargs = call_args
            assert args == ()
            assert list(kwargs.values()) == [callback.value, callback.key]


@pytest.mark.parametrize(
    "callback_type, ls_callback_keys",
    [
        ("score", ["key_1", "key_2"]),
        ("array", ["key_1", "key_2"]),
        ("plot", ["key_1", "key_2"]),
        ("score_collection", ["key_1", "key_2"]),
        ("array_collection", ["key_1", "key_2"]),
        ("plot_collection", ["key_1", "key_2"]),
    ],
)
def test_handler_clear(
    resources_factory: Callable[[], CallbackResources],
    handler_factory: Callable[
        [str, List[str], Optional[TrackingClient]],
        Tuple[TrackingCallbackHandler, List[TrackingCallback]],
    ],
    callback_type: str,
    ls_callback_keys: List[str],
):
    callback_resources = resources_factory()
    tracking_client = None
    handler, ls_callbacks = handler_factory(callback_type, ls_callback_keys, tracking_client)
    handler.tracking_client = tracking_client
    handler.execute(resources=callback_resources)
    assert len(handler.active_cache) == len(ls_callback_keys)
    for callback_key in ls_callback_keys:
        assert callback_key in handler.active_cache
        assert handler.active_cache[callback_key] is not None
    handler.clear()
    assert len(handler.active_cache) == 0
    assert all(callback.value is None for callback in ls_callbacks)


@pytest.mark.parametrize(
    "callback_type, ls_callback_keys, remove_tracking_client",
    [
        ("score", ["key_1", "key_2"], True),
        ("score", ["key_1", "key_2"], False),
        ("array", ["key_1", "key_2"], True),
        ("array", ["key_1", "key_2"], False),
        ("plot", ["key_1", "key_2"], True),
        ("plot", ["key_1", "key_2"], False),
        ("score_collection", ["key_1", "key_2"], True),
        ("score_collection", ["key_1", "key_2"], False),
        ("array_collection", ["key_1", "key_2"], True),
        ("array_collection", ["key_1", "key_2"], False),
        ("plot_collection", ["key_1", "key_2"], True),
        ("plot_collection", ["key_1", "key_2"], False),
    ],
)
def test_handler_tracking_client_setter(
    handler_factory: Callable[
        [str, List[str], Optional[TrackingClient]],
        Tuple[TrackingCallbackHandler, List[TrackingCallback]],
    ],
    mock_tracking_client_factory: Callable[[], MagicMock],
    callback_type: str,
    ls_callback_keys: List[str],
    remove_tracking_client: bool,
):
    tracking_client = mock_tracking_client_factory() if remove_tracking_client is not None else None
    handler, ls_callbacks = handler_factory(callback_type, ls_callback_keys, tracking_client)
    handler.tracking_client = tracking_client
    assert handler.tracking_client == tracking_client
    assert handler.tracking_enabled == (tracking_client is not None)
    for callback in ls_callbacks:
        assert callback.tracking_client is None
        assert not callback.tracking_enabled
