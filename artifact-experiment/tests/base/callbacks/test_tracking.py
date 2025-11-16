from typing import Callable, Dict, List, Literal, Optional, Tuple, overload
from unittest.mock import MagicMock

import pytest
from artifact_core._base.primitives import ArtifactResult
from artifact_experiment.base.components.callbacks.base import CallbackResources
from artifact_experiment.base.components.callbacks.tracking import (
    TrackingArrayCallback,
    TrackingArrayCollectionCallback,
    TrackingCallback,
    TrackingPlotCallback,
    TrackingPlotCollectionCallback,
    TrackingScoreCallback,
    TrackingScoreCollectionCallback,
)
from artifact_experiment.base.components.handlers.tracking import (
    TrackingArrayCollectionHandler,
    TrackingArrayHandler,
    TrackingCallbackHandler,
    TrackingPlotCollectionHandler,
    TrackingPlotHandler,
    TrackingScoreCollectionHandler,
    TrackingScoreHandler,
)
from artifact_experiment.base.tracking.backend.client import TrackingClient

from tests.base.callbacks.dummy.tracking import (
    DummyArrayCallback,
    DummyArrayCollectionCallback,
    DummyPlotCallback,
    DummyPlotCollectionCallback,
    DummyScoreCallback,
    DummyScoreCollectionCallback,
)

CallbackType = Literal[
    "score",
    "array",
    "plot",
    "score_collection",
    "array_collection",
    "plot_collection",
]


@pytest.fixture
def resources_factory() -> Callable[[], CallbackResources]:
    def _factory() -> CallbackResources:
        return CallbackResources()

    return _factory


@pytest.fixture
def score_callback_factory() -> Callable[
    [str, Optional[float], Optional[TrackingClient]], TrackingScoreCallback
]:
    def _factory(
        callback_key: str, compute_value: Optional[float], tracking_client: Optional[TrackingClient]
    ) -> TrackingScoreCallback:
        return DummyScoreCallback(
            key=callback_key, compute_value=compute_value, tracking_client=tracking_client
        )

    return _factory


@pytest.fixture
def array_callback_factory() -> Callable[
    [str, Optional[Array], Optional[TrackingClient]], TrackingArrayCallback
]:
    def _factory(
        callback_key: str,
        compute_value: Optional[Array],
        tracking_client: Optional[TrackingClient],
    ) -> TrackingArrayCallback:
        return DummyArrayCallback(
            key=callback_key, compute_value=compute_value, tracking_client=tracking_client
        )

    return _factory


@pytest.fixture
def plot_callback_factory() -> Callable[
    [str, Optional[Figure], Optional[TrackingClient]], TrackingPlotCallback
]:
    def _factory(
        callback_key: str,
        compute_value: Optional[Figure],
        tracking_client: Optional[TrackingClient],
    ) -> TrackingPlotCallback:
        return DummyPlotCallback(
            key=callback_key, compute_value=compute_value, tracking_client=tracking_client
        )

    return _factory


@pytest.fixture
def score_collection_callback_factory() -> Callable[
    [str, Optional[Dict[str, float]], Optional[TrackingClient]], TrackingScoreCollectionCallback
]:
    def _factory(
        callback_key: str,
        compute_value: Optional[Dict[str, float]],
        tracking_client: Optional[TrackingClient],
    ) -> TrackingScoreCollectionCallback:
        return DummyScoreCollectionCallback(
            key=callback_key, compute_value=compute_value, tracking_client=tracking_client
        )

    return _factory


@pytest.fixture
def array_collection_callback_factory() -> Callable[
    [str, Optional[Dict[str, Array]], Optional[TrackingClient]], TrackingArrayCollectionCallback
]:
    def _factory(
        callback_key: str,
        compute_value: Optional[Dict[str, Array]],
        tracking_client: Optional[TrackingClient],
    ) -> TrackingArrayCollectionCallback:
        return DummyArrayCollectionCallback(
            key=callback_key, compute_value=compute_value, tracking_client=tracking_client
        )

    return _factory


@pytest.fixture
def plot_collection_callback_factory() -> Callable[
    [str, Optional[Dict[str, Figure]], Optional[TrackingClient]], TrackingPlotCollectionCallback
]:
    def _factory(
        callback_key: str,
        compute_value: Optional[Dict[str, Figure]],
        tracking_client: Optional[TrackingClient],
    ) -> TrackingPlotCollectionCallback:
        return DummyPlotCollectionCallback(
            key=callback_key, compute_value=compute_value, tracking_client=tracking_client
        )

    return _factory


@overload
def callback_factory(
    callback_type: Literal["score"],
) -> Callable[[str, Optional[TrackingClient], Optional[float]], TrackingScoreCallback]: ...
@overload
def callback_factory(
    callback_type: Literal["array"],
) -> Callable[[str, Optional[TrackingClient], Optional[Array]], TrackingArrayCallback]: ...
@overload
def callback_factory(
    callback_type: Literal["plot"],
) -> Callable[[str, Optional[TrackingClient], Optional[Figure]], TrackingPlotCallback]: ...
@overload
def callback_factory(
    callback_type: Literal["score_collection"],
) -> Callable[
    [str, Optional[TrackingClient], Optional[float]], TrackingScoreCollectionCallback
]: ...
@overload
def callback_factory(
    callback_type: Literal["array_collection"],
) -> Callable[
    [str, Optional[TrackingClient], Optional[Array]], TrackingArrayCollectionCallback
]: ...
@overload
def callback_factory(
    callback_type: Literal["plot_collection"],
) -> Callable[
    [str, Optional[TrackingClient], Optional[Figure]], TrackingPlotCollectionCallback
]: ...


@pytest.fixture
def callback_factory(
    request,
) -> Callable[
    [CallbackType, str, Optional[ArtifactResult], Optional[TrackingClient]], TrackingCallback
]:
    def _factory(
        callback_type: CallbackType,
        callback_key: str,
        compute_value: Optional[ArtifactResult] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> TrackingCallback:
        factory_name = f"{callback_type}_callback_factory"

        try:
            factory = request.getfixturevalue(factory_name)
        except pytest.FixtureLookupError:
            raise ValueError(f"Unknown or missing callback factory for type '{callback_type}'")
        return factory(callback_key, compute_value, tracking_client)

    return _factory


@pytest.fixture
def handler_factory(
    callback_factory: Callable[
        [str, str, Optional[ArtifactResult], Optional[TrackingClient]], TrackingCallback
    ],
) -> Callable[
    [str, List[str], Optional[List[Optional[ArtifactResult]]], Optional[TrackingClient]],
    Tuple[TrackingCallbackHandler, List[TrackingCallback]],
]:
    def _factory(
        callback_type: str,
        ls_callback_keys: List[str],
        ls_callback_values: Optional[List[Optional[ArtifactResult]]],
        tracking_client: Optional[TrackingClient],
    ) -> Tuple[TrackingCallbackHandler, List[TrackingCallback]]:
        if ls_callback_values is None:
            ls_callback_values = [None for _ in ls_callback_keys]
        ls_callbacks = [
            callback_factory(callback_type, callback_key, callback_value, tracking_client)
            for callback_key, callback_value in zip(ls_callback_keys, ls_callback_values)
        ]
        if callback_type == "score":
            handler = TrackingScoreHandler(callbacks=ls_callbacks, tracking_client=tracking_client)
        elif callback_type == "array":
            handler = TrackingArrayHandler(callbacks=ls_callbacks, tracking_client=tracking_client)
        elif callback_type == "plot":
            handler = TrackingPlotHandler(callbacks=ls_callbacks, tracking_client=tracking_client)
        elif callback_type == "score_collection":
            handler = TrackingScoreCollectionHandler(
                callbacks=ls_callbacks, tracking_client=tracking_client
            )
        elif callback_type == "array_collection":
            handler = TrackingArrayCollectionHandler(
                callbacks=ls_callbacks, tracking_client=tracking_client
            )
        elif callback_type == "plot_collection":
            handler = TrackingPlotCollectionHandler(
                callbacks=ls_callbacks, tracking_client=tracking_client
            )
        else:
            raise ValueError(f"Unknown callback type: {callback_type}")
        return handler, ls_callbacks

    return _factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "callback_type, artifact_result, has_tracking_client",
    [
        ("score", "score_1", True),
        ("score", "score_1", False),
        ("score", "score_2", True),
        ("score", "score_2", False),
        ("array", "array_1", True),
        ("array", "array_1", False),
        ("array", "array_2", True),
        ("array", "array_2", False),
        ("plot", "plot_1", True),
        ("plot", "plot_1", False),
        ("plot", "plot_2", True),
        ("plot", "plot_2", False),
        ("score_collection", "score_collection_1", True),
        ("score_collection", "score_collection_1", False),
        ("score_collection", "score_collection_2", True),
        ("score_collection", "score_collection_2", False),
        ("array_collection", "array_collection_1", True),
        ("array_collection", "array_collection_1", False),
        ("array_collection", "array_collection_2", True),
        ("array_collection", "array_collection_2", False),
        ("plot_collection", "plot_collection_1", True),
        ("plot_collection", "plot_collection_1", False),
        ("plot_collection", "plot_collection_2", True),
        ("plot_collection", "plot_collection_2", False),
    ],
    indirect=["artifact_result"],
)
def test_callback_execute(
    resources_factory: Callable[[], CallbackResources],
    callback_factory: Callable[
        [CallbackType, str, Optional[ArtifactResult], Optional[TrackingClient]], TrackingCallback
    ],
    mock_tracking_client_factory: Callable[[], MagicMock],
    callback_type: CallbackType,
    artifact_result: ArtifactResult,
    has_tracking_client: bool,
):
    callback_key = "test_key"
    resources = resources_factory()
    tracking_client = mock_tracking_client_factory() if has_tracking_client else None
    callback = callback_factory(callback_type, callback_key, artifact_result, tracking_client)
    assert callback.tracking_enabled == has_tracking_client
    callback.execute(resources=resources)
    assert callback.key == callback_key
    if tracking_client is not None:
        log_method = getattr(tracking_client, "log_" + callback_type)
        log_method.assert_called_once()
        log_method_args, log_method_kwargs = log_method.call_args
        assert log_method_args == ()
        assert list(log_method_kwargs.values()) == [callback.value, callback.key]


@pytest.mark.unit
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
    callback_factory: Callable[
        [str, str, Optional[ArtifactResult], Optional[TrackingClient]], TrackingCallback
    ],
    mock_tracking_client_factory: Callable[[], MagicMock],
    callback_type: str,
    tracking_is_enabled_start: bool,
    tracking_is_enabled_end: bool,
):
    callback_key = "test_key"
    initial_tracking_client = mock_tracking_client_factory() if tracking_is_enabled_start else None
    new_tracking_client = mock_tracking_client_factory() if tracking_is_enabled_end else None
    callback = callback_factory(callback_type, callback_key, None, initial_tracking_client)
    assert callback.tracking_enabled == tracking_is_enabled_start
    callback.tracking_client = new_tracking_client
    assert callback._tracking_client == new_tracking_client
    assert callback.tracking_enabled == tracking_is_enabled_end


@pytest.mark.unit
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
        [str, List[str], Optional[List[Optional[ArtifactResult]]], Optional[TrackingClient]],
        Tuple[TrackingCallbackHandler, List[TrackingCallback]],
    ],
    mock_tracking_client_factory: Callable[[], MagicMock],
    callback_type: str,
    ls_callback_keys: List[str],
    has_tracking_client: bool,
):
    callback_resources = resources_factory()
    tracking_client = mock_tracking_client_factory() if has_tracking_client else None
    handler, ls_callbacks = handler_factory(callback_type, ls_callback_keys, None, tracking_client)
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


@pytest.mark.unit
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
        [str, List[str], Optional[List[Optional[ArtifactResult]]], Optional[TrackingClient]],
        Tuple[TrackingCallbackHandler, List[TrackingCallback]],
    ],
    callback_type: str,
    ls_callback_keys: List[str],
):
    callback_resources = resources_factory()
    tracking_client = None
    handler, ls_callbacks = handler_factory(callback_type, ls_callback_keys, None, tracking_client)
    handler.tracking_client = tracking_client
    handler.execute(resources=callback_resources)
    assert len(handler.active_cache) == len(ls_callback_keys)
    for callback_key in ls_callback_keys:
        assert callback_key in handler.active_cache
        assert handler.active_cache[callback_key] is not None
    handler.clear()
    assert len(handler.active_cache) == 0
    assert all(callback.value is None for callback in ls_callbacks)


@pytest.mark.unit
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
        [str, List[str], Optional[List[Optional[ArtifactResult]]], Optional[TrackingClient]],
        Tuple[TrackingCallbackHandler, List[TrackingCallback]],
    ],
    mock_tracking_client_factory: Callable[[], MagicMock],
    callback_type: str,
    ls_callback_keys: List[str],
    remove_tracking_client: bool,
):
    tracking_client = mock_tracking_client_factory() if remove_tracking_client is not None else None
    handler, ls_callbacks = handler_factory(callback_type, ls_callback_keys, None, tracking_client)
    handler.tracking_client = tracking_client
    assert handler.tracking_client == tracking_client
    assert handler.tracking_enabled == (tracking_client is not None)
    for callback in ls_callbacks:
        assert callback.tracking_client is None
        assert not callback.tracking_enabled
