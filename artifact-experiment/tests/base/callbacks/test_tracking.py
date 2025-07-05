from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
from artifact_core.base.artifact_dependencies import ArtifactResult
from artifact_experiment.base.callbacks.base import CallbackResources
from artifact_experiment.base.callbacks.tracking import TrackingCallback, TrackingCallbackHandler

from tests.base.callbacks.dummy.tracking import (
    DummyArrayCallback,
    DummyArrayCallbackHandler,
    DummyScoreCallback,
    DummyScoreCallbackHandler,
    DummyScoreCollectionCallback,
    DummyScoreCollectionCallbackHandler,
)


@pytest.mark.parametrize(
    "initial_client, new_client",
    [
        (None, True),
        (True, None),
        (True, True),
        (None, None),
    ],
)
def test_callback_tracking_client_setter(
    callback_key: str,
    mock_tracking_client: MagicMock,
    initial_client: Optional[bool],
    new_client: Optional[bool],
):
    initial_tracking_client = mock_tracking_client if initial_client else None
    new_tracking_client = mock_tracking_client if new_client else None
    callback = DummyScoreCallback(key=callback_key, tracking_client=initial_tracking_client)
    assert callback.tracking_enabled == (initial_client is not None)
    callback.tracking_client = new_tracking_client
    assert callback._tracking_client == new_tracking_client
    assert callback.tracking_enabled == (new_client is not None)


@pytest.mark.parametrize(
    "key, compute_value, has_tracking_client",
    [
        ("test_key", 42.0, True),
        ("another_key", 1.5, False),
        ("third_key", 0.0, True),
        ("negative_key", -1.0, False),
    ],
)
def test_execute(
    key: str,
    compute_value: float,
    has_tracking_client: bool,
    callback_resources: CallbackResources,
    mock_tracking_client: MagicMock,
):
    tracking_client = mock_tracking_client if has_tracking_client else None
    callback = DummyScoreCallback(
        key=key, compute_value=compute_value, tracking_client=tracking_client
    )
    assert callback.tracking_enabled == has_tracking_client

    callback.execute(resources=callback_resources)
    assert callback.key == key
    assert callback.value == compute_value
    if tracking_client is not None:
        tracking_client.log_score.assert_called_once_with(score=compute_value, name=key)


@pytest.mark.parametrize(
    "callback, expected_value, expected_log_method, expected_key",
    [
        (
            DummyScoreCallback(key="dummy_score", compute_value=42.0),
            42.0,
            "log_score",
            "dummy_score",
        ),
        (
            DummyArrayCallback(key="dummy_array", compute_value=np.array([1, 2, 3])),
            np.array([1, 2, 3]),
            "log_array",
            "dummy_array",
        ),
        (
            DummyScoreCollectionCallback(
                key="dummy_score_collection", compute_value={"score1": 1.0, "score2": 2.0}
            ),
            {"score1": 1.0, "score2": 2.0},
            "log_score_collection",
            "dummy_score_collection",
        ),
    ],
)
def test_execute_concrete(
    callback_resources: CallbackResources,
    mock_tracking_client: MagicMock,
    callback: TrackingCallback[CallbackResources, ArtifactResult],
    expected_value: ArtifactResult,
    expected_log_method: str,
    expected_key: str,
):
    tracking_client = mock_tracking_client
    callback.tracking_client = tracking_client
    assert callback.tracking_enabled
    callback.execute(resources=callback_resources)
    log_method = getattr(tracking_client, expected_log_method)
    if expected_log_method == "log_score":
        log_method.assert_called_once_with(score=expected_value, name=expected_key)
    elif expected_log_method == "log_array":
        assert isinstance(expected_value, np.ndarray)
        log_method.assert_called_once()
        args, kwargs = log_method.call_args
        assert "array" in kwargs
        assert "name" in kwargs
        assert np.array_equal(kwargs["array"], expected_value)
        assert kwargs["name"] == expected_key
    elif expected_log_method == "log_score_collection":
        log_method.assert_called_once_with(score_collection=expected_value, name=expected_key)


@pytest.mark.parametrize(
    "handler, has_tracking_client, expected_compute_values, expected_log_method",
    [
        (
            DummyScoreCallbackHandler(
                ls_callbacks=[
                    DummyScoreCallback(key="key1", compute_value=1.0),
                    DummyScoreCallback(key="key2", compute_value=2.0),
                    DummyScoreCallback(key="key3", compute_value=3.0),
                ]
            ),
            True,
            [1.0, 2.0, 3.0],
            "log_score",
        ),
        (
            DummyScoreCallbackHandler(
                ls_callbacks=[
                    DummyScoreCallback(key="key1", compute_value=1.0),
                    DummyScoreCallback(key="key2", compute_value=2.0),
                ]
            ),
            False,
            [1.0, 2.0],
            "log_score",
        ),
        (
            DummyArrayCallbackHandler(
                ls_callbacks=[
                    DummyArrayCallback(key="key1", compute_value=np.array([1])),
                    DummyArrayCallback(key="key2", compute_value=np.array([2, 3])),
                ]
            ),
            True,
            [np.array([1]), np.array([2, 3])],
            "log_array",
        ),
        (
            DummyScoreCollectionCallbackHandler(
                ls_callbacks=[
                    DummyScoreCollectionCallback(key="key1", compute_value={"score1": 1.0}),
                    DummyScoreCollectionCallback(key="key2", compute_value={"score2": 2.0}),
                    DummyScoreCollectionCallback(key="key3", compute_value={"score3": 3.0}),
                ]
            ),
            False,
            [{"score1": 1.0}, {"score2": 2.0}, {"score3": 3.0}],
            "log_score_collection",
        ),
    ],
)
def test_handler_execute(
    mock_tracking_client: MagicMock,
    callback_resources: CallbackResources,
    handler: TrackingCallbackHandler[
        TrackingCallback[CallbackResources, ArtifactResult], CallbackResources, ArtifactResult
    ],
    has_tracking_client: bool,
    expected_compute_values: List[ArtifactResult],
    expected_log_method: str,
):
    tracking_client = mock_tracking_client if has_tracking_client else None
    handler.tracking_client = tracking_client
    assert handler.tracking_enabled == has_tracking_client
    handler.execute(resources=callback_resources)
    assert len(handler.active_cache) == len(expected_compute_values)
    for i, expected_value in enumerate(expected_compute_values):
        key = f"key{i + 1}"
        if isinstance(expected_value, np.ndarray):
            actual_value = handler.active_cache[key]
            assert isinstance(actual_value, np.ndarray)
            assert np.array_equal(actual_value, expected_value)
        else:
            assert handler.active_cache[key] == expected_value
    if tracking_client is not None:
        log_method = getattr(tracking_client, expected_log_method)
        assert log_method.call_count == len(expected_compute_values)
        if expected_log_method == "log_score":
            for i, value in enumerate(expected_compute_values):
                key = f"key{i + 1}"
                log_method.assert_any_call(score=value, name=key)


@pytest.mark.parametrize(
    "handler, expected_compute_values",
    [
        (
            DummyScoreCallbackHandler(
                ls_callbacks=[
                    DummyScoreCallback(key="key1", compute_value=1.0),
                    DummyScoreCallback(key="key2", compute_value=2.0),
                ]
            ),
            [1.0, 2.0],
        ),
        (
            DummyArrayCallbackHandler(
                ls_callbacks=[
                    DummyArrayCallback(key="key1", compute_value=np.array([1])),
                    DummyArrayCallback(key="key2", compute_value=np.array([2, 3])),
                ]
            ),
            [np.array([1]), np.array([2, 3])],
        ),
        (
            DummyScoreCollectionCallbackHandler(
                ls_callbacks=[
                    DummyScoreCollectionCallback(key="key1", compute_value={"score1": 1.0}),
                ]
            ),
            [{"score1": 1.0}],
        ),
    ],
)
def test_handler_clear(
    callback_resources: CallbackResources,
    handler: TrackingCallbackHandler[
        TrackingCallback[CallbackResources, ArtifactResult], CallbackResources, ArtifactResult
    ],
    expected_compute_values: List[ArtifactResult],
):
    handler.execute(resources=callback_resources)
    assert len(handler.active_cache) == len(expected_compute_values)
    for i, expected_value in enumerate(expected_compute_values):
        key = f"key{i + 1}"
        if isinstance(expected_value, np.ndarray):
            actual_value = handler.active_cache[key]
            assert isinstance(actual_value, np.ndarray)
            assert np.array_equal(actual_value, expected_value)
        else:
            assert handler.active_cache[key] == expected_value

    handler.clear()
    assert len(handler.active_cache) == 0
    assert all(callback.value is None for callback in handler._ls_callbacks)


@pytest.mark.parametrize(
    "ls_callbacks, handler, tracking_client",
    [
        (
            [
                DummyScoreCallback(key="score1", compute_value=1.0, tracking_client=MagicMock()),
                DummyScoreCallback(key="score2", compute_value=2.0, tracking_client=MagicMock()),
            ],
            DummyScoreCallbackHandler(tracking_client=MagicMock()),
            MagicMock(),
        ),
        (
            [
                DummyScoreCallback(key="score1", compute_value=1.0, tracking_client=None),
                DummyScoreCallback(key="score2", compute_value=2.0, tracking_client=None),
            ],
            DummyScoreCallbackHandler(tracking_client=MagicMock()),
            MagicMock(),
        ),
        (
            [
                DummyScoreCallback(key="score1", compute_value=1.0, tracking_client=None),
                DummyScoreCallback(key="score2", compute_value=2.0, tracking_client=None),
            ],
            DummyScoreCallbackHandler(tracking_client=None),
            MagicMock(),
        ),
        (
            [
                DummyScoreCallback(key="score1", compute_value=1.0, tracking_client=None),
                DummyScoreCallback(key="score2", compute_value=2.0, tracking_client=None),
            ],
            DummyScoreCallbackHandler(tracking_client=None),
            None,
        ),
        (
            [
                DummyScoreCallback(key="score1", compute_value=1.0, tracking_client=MagicMock()),
                DummyScoreCallback(key="score2", compute_value=2.0, tracking_client=MagicMock()),
            ],
            DummyScoreCallbackHandler(tracking_client=MagicMock()),
            None,
        ),
        (
            [
                DummyArrayCallback(
                    key="array1", compute_value=np.array([1, 2]), tracking_client=MagicMock()
                ),
                DummyArrayCallback(
                    key="array2", compute_value=np.array([3, 4]), tracking_client=MagicMock()
                ),
            ],
            DummyArrayCallbackHandler(tracking_client=None),
            MagicMock(),
        ),
        (
            [
                DummyScoreCollectionCallback(
                    key="collection1", compute_value={"s1": 1.0}, tracking_client=MagicMock()
                ),
            ],
            DummyScoreCollectionCallbackHandler(tracking_client=None),
            MagicMock(),
        ),
    ],
)
def test_handler_tracking_client_setter(
    ls_callbacks: List[TrackingCallback[CallbackResources, ArtifactResult]],
    handler: TrackingCallbackHandler[
        TrackingCallback[CallbackResources, ArtifactResult], CallbackResources, ArtifactResult
    ],
    tracking_client: Optional[MagicMock],
):
    for callback in ls_callbacks:
        handler.add(callback=callback)
    handler.tracking_client = tracking_client
    assert handler.tracking_client == tracking_client
    assert handler.tracking_enabled == (tracking_client is not None)
    for callback in ls_callbacks:
        assert callback.tracking_client is None
        assert not callback.tracking_enabled
