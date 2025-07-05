from typing import Dict, List, Optional, Union
from unittest.mock import MagicMock

import numpy as np
import pytest
from artifact_experiment.base.callbacks.base import CallbackResources

from tests.base.callbacks.dummy.tracking import (
    DummyArrayCallback,
    DummyArrayCallbackHandler,
    DummyScoreCallback,
    DummyScoreCallbackHandler,
    DummyScoreCollectionCallback,
    DummyScoreCollectionCallbackHandler,
    DummyTrackingCallback,
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
def test_set_tracking_client(
    initial_client: Optional[bool],
    new_client: Optional[bool],
    mock_tracking_client: MagicMock,
    callback_key: str,
):
    initial_tracking_client = mock_tracking_client if initial_client else None
    new_tracking_client = mock_tracking_client if new_client else None
    callback = DummyTrackingCallback(key=callback_key, tracking_client=initial_tracking_client)
    callback.set_tracking_client(tracking_client=new_tracking_client)
    assert callback._tracking_client == new_tracking_client


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
    test_resources: CallbackResources,
    mock_tracking_client: MagicMock,
):
    tracking_client = mock_tracking_client if has_tracking_client else None
    callback = DummyTrackingCallback(
        key=key, compute_value=compute_value, tracking_client=tracking_client
    )

    callback.execute(resources=test_resources)
    assert callback.key == key
    assert callback.value == compute_value
    if tracking_client is not None:
        tracking_client.log_score.assert_called_once_with(score=compute_value, name=key)


@pytest.mark.parametrize(
    "callback_class, compute_value, expected_log_method",
    [
        (DummyScoreCallback, 42.0, "log_score"),
        (DummyArrayCallback, np.array([1, 2, 3]), "log_array"),
        (DummyScoreCollectionCallback, {"score1": 1.0, "score2": 2.0}, "log_score_collection"),
    ],
)
def test_execute_concrete(
    callback_class: type,
    compute_value: Union[float, np.ndarray, Dict],
    expected_log_method: str,
    test_resources: CallbackResources,
    mock_tracking_client: MagicMock,
    callback_key: str,
):
    tracking_client = mock_tracking_client
    callback = callback_class(
        key=callback_key, compute_value=compute_value, tracking_client=tracking_client
    )

    callback.execute(resources=test_resources)
    log_method = getattr(tracking_client, expected_log_method)
    if expected_log_method == "log_score":
        log_method.assert_called_once_with(score=compute_value, name=callback_key)
    elif expected_log_method == "log_array":
        log_method.assert_called_once_with(array=compute_value, name=callback_key)
    elif expected_log_method == "log_score_collection":
        log_method.assert_called_once_with(score_collection=compute_value, name=callback_key)


@pytest.mark.parametrize(
    "handler_class, callback_class, compute_values, has_tracking_client",
    [
        (DummyScoreCallbackHandler, DummyScoreCallback, [1.0, 2.0, 3.0], True),
        (DummyScoreCallbackHandler, DummyScoreCallback, [1.0, 2.0], False),
        (DummyArrayCallbackHandler, DummyArrayCallback, [np.array([1]), np.array([2, 3])], True),
        (
            DummyScoreCollectionCallbackHandler,
            DummyScoreCollectionCallback,
            [{"score1": 1.0}, {"score2": 2.0}, {"score3": 3.0}],
            False,
        ),
    ],
)
def test_handler_execute(
    mock_tracking_client: MagicMock,
    test_resources: CallbackResources,
    multiple_callback_keys: List[str],
    handler_class: type,
    callback_class: type,
    compute_values: List,
    has_tracking_client: bool,
):
    tracking_client = mock_tracking_client if has_tracking_client else None
    callbacks = [
        callback_class(key=key, compute_value=value, tracking_client=tracking_client)
        for key, value in zip(multiple_callback_keys, compute_values)
    ]
    handler = handler_class(ls_callbacks=callbacks, tracking_client=tracking_client)
    handler.execute(resources=test_resources)
    assert len(handler.active_cache) == len(compute_values)
    for key, expected_value in zip(multiple_callback_keys, compute_values):
        if isinstance(expected_value, np.ndarray):
            assert np.array_equal(handler.active_cache[key], expected_value)
        else:
            assert handler.active_cache[key] == expected_value
    if tracking_client is not None:
        if callback_class == DummyScoreCallback:
            assert tracking_client.log_score.call_count == len(compute_values)
            for key, value in zip(multiple_callback_keys, compute_values):
                tracking_client.log_score.assert_any_call(score=value, name=key)
        elif callback_class == DummyArrayCallback:
            assert tracking_client.log_array.call_count == len(compute_values)
        elif callback_class == DummyScoreCollectionCallback:
            assert tracking_client.log_score_collection.call_count == len(compute_values)
    handler.clear()
    assert len(handler.active_cache) == 0
    assert all(callback.value is None for callback in callbacks)
