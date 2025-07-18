from typing import Callable, List

import pytest
from artifact_experiment.base.callbacks.cache import CacheCallback

from tests.base.callbacks.dummy.cache import (
    AddCacheCallback,
    DummyCacheCallbackHandler,
    DummyCallbackResources,
    MultiplyCacheCallback,
)


@pytest.fixture
def resources_factory() -> Callable[[float, float], DummyCallbackResources]:
    def _factory(x: float = 1.0, y: float = 2.0) -> DummyCallbackResources:
        return DummyCallbackResources(x=x, y=y)

    return _factory


@pytest.fixture
def callback_factory() -> Callable[[str, str], CacheCallback]:
    def _factory(callback_type: str, callback_key: str) -> CacheCallback:
        if callback_type == "add":
            return AddCacheCallback(key=callback_key)
        elif callback_type == "multiply":
            return MultiplyCacheCallback(key=callback_key)
        else:
            raise ValueError(f"Unknown callback type: {callback_type}")

    return _factory


@pytest.mark.parametrize(
    "callback_type, callback_key",
    [
        ("add", "test_callback"),
        ("add", "another_callback"),
        ("add", "yet_another_callback"),
    ],
)
def test_callback_key(
    callback_factory: Callable[[str, str], CacheCallback],
    callback_type: str,
    callback_key: str,
):
    callback = callback_factory(callback_type, callback_key)
    assert isinstance(callback.key, str)
    assert callback.key == callback_key
    assert len(callback.cache) == 1
    assert callback.key in callback.cache
    assert callback.cache[callback_key] is None


@pytest.mark.parametrize(
    "callback_type, callback_key, x, y, expected_result",
    [
        ("add", "test_key", 1.0, 2.0, 3.0),
        ("add", "test_key", 0.0, 0.0, 0.0),
        ("add", "test_key", -1.0, 1.0, 0.0),
        ("multiply", "test_key", 2.0, 3.0, 6.0),
        ("multiply", "test_key", 0.0, 5.0, 0.0),
        ("multiply", "test_key", -2.0, 3.0, -6.0),
    ],
)
def test_callback_execute(
    resources_factory: Callable[[float, float], DummyCallbackResources],
    callback_factory: Callable[[str, str], CacheCallback],
    callback_type: str,
    callback_key: str,
    x: float,
    y: float,
    expected_result: float,
):
    callback = callback_factory(callback_type, callback_key)
    resources = resources_factory(x, y)
    assert callback.value is None
    callback.execute(resources=resources)
    assert callback.value == expected_result
    assert callback.cache[callback_key] == expected_result


@pytest.mark.parametrize(
    "clear_after_execution",
    [
        (True),
        (True),
        (False),
        (True),
    ],
)
def test_callback_clear(
    resources_factory: Callable[[float, float], DummyCallbackResources],
    callback_factory: Callable[[str, str], CacheCallback],
    clear_after_execution: bool,
):
    callback = callback_factory("add", "key")
    assert callback.value is None
    resources = resources_factory(1.0, 1.0)
    callback.execute(resources=resources)
    assert callback.value is not None
    if clear_after_execution:
        callback.clear()
        assert callback.value is None


@pytest.mark.parametrize(
    "ls_callback_types, ls_callback_keys, x, y, ls_expected_results",
    [
        (["add", "add"], ["add1", "add2"], 1.0, 1.0, [2.0, 2.0]),
        (
            ["multiply", "multiply", "multiply"],
            ["mult1", "mult2", "mult3"],
            2.0,
            3.0,
            [6.0, 6.0, 6.0],
        ),
        (["add", "multiply"], ["mixed1", "mixed2"], 1.0, 2.0, [3.0, 2.0]),
    ],
)
def test_handler_execute(
    resources_factory: Callable[[float, float], DummyCallbackResources],
    callback_factory: Callable[[str, str], CacheCallback],
    ls_callback_types: List[str],
    ls_callback_keys: List[str],
    x: float,
    y: float,
    ls_expected_results: List[float],
):
    ls_callbacks = [
        callback_factory(cb_type, cb_key)
        for cb_type, cb_key in zip(ls_callback_types, ls_callback_keys)
    ]
    handler = DummyCacheCallbackHandler(ls_callbacks=ls_callbacks)
    assert len(handler.cache) == len(ls_callbacks)
    for key in ls_callback_keys:
        assert key in handler.cache
        assert handler.cache[key] is None
    assert len(handler.active_cache) == 0
    test_resources = resources_factory(x, y)
    handler.execute(resources=test_resources)
    assert len(handler.active_cache) == len(ls_callbacks)
    for key, expected_result in zip(ls_callback_keys, ls_expected_results):
        assert key in handler.active_cache
        assert handler.active_cache[key] is not None
        assert handler.active_cache[key] == expected_result


@pytest.mark.parametrize(
    "num_callbacks, clear_after_execution",
    [
        (1, True),
        (3, True),
        (5, False),
        (2, True),
    ],
)
def test_handler_clear(
    resources_factory: Callable[[float, float], DummyCallbackResources],
    callback_factory: Callable[[str, str], CacheCallback],
    num_callbacks: int,
    clear_after_execution: bool,
):
    callbacks = [callback_factory("add", f"key_{i}") for i in range(num_callbacks)]
    handler = DummyCacheCallbackHandler(ls_callbacks=callbacks)
    resources = resources_factory(1.0, 1.0)
    handler.execute(resources=resources)
    assert len(handler.active_cache) == num_callbacks
    if clear_after_execution:
        handler.clear()
        assert len(handler.active_cache) == 0
        assert all(callback.value is None for callback in callbacks)
