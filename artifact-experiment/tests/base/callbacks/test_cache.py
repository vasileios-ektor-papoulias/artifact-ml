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
    def _factory(callback_type: str, key: str) -> CacheCallback:
        if callback_type == "add":
            return AddCacheCallback(key=key)
        elif callback_type == "multiply":
            return MultiplyCacheCallback(key=key)
        else:
            raise ValueError(f"Unknown callback type: {callback_type}")

    return _factory


@pytest.mark.parametrize(
    "callback_key, x, y",
    [
        ("test_callback", 1.0, 1.0),
        ("another_callback", 0.0, 5.0),
        ("third_callback", -1.0, 2.0),
    ],
)
def test_callback_key(
    callback_key: str,
    x: float,
    y: float,
    resources_factory: Callable,
):
    callback = AddCacheCallback(key=callback_key)
    resources = resources_factory(x, y)
    assert callback.key == callback_key
    callback.execute(resources=resources)
    assert callback.key == callback_key
    assert callback_key in callback.cache


@pytest.mark.parametrize(
    "callback_type, x, y, expected_result",
    [
        ("add", 1.0, 2.0, 3.0),
        ("add", 0.0, 0.0, 0.0),
        ("add", -1.0, 1.0, 0.0),
        ("multiply", 2.0, 3.0, 6.0),
        ("multiply", 0.0, 5.0, 0.0),
        ("multiply", -2.0, 3.0, -6.0),
    ],
)
def test_callback_execute(
    callback_type: str,
    x: float,
    y: float,
    expected_result: float,
    callback_factory: Callable[[str, str], CacheCallback],
    resources_factory: Callable[[float, float], DummyCallbackResources],
):
    callback = callback_factory(callback_type, "test_key")
    resources = resources_factory(x, y)
    assert callback.value is None
    callback.execute(resources=resources)
    assert callback.value == expected_result
    assert callback.cache["test_key"] == expected_result


@pytest.mark.parametrize(
    "callback, resources, expected_result",
    [
        (AddCacheCallback(key="add_result"), DummyCallbackResources(x=1, y=0), 1),
        (MultiplyCacheCallback(key="multiply_result"), DummyCallbackResources(x=1, y=0), 0),
    ],
)
def test_execute_concrete(
    callback: CacheCallback[DummyCallbackResources, float],
    resources: DummyCallbackResources,
    expected_result: float,
):
    callback.execute(resources=resources)
    assert callback.value == expected_result


@pytest.mark.parametrize(
    "keys, callback_types, x_values, y_values, expected_results",
    [
        (["add1", "add2"], ["add", "add"], [1.0, 2.0], [1.0, 3.0], [2.0, 2.0]),
        (
            ["mult1", "mult2", "mult3"],
            ["multiply", "multiply", "multiply"],
            [2.0, 3.0, 0.0],
            [3.0, 4.0, 5.0],
            [6.0, 6.0, 6.0],
        ),
        (["mixed1", "mixed2"], ["add", "multiply"], [1.0, 2.0], [2.0, 3.0], [3.0, 2.0]),
    ],
)
def test_handler_execute(
    callback_factory: Callable[[str, str], CacheCallback],
    resources_factory: Callable[[float, float], DummyCallbackResources],
    keys: List[str],
    callback_types: List[str],
    x_values: List[float],
    y_values: List[float],
    expected_results: List[float],
):
    callbacks = [callback_factory(cb_type, key) for cb_type, key in zip(callback_types, keys)]
    handler = DummyCacheCallbackHandler(ls_callbacks=callbacks)
    assert len(handler.active_cache) == 0
    test_resources = resources_factory(x_values[0], y_values[0])
    handler.execute(resources=test_resources)
    assert len(handler.active_cache) == len(callbacks)
    for key, expected_result in zip(keys, expected_results):
        assert key in handler.active_cache
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
    num_callbacks: int,
    clear_after_execution: bool,
    callback_factory: Callable[[str, str], CacheCallback],
    resources_factory: Callable[[float, float], DummyCallbackResources],
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


def test_handler_update_cache():
    callback1 = AddCacheCallback(key="test1")
    callback2 = MultiplyCacheCallback(key="test2")
    handler = DummyCacheCallbackHandler(ls_callbacks=[callback1, callback2])
    resources = DummyCallbackResources(x=2.0, y=3.0)
    callback1.execute(resources=resources)
    callback2.execute(resources=resources)
    assert len(handler.active_cache) == 0
    handler.update_cache()
    assert len(handler.active_cache) == 2
    assert handler.active_cache["test1"] == 5.0
    assert handler.active_cache["test2"] == 6.0
