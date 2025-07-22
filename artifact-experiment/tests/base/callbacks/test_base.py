from typing import Callable, Literal

import pytest

from tests.base.callbacks.dummy.base import (
    AddCallback,
    DummyCallback,
    DummyCallbackResources,
    MultiplyCallback,
)

CallbackType = Literal["add", "multiply"]


@pytest.fixture
def resources_factory() -> Callable[[float, float], DummyCallbackResources]:
    def _factory(x: float, y: float) -> DummyCallbackResources:
        return DummyCallbackResources(x=x, y=y)

    return _factory


@pytest.fixture
def callback_factory() -> Callable[[CallbackType], DummyCallback]:
    def _factory(callback_type: CallbackType) -> DummyCallback:
        if callback_type == "add":
            return AddCallback()
        elif callback_type == "multiply":
            return MultiplyCallback()
        else:
            raise ValueError(f"Unknown callback type: {callback_type}")

    return _factory


@pytest.mark.parametrize(
    "callback_type, x, y, expected_result",
    [
        ("add", 1, 0, 1),
        ("add", 1, -1, 0),
        ("add", 1, 1, 2),
        ("multiply", 1, 0, 0),
        ("multiply", 1, 1, 1),
        ("multiply", 1, -1, -1),
        ("multiply", 2, 2, 4),
    ],
)
def test_callback(
    resources_factory: Callable[[float, float], DummyCallbackResources],
    callback_factory: Callable[[CallbackType], DummyCallback],
    callback_type: CallbackType,
    x: float,
    y: float,
    expected_result: float,
):
    resources = resources_factory(x, y)
    callback = callback_factory(callback_type)
    callback.execute(resources=resources)
    assert callback.result == expected_result
