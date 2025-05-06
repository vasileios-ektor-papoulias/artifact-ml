import pytest

from tests.base.callbacks.dummy.base import (
    AddCallback,
    DummyCallback,
    DummyCallbackResources,
    MultiplyCallback,
)


@pytest.mark.parametrize(
    "callback, resources, expected_result",
    [
        (AddCallback(), DummyCallbackResources(x=1, y=0), 1),
        (MultiplyCallback(), DummyCallbackResources(x=1, y=0), 0),
    ],
)
def test_callback(
    callback: DummyCallback, resources: DummyCallbackResources, expected_result: float
):
    callback.execute(resources=resources)
    assert callback.result == expected_result
