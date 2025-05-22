import pytest
from artifact_experiment.base.callbacks.cache import CacheCallback

from tests.base.callbacks.dummy.cache import (
    AddCacheCallback,
    DummyCallbackResources,
    MultiplyCacheCallback,
)


@pytest.mark.parametrize(
    "callback, resources, expected_result",
    [
        (AddCacheCallback(key="add_result"), DummyCallbackResources(x=1, y=0), 1),
        (MultiplyCacheCallback(key="multiply_result"), DummyCallbackResources(x=1, y=0), 0),
    ],
)
def test_callback(
    callback: CacheCallback[DummyCallbackResources, float],
    resources: DummyCallbackResources,
    expected_result: float,
):
    callback.execute(resources=resources)
    assert callback.value == expected_result
