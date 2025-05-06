from artifact_experiment.base.callbacks.cache import CacheCallback, CacheCallbackHandler

from tests.base.callbacks.dummy.base import (
    DummyCallbackResources,
)


class AddCacheCallback(CacheCallback[DummyCallbackResources, float]):
    def _compute(self, resources: DummyCallbackResources) -> float:
        result = resources.x + resources.y
        return result


class MultiplyCacheCallback(CacheCallback[DummyCallbackResources, float]):
    def _compute(self, resources: DummyCallbackResources) -> float:
        result = resources.x * resources.y
        return result


DummyCacheCallbackHandler = CacheCallbackHandler[
    CacheCallback[DummyCallbackResources, float], DummyCallbackResources, float
]
