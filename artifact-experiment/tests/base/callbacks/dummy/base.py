from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from artifact_experiment.base.callbacks.base import Callback, CallbackHandler, CallbackResources


@dataclass
class DummyCallbackResources(CallbackResources):
    x: float
    y: float


class DummyCallback(Callback[DummyCallbackResources]):
    def __init__(self):
        super().__init__()
        self._result = None

    @property
    def result(self) -> Optional[float]:
        return self._result

    @abstractmethod
    def execute(self, resources: DummyCallbackResources): ...


class AddCallback(DummyCallback):
    def execute(self, resources: DummyCallbackResources):
        self._result = resources.x + resources.y


class MultiplyCallback(DummyCallback):
    def execute(self, resources: DummyCallbackResources):
        self._result = resources.x * resources.y


DummyCallbackHandler = CallbackHandler[DummyCallback, DummyCallbackResources]
