from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from artifact_experiment._base.components.resources.base import CallbackResources

CallbackResourcesTContr = TypeVar(
    "CallbackResourcesTContr", bound=CallbackResources, contravariant=True
)


class Callback(ABC, Generic[CallbackResourcesTContr]):
    @abstractmethod
    def execute(self, resources: CallbackResourcesTContr): ...
