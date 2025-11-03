from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar


@dataclass
class CallbackResources:
    pass


CallbackResourcesTContr = TypeVar(
    "CallbackResourcesTContr", bound=CallbackResources, contravariant=True
)


class Callback(ABC, Generic[CallbackResourcesTContr]):
    @abstractmethod
    def execute(self, resources: CallbackResourcesTContr): ...
