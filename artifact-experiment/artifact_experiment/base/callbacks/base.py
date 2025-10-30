from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from tqdm import tqdm


@dataclass
class CallbackResources:
    pass


CallbackResourcesTContr = TypeVar(
    "CallbackResourcesTContr", bound=CallbackResources, contravariant=True
)


class Callback(ABC, Generic[CallbackResourcesTContr]):
    @abstractmethod
    def execute(self, resources: CallbackResourcesTContr): ...


CallbackT = TypeVar("CallbackT", bound=Callback)


class CallbackHandler(Generic[CallbackT, CallbackResourcesTContr]):
    _verbose = True
    _progressbar_message = "Executing Callbacks"

    def __init__(self, ls_callbacks: Optional[List[CallbackT]] = None):
        if ls_callbacks is None:
            ls_callbacks = []
        self._ls_callbacks = ls_callbacks

    @property
    def ls_callbacks(self) -> List[CallbackT]:
        return self._ls_callbacks

    @property
    def n_callbacks(self) -> int:
        return self._n_callbacks

    @property
    def has_callbacks(self) -> bool:
        return self._has_callbacks

    @property
    def _n_callbacks(self) -> int:
        return len(self._ls_callbacks)

    @property
    def _has_callbacks(self) -> bool:
        return self._n_callbacks != 0

    def execute(self, resources: CallbackResourcesTContr):
        for callback in tqdm(
            self._ls_callbacks,
            desc=self._progressbar_message,
            disable=self._verbose,
            leave=False,
        ):
            callback.execute(resources=resources)

    def add(self, callback: CallbackT):
        self._ls_callbacks.append(callback)
