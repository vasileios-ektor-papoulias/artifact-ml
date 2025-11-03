from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, Sequence, TypeVar

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


CallbackTCov = TypeVar("CallbackTCov", bound=Callback, covariant=True)


class CallbackHandler(Generic[CallbackTCov, CallbackResourcesTContr]):
    _verbose = True
    _progressbar_message = "Executing Callbacks"

    def __init__(self, callbacks: Optional[Sequence[CallbackTCov]] = None):
        if callbacks is None:
            callbacks = []
        self._ls_callbacks = list(callbacks)

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


CallbackHandlerTCov = TypeVar(
    "CallbackHandlerTCov", bound=CallbackHandler[Any, Any], covariant=True
)


@dataclass(frozen=True)
class CallbackHandlerSuite(Generic[CallbackHandlerTCov]):
    score_handler: CallbackHandlerTCov
    array_handler: CallbackHandlerTCov
    plot_handler: CallbackHandlerTCov
    score_collection_handler: CallbackHandlerTCov
    array_collection_handler: CallbackHandlerTCov
    plot_collection_handler: CallbackHandlerTCov
