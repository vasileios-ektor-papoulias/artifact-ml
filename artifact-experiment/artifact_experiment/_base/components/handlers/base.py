from typing import Any, Generic, Optional, Sequence, TypeVar

from tqdm import tqdm

from artifact_experiment._base.components.callbacks.base import Callback, CallbackResources

CallbackResourcesTContr = TypeVar(
    "CallbackResourcesTContr", bound=CallbackResources, contravariant=True
)
CallbackTCov = TypeVar("CallbackTCov", bound=Callback[Any], covariant=True)


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
