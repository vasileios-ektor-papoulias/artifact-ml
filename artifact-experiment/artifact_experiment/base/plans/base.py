from abc import ABC
from typing import Any, Dict, Generic, Optional, Sequence, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.callbacks.base import CallbackResources
from artifact_experiment.base.callbacks.tracking import TrackingCallback
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.handlers.base import CallbackHandlerSuite
from artifact_experiment.base.handlers.tracking import TrackingCallbackHandler
from artifact_experiment.base.tracking.client import TrackingClient

TrackingCallbackHandlerTCov = TypeVar(
    "TrackingCallbackHandlerTCov", bound=TrackingCallbackHandler[Any, Any, Any], covariant=True
)
TrackingCallbackTCov = TypeVar("TrackingCallbackTCov", bound=TrackingCallback, covariant=True)
CallbackResourcesTContr = TypeVar(
    "CallbackResourcesTContr", bound=CallbackResources, contravariant=True
)


class CallbackExecutionPlan(
    ABC, Generic[TrackingCallbackHandlerTCov, TrackingCallbackTCov, CallbackResourcesTContr]
):
    def __init__(
        self,
        callback_handlers: CallbackHandlerSuite[TrackingCallbackHandlerTCov],
        data_split: Optional[DataSplit] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        self._callback_handlers = callback_handlers
        self._data_split = data_split
        self._tracking_client = tracking_client

    @property
    def scores(self) -> Dict[str, float]:
        return self._score_handler.active_cache

    @property
    def arrays(self) -> Dict[str, ndarray]:
        return self._array_handler.active_cache

    @property
    def plots(self) -> Dict[str, Figure]:
        return self._plot_handler.active_cache

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        return self._score_collection_handler.active_cache

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return self._array_collection_handler.active_cache

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return self._plot_collection_handler.active_cache

    @property
    def data_split(self) -> Optional[DataSplit]:
        return self._data_split

    @property
    def tracking_client(self) -> Optional[TrackingClient]:
        return self._tracking_client

    @tracking_client.setter
    def tracking_client(self, tracking_client: Optional[TrackingClient]):
        self._tracking_client = tracking_client
        for handler in self._ls_handlers:
            handler.tracking_client = tracking_client

    @property
    def tracking_enabled(self) -> bool:
        return self.tracking_client is not None

    @property
    def _ls_handlers(self) -> Sequence[TrackingCallbackHandlerTCov]:
        return [
            self._score_handler,
            self._array_handler,
            self._plot_handler,
            self._score_collection_handler,
            self._array_collection_handler,
            self._plot_collection_handler,
        ]

    @property
    def _score_handler(self) -> TrackingCallbackHandlerTCov:
        return self._callback_handlers.score_handler

    @property
    def _array_handler(self) -> TrackingCallbackHandlerTCov:
        return self._callback_handlers.array_handler

    @property
    def _plot_handler(self) -> TrackingCallbackHandlerTCov:
        return self._callback_handlers.plot_handler

    @property
    def _score_collection_handler(self) -> TrackingCallbackHandlerTCov:
        return self._callback_handlers.score_collection_handler

    @property
    def _array_collection_handler(self) -> TrackingCallbackHandlerTCov:
        return self._callback_handlers.array_collection_handler

    @property
    def _plot_collection_handler(self) -> TrackingCallbackHandlerTCov:
        return self._callback_handlers.plot_collection_handler

    def execute(self, resources: CallbackResourcesTContr):
        for handler in self._ls_handlers:
            handler.execute(resources=resources)

    def clear_cache(self):
        for handler in self._ls_handlers:
            handler.clear()
