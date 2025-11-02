from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Sequence, Type, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.callbacks.base import CallbackResources
from artifact_experiment.base.callbacks.tracking import TrackingCallback, TrackingCallbackHandler
from artifact_experiment.base.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient

CallbackResourcesTContr = TypeVar(
    "CallbackResourcesTContr", bound=CallbackResources, contravariant=True
)
TrackingCallbackT = TypeVar("TrackingCallbackT", bound=TrackingCallback)
TrackingCallbackHandlerT = TypeVar(
    "TrackingCallbackHandlerT", bound=TrackingCallbackHandler[Any, Any, Any]
)

CallbackHandlerCollectionT = TypeVar(
    "CallbackHandlerCollectionT", bound="CallbackHandlerCollection"
)


@dataclass
class CallbackHandlerCollection(
    ABC, Generic[TrackingCallbackHandlerT, TrackingCallbackT, CallbackResourcesTContr]
):
    score_handler: TrackingCallbackHandlerT
    array_handler: TrackingCallbackHandlerT
    plot_handler: TrackingCallbackHandlerT
    score_collection_handler: TrackingCallbackHandlerT
    array_collection_handler: TrackingCallbackHandlerT
    plot_collection_handler: TrackingCallbackHandlerT

    @classmethod
    @abstractmethod
    def build(
        cls: Type[CallbackHandlerCollectionT],
        score_callbacks: Sequence[TrackingCallbackT],
        array_callbacks: Sequence[TrackingCallbackT],
        plot_callbacks: Sequence[TrackingCallbackT],
        score_collection_callbacks: Sequence[TrackingCallbackT],
        array_collection_callbacks: Sequence[TrackingCallbackT],
        plot_collection_callbacks: Sequence[TrackingCallbackT],
        tracking_client: Optional[TrackingClient] = None,
    ) -> CallbackHandlerCollectionT: ...


CallbackExecutionPlanT = TypeVar("CallbackExecutionPlanT", bound="CallbackExecutionPlan")


class CallbackExecutionPlan(
    ABC, Generic[TrackingCallbackHandlerT, TrackingCallbackT, CallbackResourcesTContr]
):
    def __init__(
        self,
        handlers: CallbackHandlerCollection[
            TrackingCallbackHandlerT, TrackingCallbackT, CallbackResourcesTContr
        ],
        data_split: Optional[DataSplit] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        self._handlers = handlers
        self._data_split = data_split
        self._tracking_client = tracking_client

    @classmethod
    @abstractmethod
    def build(
        cls: Type[CallbackExecutionPlanT],
        data_split: Optional[DataSplit] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> CallbackExecutionPlanT:
        plan = cls._build(
            score_callbacks=cls._get_score_callbacks(
                data_split=data_split, tracking_client=tracking_client
            ),
            array_callbacks=cls._get_array_callbacks(
                data_split=data_split, tracking_client=tracking_client
            ),
            plot_callbacks=cls._get_plot_callbacks(
                data_split=data_split, tracking_client=tracking_client
            ),
            score_collection_callbacks=cls._get_score_collection_callbacks(
                data_split=data_split, tracking_client=tracking_client
            ),
            array_collection_callbacks=cls._get_array_collection_callbacks(
                data_split=data_split, tracking_client=tracking_client
            ),
            plot_collection_callbacks=cls._get_plot_collection_callbacks(
                data_split=data_split, tracking_client=tracking_client
            ),
            data_split=data_split,
            tracking_client=tracking_client,
        )
        return plan

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
    def scores_collections(self) -> Dict[str, Dict[str, float]]:
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
    def _ls_handlers(self) -> Sequence[TrackingCallbackHandlerT]:
        return [
            self._score_handler,
            self._array_handler,
            self._plot_handler,
            self._score_collection_handler,
            self._array_collection_handler,
            self._plot_collection_handler,
        ]

    @property
    def _score_handler(self) -> TrackingCallbackHandlerT:
        return self._handlers.score_handler

    @property
    def _array_handler(self) -> TrackingCallbackHandlerT:
        return self._handlers.array_handler

    @property
    def _plot_handler(self) -> TrackingCallbackHandlerT:
        return self._handlers.plot_handler

    @property
    def _score_collection_handler(self) -> TrackingCallbackHandlerT:
        return self._handlers.score_collection_handler

    @property
    def _array_collection_handler(self) -> TrackingCallbackHandlerT:
        return self._handlers.array_collection_handler

    @property
    def _plot_collection_handler(self) -> TrackingCallbackHandlerT:
        return self._handlers.plot_collection_handler

    @classmethod
    @abstractmethod
    def _build(
        cls: Type[CallbackExecutionPlanT],
        score_callbacks: Sequence[TrackingCallbackT],
        array_callbacks: Sequence[TrackingCallbackT],
        plot_callbacks: Sequence[TrackingCallbackT],
        score_collection_callbacks: Sequence[TrackingCallbackT],
        array_collection_callbacks: Sequence[TrackingCallbackT],
        plot_collection_callbacks: Sequence[TrackingCallbackT],
        data_split: Optional[DataSplit],
        tracking_client: Optional[TrackingClient],
    ) -> CallbackExecutionPlanT: ...

    @staticmethod
    @abstractmethod
    def _get_score_callbacks(
        data_split: Optional[DataSplit], tracking_client: Optional[TrackingClient]
    ) -> Sequence[TrackingCallbackT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_callbacks(
        data_split: Optional[DataSplit], tracking_client: Optional[TrackingClient]
    ) -> Sequence[TrackingCallbackT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_callbacks(
        data_split: Optional[DataSplit], tracking_client: Optional[TrackingClient]
    ) -> Sequence[TrackingCallbackT]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_callbacks(
        data_split: Optional[DataSplit], tracking_client: Optional[TrackingClient]
    ) -> Sequence[TrackingCallbackT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_callbacks(
        data_split: Optional[DataSplit], tracking_client: Optional[TrackingClient]
    ) -> Sequence[TrackingCallbackT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_callbacks(
        data_split: Optional[DataSplit], tracking_client: Optional[TrackingClient]
    ) -> Sequence[TrackingCallbackT]: ...

    def execute(self, resources: CallbackResourcesTContr):
        for handler in self._ls_handlers:
            handler.execute(resources=resources)
