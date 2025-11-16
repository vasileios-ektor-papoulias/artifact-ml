from typing import Any, Generic, List, Mapping, Sequence, Type, TypeVar

from artifact_core.typing import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.components.callbacks.cache import (
    CacheCallback,
    CacheCallbackResources,
)
from artifact_experiment._base.components.handlers.cache import CacheCallbackHandler

CacheCallbackHandlerTCov = TypeVar(
    "CacheCallbackHandlerTCov", bound=CacheCallbackHandler[Any, Any, Any], covariant=True
)
CacheCallbackTCov = TypeVar("CacheCallbackTCov", bound=CacheCallback[Any, Any], covariant=True)
CacheCallbackResourcesTContr = TypeVar(
    "CacheCallbackResourcesTContr", bound=CacheCallbackResources, contravariant=True
)
CallbackHandlerSuiteT = TypeVar(
    "CallbackHandlerSuiteT", bound="CallbackHandlerSuite[Any, Any, Any]"
)


class CallbackHandlerSuite(
    Generic[CacheCallbackHandlerTCov, CacheCallbackTCov, CacheCallbackResourcesTContr]
):
    def __init__(
        self,
        score_handler: CacheCallbackHandlerTCov,
        array_handler: CacheCallbackHandlerTCov,
        plot_handler: CacheCallbackHandlerTCov,
        score_collection_handler: CacheCallbackHandlerTCov,
        array_collection_handler: CacheCallbackHandlerTCov,
        plot_collection_handler: CacheCallbackHandlerTCov,
    ):
        self._score_handler = score_handler
        self._array_handler = array_handler
        self._plot_handler = plot_handler
        self._score_collection_handler = score_collection_handler
        self._array_collection_handler = array_collection_handler
        self._plot_collection_handler = plot_collection_handler

    @classmethod
    def build(
        cls: Type[CallbackHandlerSuiteT],
        score_callbacks: Sequence[CacheCallbackTCov],
        array_callbacks: Sequence[CacheCallbackTCov],
        plot_callbacks: Sequence[CacheCallbackTCov],
        score_collection_callbacks: Sequence[CacheCallbackTCov],
        array_collection_callbacks: Sequence[CacheCallbackTCov],
        plot_collection_callbacks: Sequence[CacheCallbackTCov],
    ) -> CallbackHandlerSuiteT:
        handler_suite = cls(
            score_handler=CacheCallbackHandler(callbacks=score_callbacks),
            array_handler=CacheCallbackHandler(callbacks=array_callbacks),
            plot_handler=CacheCallbackHandler(callbacks=plot_callbacks),
            score_collection_handler=CacheCallbackHandler(callbacks=score_collection_callbacks),
            array_collection_handler=CacheCallbackHandler(callbacks=array_collection_callbacks),
            plot_collection_handler=CacheCallbackHandler(callbacks=plot_collection_callbacks),
        )
        return handler_suite

    @property
    def scores(self) -> Mapping[str, Score]:
        return self._score_handler.active_cache

    @property
    def arrays(self) -> Mapping[str, Array]:
        return self._array_handler.active_cache

    @property
    def plots(self) -> Mapping[str, Plot]:
        return self._plot_handler.active_cache

    @property
    def score_collections(self) -> Mapping[str, ScoreCollection]:
        return self._score_collection_handler.active_cache

    @property
    def array_collections(self) -> Mapping[str, ArrayCollection]:
        return self._array_collection_handler.active_cache

    @property
    def plot_collections(self) -> Mapping[str, PlotCollection]:
        return self._plot_collection_handler.active_cache

    @property
    def _ls_handlers(self) -> List[CacheCallbackHandlerTCov]:
        return [
            self._score_handler,
            self._array_handler,
            self._plot_handler,
            self._score_collection_handler,
            self._array_collection_handler,
            self._plot_collection_handler,
        ]

    def execute(self, resources: CacheCallbackResourcesTContr):
        for handler in self._ls_handlers:
            handler.execute(resources=resources)

    def clear_cache(self):
        for handler in self._ls_handlers:
            handler.clear()
