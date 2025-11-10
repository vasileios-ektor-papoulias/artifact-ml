from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Mapping, Optional, Sequence, Type, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.components.callbacks.tracking import (
    TrackingCallback,
    TrackingCallbackResources,
)
from artifact_experiment.base.components.handler_suites.base import CallbackHandlerSuite
from artifact_experiment.base.tracking.background.tracking_queue import TrackingQueue
from artifact_experiment.base.tracking.background.writer import (
    ArrayCollectionWriter,
    ArrayWriter,
    PlotCollectionWriter,
    PlotWriter,
    ScoreCollectionWriter,
    ScoreWriter,
)


@dataclass(frozen=True)
class PlanBuildContext:
    tracking_queue: Optional[TrackingQueue]

    @property
    def score_writer(self) -> Optional[ScoreWriter]:
        return self.tracking_queue.score_writer if self.tracking_queue is not None else None

    @property
    def array_writer(self) -> Optional[ArrayWriter]:
        return self.tracking_queue.array_writer if self.tracking_queue is not None else None

    @property
    def plot_writer(self) -> Optional[PlotWriter]:
        return self.tracking_queue.plot_writer if self.tracking_queue is not None else None

    @property
    def score_collection_writer(self) -> Optional[ScoreCollectionWriter]:
        return (
            self.tracking_queue.score_collection_writer if self.tracking_queue is not None else None
        )

    @property
    def array_collection_writer(self) -> Optional[ArrayCollectionWriter]:
        return (
            self.tracking_queue.array_collection_writer if self.tracking_queue is not None else None
        )

    @property
    def plot_collection_writer(self) -> Optional[PlotCollectionWriter]:
        return (
            self.tracking_queue.plot_collection_writer if self.tracking_queue is not None else None
        )


CallbackHandlerSuiteTCov = TypeVar(
    "CallbackHandlerSuiteTCov", bound=CallbackHandlerSuite[Any, Any, Any], covariant=True
)
TrackingCallbackTCov = TypeVar("TrackingCallbackTCov", bound=TrackingCallback, covariant=True)
TrackingCallbackResourcesTContr = TypeVar(
    "TrackingCallbackResourcesTContr", bound=TrackingCallbackResources, contravariant=True
)
PlanBuildContextTContr = TypeVar(
    "PlanBuildContextTContr", bound=PlanBuildContext, contravariant=True
)
PlanT = TypeVar("PlanT", bound="CallbackExecutionPlan")


class CallbackExecutionPlan(
    ABC,
    Generic[
        CallbackHandlerSuiteTCov,
        TrackingCallbackTCov,
        TrackingCallbackResourcesTContr,
        PlanBuildContextTContr,
    ],
):
    def __init__(self, handler_suite: CallbackHandlerSuiteTCov, context: PlanBuildContextTContr):
        self._handler_suite = handler_suite
        self._context = context

    @classmethod
    def build(cls: Type[PlanT], context: PlanBuildContextTContr) -> PlanT:
        score_callbacks = cls._get_score_callbacks(context=context)
        array_callbacks = cls._get_array_callbacks(context=context)
        plot_callbacks = cls._get_plot_callbacks(context=context)
        score_collection_callbacks = cls._get_score_collection_callbacks(context=context)
        array_collection_callbacks = cls._get_array_collection_callbacks(context=context)
        plot_collection_callbacks = cls._get_plot_collection_callbacks(context=context)
        handler_suite = cls._get_handler_suite_type().build(
            score_callbacks=score_callbacks,
            array_callbacks=array_callbacks,
            plot_callbacks=plot_callbacks,
            score_collection_callbacks=score_collection_callbacks,
            array_collection_callbacks=array_collection_callbacks,
            plot_collection_callbacks=plot_collection_callbacks,
        )
        plan = cls(handler_suite=handler_suite, context=context)
        return plan

    @property
    def scores(self) -> Mapping[str, float]:
        return self._handler_suite.scores

    @property
    def arrays(self) -> Mapping[str, ndarray]:
        return self._handler_suite.arrays

    @property
    def plots(self) -> Mapping[str, Figure]:
        return self._handler_suite.plots

    @property
    def score_collections(self) -> Mapping[str, Dict[str, float]]:
        return self._handler_suite.score_collections

    @property
    def array_collections(self) -> Mapping[str, Dict[str, ndarray]]:
        return self._handler_suite.array_collections

    @property
    def plot_collections(self) -> Mapping[str, Dict[str, Figure]]:
        return self._handler_suite.plot_collections

    @classmethod
    @abstractmethod
    def _get_handler_suite_type(cls) -> Type[CallbackHandlerSuiteTCov]: ...

    @classmethod
    @abstractmethod
    def _get_score_callbacks(
        cls, context: PlanBuildContextTContr
    ) -> Sequence[TrackingCallbackTCov]: ...

    @classmethod
    @abstractmethod
    def _get_array_callbacks(
        cls, context: PlanBuildContextTContr
    ) -> Sequence[TrackingCallbackTCov]: ...

    @classmethod
    @abstractmethod
    def _get_plot_callbacks(
        cls, context: PlanBuildContextTContr
    ) -> Sequence[TrackingCallbackTCov]: ...

    @classmethod
    @abstractmethod
    def _get_score_collection_callbacks(
        cls, context: PlanBuildContextTContr
    ) -> Sequence[TrackingCallbackTCov]: ...

    @classmethod
    @abstractmethod
    def _get_array_collection_callbacks(
        cls, context: PlanBuildContextTContr
    ) -> Sequence[TrackingCallbackTCov]: ...

    @classmethod
    @abstractmethod
    def _get_plot_collection_callbacks(
        cls, context: PlanBuildContextTContr
    ) -> Sequence[TrackingCallbackTCov]: ...

    def execute(self, resources: TrackingCallbackResourcesTContr):
        self._handler_suite.execute(resources=resources)

    def clear_cache(self):
        self._handler_suite.clear_cache()
