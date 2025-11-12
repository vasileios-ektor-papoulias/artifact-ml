from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Mapping, Sequence, Type, TypeVar

from artifact_experiment.base.components.callbacks.tracking import (
    TrackingCallback,
    TrackingCallbackResources,
)
from artifact_experiment.base.components.handler_suites.tracking import TrackingCallbackHandlerSuite
from artifact_experiment.base.components.plans.build_context import PlanBuildContext

TrackingCallbackHandlerSuiteTCov = TypeVar(
    "TrackingCallbackHandlerSuiteTCov",
    bound=TrackingCallbackHandlerSuite[Any, Any, Any],
    covariant=True,
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
        TrackingCallbackHandlerSuiteTCov,
        TrackingCallbackTCov,
        TrackingCallbackResourcesTContr,
        PlanBuildContextTContr,
    ],
):
    def __init__(
        self, handler_suite: TrackingCallbackHandlerSuiteTCov, context: PlanBuildContextTContr
    ):
        self._handler_suite = handler_suite
        self._context = context

    @classmethod
    def build(cls: Type[PlanT], context: PlanBuildContextTContr) -> PlanT:
        handler_suite = cls._build_handler_suite(context=context)
        plan = cls(handler_suite=handler_suite, context=context)
        return plan

    @property
    def scores(self) -> Mapping[str, float]:
        return self._handler_suite.scores

    @property
    def arrays(self) -> Mapping[str, Array]:
        return self._handler_suite.arrays

    @property
    def plots(self) -> Mapping[str, Figure]:
        return self._handler_suite.plots

    @property
    def score_collections(self) -> Mapping[str, Dict[str, float]]:
        return self._handler_suite.score_collections

    @property
    def array_collections(self) -> Mapping[str, Dict[str, Array]]:
        return self._handler_suite.array_collections

    @property
    def plot_collections(self) -> Mapping[str, Dict[str, Figure]]:
        return self._handler_suite.plot_collections

    @classmethod
    @abstractmethod
    def _get_handler_suite(cls) -> Type[TrackingCallbackHandlerSuiteTCov]: ...

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

    @classmethod
    def _build_handler_suite(
        cls, context: PlanBuildContextTContr
    ) -> TrackingCallbackHandlerSuiteTCov:
        score_callbacks = cls._get_score_callbacks(context=context)
        array_callbacks = cls._get_array_callbacks(context=context)
        plot_callbacks = cls._get_plot_callbacks(context=context)
        score_collection_callbacks = cls._get_score_collection_callbacks(context=context)
        array_collection_callbacks = cls._get_array_collection_callbacks(context=context)
        plot_collection_callbacks = cls._get_plot_collection_callbacks(context=context)
        handler_suite_class = cls._get_handler_suite()
        handler_suite = handler_suite_class.build(
            score_callbacks=score_callbacks,
            array_callbacks=array_callbacks,
            plot_callbacks=plot_callbacks,
            score_collection_callbacks=score_collection_callbacks,
            array_collection_callbacks=array_collection_callbacks,
            plot_collection_callbacks=plot_collection_callbacks,
        )
        return handler_suite
