from typing import Any, Sequence, Type, TypeVar

from artifact_experiment._base.components.callbacks.tracking import (
    TrackingCallback,
    TrackingCallbackResources,
)
from artifact_experiment._base.components.handler_suites.base import CallbackHandlerSuite
from artifact_experiment._base.components.handlers.tracking import TrackingCallbackHandler

TrackingCallbackHandlerTCov = TypeVar(
    "TrackingCallbackHandlerTCov", bound=TrackingCallbackHandler[Any, Any, Any], covariant=True
)
TrackingCallbackTCov = TypeVar(
    "TrackingCallbackTCov", bound=TrackingCallback[Any, Any], covariant=True
)
TrackingCallbackResourcesTContr = TypeVar(
    "TrackingCallbackResourcesTContr", bound=TrackingCallbackResources, contravariant=True
)
TrackingCallbackHandlerSuiteT = TypeVar(
    "TrackingCallbackHandlerSuiteT", bound="TrackingCallbackHandlerSuite[Any, Any, Any]"
)


class TrackingCallbackHandlerSuite(
    CallbackHandlerSuite[
        TrackingCallbackHandlerTCov, TrackingCallbackTCov, TrackingCallbackResourcesTContr
    ]
):
    @classmethod
    def build(
        cls: Type[TrackingCallbackHandlerSuiteT],
        score_callbacks: Sequence[TrackingCallbackTCov],
        array_callbacks: Sequence[TrackingCallbackTCov],
        plot_callbacks: Sequence[TrackingCallbackTCov],
        score_collection_callbacks: Sequence[TrackingCallbackTCov],
        array_collection_callbacks: Sequence[TrackingCallbackTCov],
        plot_collection_callbacks: Sequence[TrackingCallbackTCov],
    ) -> TrackingCallbackHandlerSuiteT:
        handler_suite = cls(
            score_handler=TrackingCallbackHandler(callbacks=score_callbacks),
            array_handler=TrackingCallbackHandler(callbacks=array_callbacks),
            plot_handler=TrackingCallbackHandler(callbacks=plot_callbacks),
            score_collection_handler=TrackingCallbackHandler(callbacks=score_collection_callbacks),
            array_collection_handler=TrackingCallbackHandler(callbacks=array_collection_callbacks),
            plot_collection_handler=TrackingCallbackHandler(callbacks=plot_collection_callbacks),
        )
        return handler_suite
