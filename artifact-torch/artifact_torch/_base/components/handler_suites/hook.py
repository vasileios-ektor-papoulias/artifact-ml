from collections.abc import Sequence
from typing import Any, Generic, Type, TypeVar

from artifact_experiment.spi.handler_suites import TrackingCallbackHandlerSuite

from artifact_torch._base.components.callbacks.hook import (
    HookArrayCallback,
    HookArrayCollectionCallback,
    HookCallback,
    HookPlotCallback,
    HookPlotCollectionCallback,
    HookScoreCallback,
    HookScoreCollectionCallback,
)
from artifact_torch._base.components.handlers.hook import HookCallbackHandler
from artifact_torch._base.components.resources.hook import HookCallbackResources
from artifact_torch._base.model.base import Model

ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
HookCallbackHandlerSuiteT = TypeVar(
    "HookCallbackHandlerSuiteT", bound="HookCallbackHandlerSuite[Any]"
)


class HookCallbackHandlerSuite(
    TrackingCallbackHandlerSuite[
        HookCallbackHandler[ModelTContr, Any],
        HookCallback[ModelTContr, Any, Any],
        HookCallbackResources[ModelTContr],
    ],
    Generic[ModelTContr],
):
    @classmethod
    def build(
        cls: Type[HookCallbackHandlerSuiteT],
        score_callbacks: Sequence[HookScoreCallback[ModelTContr]],
        array_callbacks: Sequence[HookArrayCallback[ModelTContr]],
        plot_callbacks: Sequence[HookPlotCallback[ModelTContr]],
        score_collection_callbacks: Sequence[HookScoreCollectionCallback[ModelTContr]],
        array_collection_callbacks: Sequence[HookArrayCollectionCallback[ModelTContr]],
        plot_collection_callbacks: Sequence[HookPlotCollectionCallback[ModelTContr]],
    ) -> HookCallbackHandlerSuiteT:
        handler_suite = cls(
            score_handler=HookCallbackHandler(callbacks=score_callbacks),
            array_handler=HookCallbackHandler(callbacks=array_callbacks),
            plot_handler=HookCallbackHandler(callbacks=plot_callbacks),
            score_collection_handler=HookCallbackHandler(callbacks=score_collection_callbacks),
            array_collection_handler=HookCallbackHandler(callbacks=array_collection_callbacks),
            plot_collection_handler=HookCallbackHandler(callbacks=plot_collection_callbacks),
        )
        return handler_suite

    def attach(self, resources: HookCallbackResources[ModelTContr]) -> bool:
        any_attached = False
        for handler in self._ls_handlers:
            any_attached |= handler.attach(resources=resources)
        return any_attached
