from collections.abc import Sequence
from typing import Any, Generic, Type, TypeVar

from artifact_experiment.spi.handler_suites import TrackingCallbackHandlerSuite

from artifact_torch._base.components.callbacks.model_io import (
    ModelIOArrayCallback,
    ModelIOArrayCollectionCallback,
    ModelIOCallback,
    ModelIOPlotCallback,
    ModelIOPlotCollectionCallback,
    ModelIOScoreCallback,
    ModelIOScoreCollectionCallback,
)
from artifact_torch._base.components.handlers.model_io import ModelIOCallbackHandler
from artifact_torch._base.components.resources.model_io import ModelIOCallbackResources
from artifact_torch._base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
ModelIOCallbackHandlerSuiteT = TypeVar(
    "ModelIOCallbackHandlerSuiteT", bound="ModelIOCallbackHandlerSuite[Any, Any]"
)


class ModelIOCallbackHandlerSuite(
    TrackingCallbackHandlerSuite[
        ModelIOCallbackHandler[ModelOutputTContr, Any],
        ModelIOCallback[ModelInputTContr, ModelOutputTContr, Any, Any],
        ModelIOCallbackResources[ModelOutputTContr],
    ],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    @classmethod
    def build(
        cls: Type[ModelIOCallbackHandlerSuiteT],
        score_callbacks: Sequence[ModelIOScoreCallback[ModelInputTContr, ModelOutputTContr]],
        array_callbacks: Sequence[ModelIOArrayCallback[ModelInputTContr, ModelOutputTContr]],
        plot_callbacks: Sequence[ModelIOPlotCallback[ModelInputTContr, ModelOutputTContr]],
        score_collection_callbacks: Sequence[
            ModelIOScoreCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
        array_collection_callbacks: Sequence[
            ModelIOArrayCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
        plot_collection_callbacks: Sequence[
            ModelIOPlotCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
    ) -> ModelIOCallbackHandlerSuiteT:
        handler_suite = cls(
            score_handler=ModelIOCallbackHandler(callbacks=score_callbacks),
            array_handler=ModelIOCallbackHandler(callbacks=array_callbacks),
            plot_handler=ModelIOCallbackHandler(callbacks=plot_callbacks),
            score_collection_handler=ModelIOCallbackHandler(callbacks=score_collection_callbacks),
            array_collection_handler=ModelIOCallbackHandler(callbacks=array_collection_callbacks),
            plot_collection_handler=ModelIOCallbackHandler(callbacks=plot_collection_callbacks),
        )
        return handler_suite

    def attach(self, resources: ModelIOCallbackResources[ModelOutputTContr]) -> bool:
        any_attached = False
        for handler in self._ls_handlers:
            any_attached |= handler.attach(resources=resources)
        return any_attached
