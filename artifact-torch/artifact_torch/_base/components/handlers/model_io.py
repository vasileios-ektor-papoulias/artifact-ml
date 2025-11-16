from typing import Any, Generic, TypeVar

from artifact_core.typing import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_torch._base.components.handlers.hook import HookCallbackHandler
from artifact_torch._base.model.base import Model
from artifact_torch._base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
CacheDataTCov = TypeVar("CacheDataTCov", bound=ArtifactResult, covariant=True)


class ModelIOCallbackHandler(
    HookCallbackHandler[Model[Any, ModelOutputTContr], CacheDataTCov],
    Generic[ModelOutputTContr, CacheDataTCov],
):
    pass


ModelIOScoreHandler = ModelIOCallbackHandler[ModelOutputTContr, Score]


ModelIOArrayHandler = ModelIOCallbackHandler[ModelOutputTContr, Array]


ModelIOPlotHandler = ModelIOCallbackHandler[ModelOutputTContr, Plot]


ModelIOScoreCollectionHandler = ModelIOCallbackHandler[ModelOutputTContr, ScoreCollection]


ModelIOArrayCollectionHandler = ModelIOCallbackHandler[ModelOutputTContr, ArrayCollection]


ModelIOPlotCollectionHandler = ModelIOCallbackHandler[ModelOutputTContr, PlotCollection]
