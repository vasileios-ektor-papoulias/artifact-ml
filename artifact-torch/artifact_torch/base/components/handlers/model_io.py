from typing import Any, Dict, Generic, TypeVar

from artifact_core._base.primitives import ArtifactResult

from artifact_torch.base.components.handlers.hook import HookCallbackHandler
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
CacheDataTCov = TypeVar("CacheDataTCov", bound=ArtifactResult, covariant=True)


class ModelIOCallbackHandler(
    HookCallbackHandler[Model[Any, ModelOutputTContr], CacheDataTCov],
    Generic[ModelOutputTContr, CacheDataTCov],
):
    pass


ModelIOScoreHandler = ModelIOCallbackHandler[ModelOutputTContr, float]


ModelIOArrayHandler = ModelIOCallbackHandler[ModelOutputTContr, Array]


ModelIOPlotHandler = ModelIOCallbackHandler[ModelOutputTContr, Figure]


ModelIOScoreCollectionHandler = ModelIOCallbackHandler[ModelOutputTContr, Dict[str, float]]


ModelIOArrayCollectionHandler = ModelIOCallbackHandler[ModelOutputTContr, Dict[str, Array]]


ModelIOPlotCollectionHandler = ModelIOCallbackHandler[ModelOutputTContr, Dict[str, Figure]]
