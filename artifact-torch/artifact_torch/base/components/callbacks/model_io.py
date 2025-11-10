from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from artifact_core.base.artifact_dependencies import ArtifactResult
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.callbacks.forward_hook import ForwardHookCallback
from artifact_torch.base.components.callbacks.hook import HookCallbackResources
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelOutputTCov = TypeVar("ModelOutputTCov", bound=ModelOutput, covariant=True)


ModelIOCallbackResources = HookCallbackResources[Model[Any, ModelOutputTCov]]


ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
CacheDataT = TypeVar("CacheDataT", bound=ArtifactResult, covariant=True)
BatchResultT = TypeVar("BatchResultT")


class ModelIOCallback(
    ForwardHookCallback[Model[Any, ModelOutputTContr], CacheDataT, BatchResultT],
    Generic[ModelInputTContr, ModelOutputTContr, CacheDataT, BatchResultT],
):
    @classmethod
    @abstractmethod
    def _get_base_key(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls, model_input: ModelInputTContr, model_output: ModelOutputTContr
    ) -> BatchResultT: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(cls, ls_batch_results: List[BatchResultT]) -> CacheDataT: ...

    @classmethod
    def _get_layers(cls, model: Model[ModelInputTContr, ModelOutputTContr]) -> Sequence[nn.Module]:
        return [model]

    @classmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[BatchResultT]:
        _ = module
        return cls._compute_on_batch(model_input=inputs[0], model_output=output)

    @classmethod
    def _aggregate(cls, hook_results: Dict[str, List[BatchResultT]]) -> CacheDataT:
        ls_batch_results = next(iter(hook_results.values()), [])
        return cls._aggregate_batch_results(ls_batch_results=ls_batch_results)


ModelIOScoreCallback = ModelIOCallback[
    ModelInputTContr, ModelOutputTContr, float, Dict[str, torch.Tensor]
]


ModelIOArrayCallback = ModelIOCallback[
    ModelInputTContr, ModelOutputTContr, ndarray, Dict[str, torch.Tensor]
]


ModelIOPlotCallback = ModelIOCallback[
    ModelInputTContr, ModelOutputTContr, Figure, Dict[str, torch.Tensor]
]


ModelIOScoreCollectionCallback = ModelIOCallback[
    ModelInputTContr, ModelOutputTContr, Dict[str, float], Dict[str, torch.Tensor]
]


ModelIOArrayCollectionCallback = ModelIOCallback[
    ModelInputTContr, ModelOutputTContr, Dict[str, ndarray], Dict[str, torch.Tensor]
]


ModelIOPlotCollectionCallback = ModelIOCallback[
    ModelInputTContr, ModelOutputTContr, Dict[str, Figure], Dict[str, torch.Tensor]
]
