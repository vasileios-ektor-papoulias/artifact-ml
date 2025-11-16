from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from artifact_core.typing import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)
from torch.utils.hooks import RemovableHandle

from artifact_torch._base.components.callbacks.hook import HookCallback
from artifact_torch._base.model.base import Model

ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
CacheDataTCov = TypeVar("CacheDataTCov", bound=ArtifactResult, covariant=True)
HookResultT = TypeVar("HookResultT")


class ForwardHookCallback(
    HookCallback[ModelTContr, CacheDataTCov, HookResultT],
    Generic[ModelTContr, CacheDataTCov, HookResultT],
):
    @classmethod
    @abstractmethod
    def _get_base_key(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> Sequence[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[HookResultT]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[HookResultT]]) -> CacheDataTCov: ...

    @classmethod
    def _attach(
        cls, model: ModelTContr, sink: Dict[str, List[HookResultT]], handles: List[RemovableHandle]
    ):
        def _wrapped_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            hook_return_value = cls._hook(module, inputs, output)
            if hook_return_value is not None:
                module_name = f"{module.__class__.__name__}"
                if module_name not in sink:
                    sink[module_name] = []
                sink[module_name].append(hook_return_value)

        for module in cls._get_layers(model):
            handles.append(module.register_forward_hook(_wrapped_hook))


ForwardHookScoreCallback = ForwardHookCallback[ModelTContr, Score, Dict[str, torch.Tensor]]

ForwardHookArrayCallback = ForwardHookCallback[ModelTContr, Array, Dict[str, torch.Tensor]]


ForwardHookPlotCallback = ForwardHookCallback[ModelTContr, Plot, Dict[str, torch.Tensor]]


ForwardHookScoreCollectionCallback = ForwardHookCallback[
    ModelTContr, ScoreCollection, Dict[str, torch.Tensor]
]


ForwardHookArrayCollectionCallback = ForwardHookCallback[
    ModelTContr, ArrayCollection, Dict[str, torch.Tensor]
]


ForwardHookPlotCollectionCallback = ForwardHookCallback[
    ModelTContr, PlotCollection, Dict[str, torch.Tensor]
]
