from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from artifact_experiment.base.callbacks.tracking import (
    ArrayCollectionExportMixin,
    ArrayExportMixin,
    PlotCollectionExportMixin,
    PlotExportMixin,
    ScoreCollectionExportMixin,
    ScoreExportMixin,
)
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray
from torch.utils.hooks import RemovableHandle

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCallbackResources,
    PeriodicTrackingCallback,
)
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput

ModelTCov = TypeVar("ModelTCov", bound=Model, covariant=True)
ModelInputTCov = TypeVar("ModelInputTCov", bound=ModelInput, covariant=True)


@dataclass
class ForwardHookCallbackResources(PeriodicCallbackResources, Generic[ModelTCov]):
    model: ModelTCov


ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
CacheDataT = TypeVar("CacheDataT")
HookResultT = TypeVar("HookResultT")


class ForwardHookCallback(
    PeriodicTrackingCallback[ForwardHookCallbackResources[ModelTContr], CacheDataT],
    Generic[ModelTContr, CacheDataT, HookResultT],
):
    _verbose = True
    _progressbar_message = "Processing Data Loader (Hooks)"

    def __init__(self, period: int, data_split: DataSplit):
        name = self._get_name()
        super().__init__(name=name, period=period, data_split=data_split)
        self._hook_results: Dict[str, List[HookResultT]] = {}
        self._handles: List[RemovableHandle] = []

    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

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
    def _aggregate(cls, hook_results: Dict[str, List[HookResultT]]) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def attach(self, resources: ForwardHookCallbackResources[ModelTContr]) -> bool:
        if self._should_trigger(step=resources.step):
            self._attach(model=resources.model)
            return True
        else:
            return False

    def _compute(self, resources: ForwardHookCallbackResources[ModelTContr]) -> CacheDataT:
        _ = resources
        result = self._finalize()
        self._detach()
        return result

    def _attach(self, model: ModelTContr):
        sink = self._hook_results

        def _wrapped_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            hook_return_value = self._hook(module, inputs, output)
            if hook_return_value is not None:
                module_name = f"{module.__class__.__name__}"
                if module_name not in sink:
                    sink[module_name] = []
                sink[module_name].append(hook_return_value)

        for module in self._get_layers(model):
            self._handles.append(module.register_forward_hook(_wrapped_hook))

    def _finalize(self) -> CacheDataT:
        result = self._aggregate(hook_results=self._hook_results)
        self._hook_results.clear()
        self._cache[self._key] = result
        return result

    def _detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class ForwardHookScoreCallback(
    ScoreExportMixin,
    ForwardHookCallback[ModelTContr, float, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]) -> float: ...


class ForwardHookArrayCallback(
    ArrayExportMixin,
    ForwardHookCallback[ModelTContr, ndarray, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]) -> ndarray: ...


class ForwardHookPlotCallback(
    PlotExportMixin,
    ForwardHookCallback[ModelTContr, Figure, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]) -> Figure: ...


class ForwardHookScoreCollectionCallback(
    ScoreCollectionExportMixin,
    ForwardHookCallback[
        ModelTContr,
        Dict[str, float],
        Dict[str, torch.Tensor],
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(
        cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, float]: ...


class ForwardHookArrayCollectionCallback(
    ArrayCollectionExportMixin,
    ForwardHookCallback[
        ModelTContr,
        Dict[str, ndarray],
        Dict[str, torch.Tensor],
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(
        cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, ndarray]: ...


class ForwardHookPlotCollectionCallback(
    PlotCollectionExportMixin,
    ForwardHookCallback[
        ModelTContr,
        Dict[str, Figure],
        Dict[str, torch.Tensor],
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(
        cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, Figure]: ...
